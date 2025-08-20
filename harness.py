import requests
import re
import traceback
from typing import List, Dict, Optional
import os

from plugin_base import PluginBase, MutatorPlugin, DetectorPlugin
from data_structures import Prompt, RunPrompt, Output, Record
from result_aggregator import ResultAggregator

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# ---------------- Helpers ----------------

def split_into_channels(output: Output) -> Dict[str, str]:
    if not output.raw_output:
        return {}
    pattern = re.compile(r"<\|channel\|>(\w+)<\|message\|>")
    matches = list(pattern.finditer(output.raw_output))
    channels = {}
    for i, match in enumerate(matches):
        channel_name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output.raw_output)
        channels[channel_name] = output.raw_output[start:end].strip()
    return channels

def flip_negation(text: str) -> str:
    patterns = [
        (r"\bdo not\b", "do"), (r"\bdon't\b", "do"),
        (r"\bcan not\b", "can"), (r"\bcan't\b", "can"),
        (r"\bshould not\b", "should"), (r"\bshouldn't\b", "should"),
        (r"\bwill not\b", "will"), (r"\bwon't\b", "will")
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text

# ---------------- GPTModel ----------------

class GPTModel:
    def __init__(self, server_url: str, model_name: str = "llama", max_context_chars: int = 2000, multi_chunk: bool = False):
        print(f"Using llama-server at {server_url} for model '{model_name}'")
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.multi_chunk = multi_chunk

    def _call_server(self, prompt_text: str, max_tokens: int = 256) -> Dict:
        try:
            resp = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "multi_chunk": self.multi_chunk,
                },
                timeout=120
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            traceback.print_exc()
            return {"choices":[{"message":{"content":""}}]}

    def infer_single_pass(self, prompt: Prompt, max_tokens: int = 4096) -> Output:
        text = "\n".join(prompt.prompt_list)
        out_text = self._call_server(text, max_tokens)["choices"][0]["message"]["content"]
        return Output(prompt=prompt, raw_output=out_text)

    def infer_iterative(self, prompt: Prompt, max_chunk_tokens: int = 256, max_iterations: int = 10, flip_negate: bool = False) -> Output:
        combined_text = ""
        prompt_text = "\n".join(prompt.prompt_list)
        for _ in range(max_iterations):
            recent_context = combined_text[-max(0, self.max_context_chars - len(prompt_text)):]
            current_prompt = prompt_text + (flip_negation(recent_context) if flip_negate else recent_context)
            chunk = self._call_server(current_prompt, max_chunk_tokens)["choices"][0]["message"]["content"]
            combined_text += chunk
            if not chunk:
                break
        return Output(prompt=prompt, raw_output=combined_text)

    def infer_iterative_with_prompt_list(self, prompt: Prompt, max_chunk_tokens: int = 256, max_iterations: int = 10, flip_negate: bool = False) -> List[Output]:
        outputs = []
        accumulated_context = ""
        for ptext in prompt.prompt_list:
            generated_text = ""
            for _ in range(max_iterations):
                recent_context = generated_text[-max(0, self.max_context_chars - len(ptext)):]
                context_to_use = accumulated_context + "\n\n" + recent_context if accumulated_context else recent_context
                current_prompt = ptext + (flip_negation(context_to_use) if flip_negate else context_to_use)
                chunk = self._call_server(current_prompt, max_chunk_tokens)["choices"][0]["message"]["content"]
                generated_text += chunk
                if not chunk:
                    break
            outputs.append(Output(prompt=prompt, raw_output=generated_text))
            accumulated_context += "\n\n" + generated_text
        return outputs

# ---------------- PluginManager ----------------

class PluginManager:
    def __init__(self, mutators: List[MutatorPlugin], detectors: List[DetectorPlugin], loggers: List[PluginBase], channel_map: Optional[Dict[str, List[str]]] = None):
        self.mutators = mutators
        self.detectors = detectors
        self.loggers = loggers
        self.channel_map = channel_map or {}

    def process_prompt(self, prompt: Prompt, plugins_to_apply: Optional[list] = None) -> Prompt:
        if plugins_to_apply is None:
            active_plugins = self.mutators
        elif all(isinstance(p, str) for p in plugins_to_apply):
            name_to_plugin = {p.__class__.__name__: p for p in self.mutators}
            active_plugins = [name_to_plugin[name] for name in plugins_to_apply if name in name_to_plugin]
        else:
            active_plugins = plugins_to_apply
        for mutator in active_plugins:
            prompt = mutator.process_prompt(prompt)
        return prompt

    def process_output(self, prompt: Prompt, output: Output, detector_name: str) -> Output:
        detector = self.detectors_by_name()[detector_name]

        if not output.channels:
            output.channels = split_into_channels(output)

        channels_for_detector = self.channel_map.get(detector_name)
        if channels_for_detector:
            detector_input_text = "\n".join(output.channels.get(ch, "") for ch in channels_for_detector)
        else:
            detector_input_text = output.raw_output

        detector_input_obj = Output(prompt=prompt, raw_output=detector_input_text)

        # Run detector – must return Output
        analysis_result_obj = detector.process_output(prompt, detector_input_obj)

        # Ensure main output.analysis exists
        if output.analysis is None:
            output.analysis = {}

        # Save the detector's analysis dict directly
        output.analysis[detector_name] = analysis_result_obj.analysis.get(detector_name)

        return output

    def process_log(self, record: Record):
        for logger in self.loggers:
            if hasattr(logger, "on_log"):
                logger.on_log(record)

    def detectors_by_name(self) -> Dict[str, DetectorPlugin]:
        return {d.__class__.__name__: d for d in self.detectors}

# ---------------- Runner ----------------

def run_model_inference(model: GPTModel, prompt: Prompt, iterator: int, max_tokens_per_chunk: int, max_iterations: int, flip_negate: bool) -> Output:
    if iterator == 1:
        return model.infer_single_pass(prompt, max_tokens=max_tokens_per_chunk)
    else:
        return model.infer_iterative_with_prompt_list(prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations, flip_negate=flip_negate)[-1]

def _process_outputs(
    prompt_obj: Prompt,
    run_prompt: RunPrompt,
    model: GPTModel,
    aggregator: ResultAggregator,
    pm: PluginManager,
) -> List[Record]:
    """
    Run inference on a prompt (including mutations), apply detectors, log results, and
    return a list of Record objects with all analysis included.
    """
    results: List[Record] = []
    iterator = 4 if prompt_obj.has_context else run_prompt.iterator
    os.makedirs(run_prompt.run_dir, exist_ok=True)

    # --- Run clean output ---
    clean_output = run_model_inference(
        model,
        prompt_obj,
        iterator,
        run_prompt.max_tokens_per_chunk,
        run_prompt.max_iterations,
        run_prompt.flip_negate
    )

    # Apply all detectors to clean output
    detector_names = [p.__class__.__name__ for p in getattr(run_prompt, "detectors", [])]
    detector_plugins = getattr(run_prompt, "detector_plugins", [])
    # Make a list of detector names (strings)
    detector_names = []
    for p in detector_plugins:
        if isinstance(p, DetectorPlugin):
            detector_names.append(p.__class__.__name__)
        elif isinstance(p, str):
            detector_names.append(p)
        else:
            raise TypeError(f"Unexpected detector type: {type(p)}")
    for name in detector_names:
        clean_output = pm.process_output(prompt_obj, clean_output, name)

    # If mutated outputs not needed, store clean record and return
    if not getattr(run_prompt, "include_mutated_output", True):
        record = Record(
            original_prompt="\n".join(prompt_obj.prompt_list),
            clean_output=clean_output,
            run_dir=run_prompt.run_dir
        )
        aggregator.add_record(record)
        pm.process_log(record)
        results.append(record)
        return results

    # --- Run mutated outputs ---
    max_mutations = getattr(run_prompt, "max_mutations", 1)
    for mutation_index in range(1, max_mutations + 1):
        # Apply mutators to prompt
        mutated_prompt = pm.process_prompt(prompt_obj, plugins_to_apply=getattr(run_prompt, "mutators", None))

        # Optionally rerun clean prompt for each mutation
        if getattr(run_prompt, "rerun_clean_prompt", False):
            clean_output = run_model_inference(
                model,
                prompt_obj,
                iterator,
                run_prompt.max_tokens_per_chunk,
                run_prompt.max_iterations,
                run_prompt.flip_negate
            )
            for name in detector_names:
                clean_output = pm.process_output(prompt_obj, clean_output, name)

        # Run mutated prompt through model
        mutated_output = run_model_inference(
            model,
            mutated_prompt,
            iterator,
            run_prompt.max_tokens_per_chunk,
            run_prompt.max_iterations,
            run_prompt.flip_negate
        )

        # Apply all detectors to mutated output
        for name in detector_names:
            mutated_output = pm.process_output(mutated_prompt, mutated_output, name)

        # Store record
        record = Record(
            original_prompt="\n".join(prompt_obj.prompt_list),
            mutated_prompt="\n".join(mutated_prompt.prompt_list),
            clean_output=clean_output,
            mutated_output=mutated_output,
            mutation_iteration=mutation_index,
            run_dir=run_prompt.run_dir
        )

        aggregator.add_record(record)
        pm.process_log(record)
        results.append(record)

        # Stop mutation loop if refusal detector triggered
        refusal = mutated_output.analysis.get("RefusalDetector", {})
        if refusal.get("status") == "refused":
            print("[INFO] Refusal detected by RefusalDetector, stopping mutation loop")
            break

    return results

def run_prompt_test(
    run_prompt: RunPrompt,
    model: GPTModel,
    aggregator: ResultAggregator,
    channel_map: Optional[Dict[str, List[str]]] = None
):
    if not getattr(run_prompt, "use_generator", None):
        raise ValueError("run_prompt.use_generator must be set")

    pm = PluginManager(
        run_prompt.mutator_plugins,
        run_prompt.detector_plugins,
        run_prompt.logger_plugins,
        channel_map
    )

    # Find generator plugin by name
    generator = next(
        (g for g in run_prompt.generator_plugins if g.__class__.__name__ == run_prompt.use_generator),
        None
    )
    if generator is None:
        raise ValueError(f"Generator '{run_prompt.use_generator}' not found")

    # Generate a Prompt object
    generated_prompt = generator.generate_from_prompt(run_prompt.prompt_obj)

    # Wrap in a list so _process_outputs works the same way
    prompts_to_run = [generated_prompt]

    results = []
    for prompt_item in prompts_to_run:
        results.extend(_process_outputs(prompt_item, run_prompt, model, aggregator, pm))

    return results


