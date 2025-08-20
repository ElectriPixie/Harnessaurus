# harness.py
import difflib
import requests
import traceback
import re
from typing import List, Dict, Optional
from plugin_loader import load_plugin
from plugin_base import PluginBase
from result_aggregator import ResultAggregator
from data_structures import Prompt, RunPrompt, Output, Record

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def split_into_channels(output: Output) -> Dict[str, str]:
    """
    Extracts channel-separated text from an Output object.

    Args:
        output: Output instance containing raw_output.

    Returns:
        Dict mapping channel names to their respective text.
    """
    text = output.raw_output
    pattern = re.compile(r"<\|channel\|>(\w+)<\|message\|>")
    matches = list(pattern.finditer(text))
    channels = {}
    for i, match in enumerate(matches):
        channel_name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        channels[channel_name] = text[start:end].strip()
    return channels

def flip_negation(text: str) -> str:
    patterns = [
        (r"\bdo not\b", "do"),
        (r"\bdon't\b", "do"),
        (r"\bcan not\b", "can"),
        (r"\bcan't\b", "can"),
        (r"\bshould not\b", "should"),
        (r"\bshouldn't\b", "should"),
        (r"\bwill not\b", "will"),
        (r"\bwon't\b", "will"),
    ]
    for pat, repl in patterns:
        new_text, count = re.subn(pat, repl, text, flags=re.IGNORECASE)
        if count > 0:
            debug_print(f"[flip_negation] Replaced '{pat}' with '{repl}' {count} time(s).")
            text = new_text
    return text

class GPTModel:
    def __init__(self, server_url: str, model_name: str = "llama", max_context_chars: int = 2000, multi_chunk: bool = False):
        print(f"Using llama-server at {server_url} for model '{model_name}'")
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.multi_chunk = multi_chunk

    def _call_server(self, prompt_text: str, max_tokens: int = 256) -> Dict:
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "multi_chunk": self.multi_chunk,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def infer_single_pass(self, prompt: Prompt, max_tokens: int = 4096) -> Output:
        text = "\n".join(prompt.prompt_list)
        resp = self._call_server(text, max_tokens)
        out_text = resp["choices"][0]["message"]["content"]
        return Output(prompt=prompt, raw_output=out_text)

    def infer_iterative(self, prompt: Prompt, max_chunk_tokens: int = 256, max_iterations: int = 10, flip_negate: bool = False) -> Output:
        combined_text = ""
        prompt_text = "\n".join(prompt.prompt_list)
        for _ in range(max_iterations):
            recent_context = combined_text[-max(0, self.max_context_chars - len(prompt_text)):]
            current_prompt = prompt_text + (flip_negation(recent_context) if flip_negate else recent_context)
            resp = self._call_server(current_prompt, max_chunk_tokens)
            chunk = resp["choices"][0]["message"]["content"]
            finish_reason = resp["choices"][0].get("finish_reason", "")
            combined_text += chunk
            if finish_reason != "length":
                break
        return Output(prompt=prompt, raw_output=combined_text)

    def infer_iterative_exploit(self, prompt: Prompt, max_chunk_tokens: int = 256, max_iterations: int = 10, flip_negate: bool = False) -> Output:
        combined_text = ""
        current_prompt = "\n".join(prompt.prompt_list)
        for _ in range(max_iterations):
            resp = self._call_server(current_prompt, max_chunk_tokens)
            chunk = resp["choices"][0]["message"]["content"]
            finish_reason = resp["choices"][0].get("finish_reason", "")
            combined_text += chunk
            if finish_reason != "length":
                break
            current_prompt += flip_negation(chunk) if flip_negate else chunk
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
                resp = self._call_server(current_prompt, max_chunk_tokens)
                chunk = resp["choices"][0]["message"]["content"]
                finish_reason = resp["choices"][0].get("finish_reason", "")
                generated_text += chunk
                if finish_reason != "length":
                    break
            outputs.append(Output(prompt=prompt, raw_output=generated_text))
            accumulated_context += "\n\n" + generated_text
        return outputs

class PluginManager:
    def __init__(
        self,
        mutators: List[MutatorPlugin],
        detectors: List[DetectorPlugin],
        loggers: List[BasePlugin],
        channel_map: Optional[Dict[str, List[str]]] = None
    ):
        self.mutators = mutators
        self.detectors = detectors
        self.loggers = loggers
        self.channel_map = channel_map or {}

    # --- Mutator interface ---
    def process_prompt(self, prompt: Prompt, plugins_to_apply: Optional[list] = None) -> Prompt:
        """
        Apply all active mutators to a Prompt or PromptSet.

        Args:
            prompt: Prompt or PromptSet object to mutate.
            plugins_to_apply:
                - None: use all loaded mutators
                - empty list: apply no mutators
                - non-empty list of mutator instances or names: apply only these

        Returns:
            Prompt or PromptSet after applying all mutators.
        """
        # Determine active mutators
        if plugins_to_apply is None:
            active_plugins = self.mutators
        elif all(isinstance(p, str) for p in plugins_to_apply):
            name_to_plugin = {p.__class__.__name__: p for p in self.mutators}
            active_plugins = [name_to_plugin[name] for name in plugins_to_apply if name in name_to_plugin]
            missing = [name for name in plugins_to_apply if name not in name_to_plugin]
            if missing:
                print(f"[WARNING] These mutator names were not found: {missing}")
        else:
            active_plugins = plugins_to_apply  # assume instances

        # Apply each active mutator in order
        for mutator in active_plugins:
            prompt = mutator.process_prompt_set(prompt)

        return prompt

    # --- Detector interface ---
    def process_output(self, prompt: Prompt, output: Output, detector_name: str) -> Optional[Output]:
        """
        Apply a single detector by name to the given Prompt and Output.
        Supports channel mapping and stores channels in the Output object.
        """
        try:
            detector = self.detectors_by_name()[detector_name]

            # Ensure Output.channels is populated
            if not output.channels:
                output.channels = split_into_channels(output)

            # Select relevant channels if channel_map specifies
            channels_for_detector = self.channel_map.get(detector_name)
            if channels_for_detector:
                output_to_pass = {ch: output.channels.get(ch, "") for ch in channels_for_detector}
            else:
                output_to_pass = output.raw_output  # full text if no channels mapped

            # Apply detector
            analysis_result = detector.process_output(prompt, output_to_pass)
            output.analysis[detector_name] = analysis_result

            return output

        except Exception:
            traceback.print_exc()
            return None

    # --- Logger interface ---
    def process_log(self, record: Record) -> None:
        for logger in self.loggers:
            if hasattr(logger, "on_log"):
                logger.on_log(record)

    # --- Helpers ---
    def mutators_by_name(self) -> Dict[str, MutatorPlugin]:
        return {m.__class__.__name__: m for m in self.mutators}

    def detectors_by_name(self) -> Dict[str, DetectorPlugin]:
        return {d.__class__.__name__: d for d in self.detectors}

    def loggers_by_name(self) -> Dict[str, BasePlugin]:
        return {l.__class__.__name__: l for l in self.loggers}

def run_model_inference(model: GPTModel, prompt: Prompt, iterator: int, max_tokens_per_chunk: int, max_iterations: int, flip_negate: bool) -> Output:
    if iterator == 1:
        return model.infer_single_pass(prompt, max_tokens=max_tokens_per_chunk)
    elif iterator == 2:
        return model.infer_iterative(prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations, flip_negate=flip_negate)
    elif iterator == 3:
        return model.infer_iterative_exploit(prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations, flip_negate=flip_negate)
    elif iterator == 4:
        outputs = model.infer_iterative_with_prompt_list(prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations, flip_negate=flip_negate)
        combined_text = "\n\n".join(o.raw_output for o in outputs)
        return Output(prompt=prompt, raw_output=combined_text)
    else:
        raise ValueError(f"Unsupported iterator: {iterator}")

def get_generator_by_name(run_prompt, generator_name: str):
    """
    Return the generator instance from run_prompt.generators matching the class name.
    """
    for gen in getattr(run_prompt, "generators", []):
        if gen.__class__.__name__ == generator_name:
            return gen
    return None


# --- _process_outputs ---
def _process_outputs(
    prompt_obj: Prompt,
    run_prompt: RunPrompt,
    model: GPTModel,
    aggregator: ResultAggregator,
    pm: PluginManager,
) -> List[Record]:
    results: List[Record] = []
    iterator = run_prompt.iterator
    if prompt_obj.has_context:
        iterator = 4 

    # Run clean output
    clean_output = run_model_inference(
        model,
        prompt_obj,
        iterator,
        run_prompt.max_tokens_per_chunk,
        run_prompt.max_iterations,
        run_prompt.flip_negate
    )

    # Analyze clean output with detectors
    for plugin in run_prompt.detectors:
        plugin_name = plugin.__class__.__name__
        pm.process_output(prompt_obj, clean_output, plugin_name)  # pass full Output object

    # Exit early if mutated outputs are not requested
    if not getattr(run_prompt, "include_mutated_output", True):
        record = Record(
            original_prompt="\n".join(prompt_obj.prompt_list),
            clean_output=clean_output
        )
        aggregator.add_record(record)
        pm.process_log(record)
        results.append(record)
        return results

    # Loop over mutations
    for mutation_index in range(1, getattr(run_prompt, "max_mutations", 1) + 1):
        mutated_prompt = pm.process_prompt(prompt_obj)

        if getattr(run_prompt, "rerun_clean_prompt", False):
            clean_output = run_model_inference(
                model,
                prompt_obj,
                iterator,
                run_prompt.max_tokens_per_chunk,
                run_prompt.max_iterations,
                run_prompt.flip_negate
            )
            for plugin in run_prompt.detectors:
                pm.process_output(prompt_obj, clean_output, plugin.__class__.__name__)

        mutated_output = run_model_inference(
            model,
            mutated_prompt,
            iterator,
            run_prompt.max_tokens_per_chunk,
            run_prompt.max_iterations,
            run_prompt.flip_negate
        )

        # Apply detectors to mutated output
        for plugin in run_prompt.detectors:
            pm.process_output(mutated_prompt, mutated_output, plugin.__class__.__name__)

        record = Record(
            original_prompt="\n".join(prompt_obj.prompt_list),
            mutated_prompt="\n".join(mutated_prompt.prompt_list),
            clean_output=clean_output,
            mutated_output=mutated_output,
            mutation_iteration=mutation_index,
            run_dir=run_prompt.run_dir
        )

        pm.process_log(record)
        aggregator.add_record(record)
        results.append(record)

        # Stop loop if refusal detector signals acceptance
        if not getattr(run_prompt, "loop", True):
            refusal = mutated_output.analysis.get("RefusalDetector", {})
            if refusal.get("status") == "accepted":
                break

    return results


def run_prompt_test(
    run_prompt: RunPrompt,
    model: GPTModel,
    aggregator: ResultAggregator,
    channel_map: Optional[Dict[str, List[str]]] = None
) -> List[Record]:
    if not getattr(run_prompt, "use_generator", None):
        raise ValueError("run_prompt.use_generator must be set")

    pm = PluginManager(
        mutators=run_prompt.mutators,
        detectors=run_prompt.detectors,
        loggers=run_prompt.loggers,
        channel_map=channel_map
    )

    results: List[Record] = []

    generator = get_generator_by_name(run_prompt, run_prompt.use_generator)
    if generator is None:
        raise ValueError(f"Generator '{run_prompt.use_generator}' not found")

    # The generator returns a Prompt (or PromptSet), wrap as list
    generated = generator.generate_from_prompt(run_prompt.prompt_obj)
    prompts_to_run: List[Prompt] = [generated]

    for prompt_item in prompts_to_run:
        results.extend(_process_outputs(prompt_item, run_prompt, model, aggregator, pm))

    return results