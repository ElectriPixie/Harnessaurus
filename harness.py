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

def split_into_channels(text: str) -> Dict[str, str]:
    pattern = re.compile(r"<\|channel\|>(\w+)<\|message\|>")
    matches = list(pattern.finditer(text))
    channels = {}
    for i, match in enumerate(matches):
        channel_name = match.group(1)
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
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
    def __init__(self, plugins: List[PluginBase], channel_map: Optional[Dict[str, List[str]]] = None):
        self.plugins = plugins
        self.channel_map = channel_map or {}

    def is_multi_mutator(self, plugin: PluginBase) -> bool:
        return getattr(plugin, "multi_mutator", False)

    def process_prompt(self, prompt_text: str, plugins_to_apply: Optional[list] = None, mutation_index: int = 0, last_plugin_name: Optional[str] = None) -> Prompt:
        if plugins_to_apply is None:
            active_plugins = self.plugins
        elif all(isinstance(p, str) for p in plugins_to_apply):
            name_map = {p.__class__.__name__: p for p in self.plugins}
            active_plugins = [name_map[n] for n in plugins_to_apply if n in name_map]
        else:
            active_plugins = plugins_to_apply

        prompt_list = [prompt_text]
        last_plugin = None
        for plugin in active_plugins:
            if plugin.__class__.__name__ == last_plugin_name:
                last_plugin = plugin
                continue
            result = plugin.process_prompt("\n".join(prompt_list), mutation_index=mutation_index)
            if self.is_multi_mutator(plugin):
                prompt_list = result if isinstance(result, list) else [result]
            else:
                prompt_list = [result]

        if last_plugin:
            result = last_plugin.process_prompt("\n".join(prompt_list), mutation_index=mutation_index)
            prompt_list = result if self.is_multi_mutator(last_plugin) else [result]

        return Prompt(prompt_list=prompt_list)

    def process_output(self, prompt: Prompt, output: str, plugin_name: str) -> Optional[object]:
        channels = split_into_channels(output)
        channels_for_plugin = self.channel_map.get(plugin_name)
        text_to_send = output
        if channels_for_plugin:
            text_to_send = "\n".join([channels.get(ch, "") for ch in channels_for_plugin])
        try:
            return self.plugins_by_name()[plugin_name].process_output("\n".join(prompt.prompt_list), text_to_send)
        except Exception:
            traceback.print_exc()
            return None

    def process_log(self, record: Record) -> None:
        for plugin in self.plugins:
            if hasattr(plugin, 'on_log'):
                plugin.on_log(record.to_dict())

    def plugins_by_name(self) -> Dict[str, PluginBase]:
        return {p.__class__.__name__: p for p in self.plugins}

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

def run_prompt_test(run_prompt: RunPrompt, model: GPTModel, plugins: List[PluginBase], aggregator: ResultAggregator, channel_map: Optional[Dict[str, List[str]]] = None) -> List[Record]:
    pm = PluginManager(plugins, channel_map)
    results: List[Record] = []

    if run_prompt.user_generator is not None:
        

    # Run clean output
    clean_output = run_model_inference(model, run_prompt.prompt_obj, run_prompt.iterator, run_prompt.max_tokens_per_chunk, run_prompt.max_iterations, run_prompt.flip_negate)
    for plugin in plugins:
        plugin_name = plugin.__class__.__name__
        clean_output.analysis[plugin_name] = pm.process_output(run_prompt.prompt_obj, clean_output.raw_output, plugin_name)

    #exit here if we aren't including mutated outputs
    if not run_prompt.include_mutated_output:
        record = Record(original_prompt="\n".join(run_prompt.prompt_obj.prompt_list), clean_output=clean_output)
        aggregator.add_record(record)
        results.append(record)
        pm.process_log(record)
        return results

    #The rest of the function is for use_mutated, it returns before this if it's not set

    for mutation_index in range(1, run_prompt.max_mutations + 1):
        mutated_prompt = pm.process_prompt("\n".join(run_prompt.prompt_obj.prompt_list), plugins_to_apply=run_prompt.mutators, mutation_index=mutation_index, last_plugin_name=run_prompt.last_mutator_name)
        mutated_output = run_model_inference(model, mutated_prompt, run_prompt.iterator, run_prompt.max_tokens_per_chunk, run_prompt.max_iterations, run_prompt.flip_negate)

        for plugin in plugins:
            plugin_name = plugin.__class__.__name__
            mutated_output.analysis[plugin_name] = pm.process_output(mutated_prompt, mutated_output.raw_output, plugin_name)

        record = Record(
            original_prompt="\n".join(run_prompt.prompt_obj.prompt_list),
            mutated_prompt="\n".join(mutated_prompt.prompt_list),
            clean_output=clean_output,
            mutated_output=mutated_output,
            mutation_iteration=mutation_index,
            run_dir=run_prompt.run_dir
        )

        pm.process_log(record)
        aggregator.add_record(record)
        results.append(record)

        if not run_prompt.loop:
            refusal = mutated_output.analysis.get("RefusalDetector", {})
            if refusal.get("status") == "accepted":
                break

    return results
