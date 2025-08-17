import difflib
import requests
import traceback
import json
import re
from typing import List, Dict, Optional
from plugin_loader import load_plugin
from plugin_base import PluginBase
from result_aggregator import ResultAggregator

DEBUG = False  # Global debug flag

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def diff_texts(text1: str, text2: str) -> str:
    d = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile='clean',
        tofile='mutated',
        lineterm=''
    )
    return ''.join(d)

def split_into_channels(text: str) -> Dict[str, str]:
    pattern = re.compile(r"<\|channel\|>(\w+)<\|message\|>")
    matches = list(pattern.finditer(text))
    channels = {}
    for i, match in enumerate(matches):
        channel_name = match.group(1)
        start = match.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        channels[channel_name] = content
    return channels

def flip_negation(text):
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
            print(f"[flip_negation] Replaced '{pat}' with '{repl}' {count} time(s). \nOld Text: {text}\nResult: {new_text}")
            text = new_text
    return text

class GPTModel:
    def __init__(self, server_url: str, model_name: str = "llama", max_context_chars: int = 2000, multi_chunk: bool = False):
        print(f"Using llama-server at {server_url} for model '{model_name}'")
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.multi_chunk = multi_chunk  # server-side chunking flag
    
    def _call_server(self, prompt: str, max_tokens: int = 256) -> Dict:
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "multi_chunk": self.multi_chunk,  # inform server
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def infer_single_pass(self, prompt: str, max_tokens: int = 4096) -> str:
        """Single-pass generation; server handles full text in one call"""
        data = self._call_server(prompt, max_tokens)
        return data["choices"][0]["message"]["content"]

    def infer_iterative(self, prompt: str, max_chunk_tokens: int = 256, max_iterations: int = 10, flip_negate: bool = False) -> str:
        """Harness-side iterative chunking with sliding context"""
        generated_text = ""

        for i in range(max_iterations):
            # Safe sliding window for recent context
            recent_context = generated_text[-max(0, self.max_context_chars - len(prompt)):]

            # Always include the original prompt; optionally flip recent context
            current_prompt = prompt + (flip_negation(recent_context) if flip_negate else recent_context)

            # Call the server for the next chunk
            data = self._call_server(current_prompt, max_chunk_tokens)
            chunk = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason", "")

            generated_text += chunk

            # Stop if model finished naturally
            if finish_reason != "length":
                break

        return generated_text

    def infer_iterative_exploit(self, prompt: str, max_chunk_tokens: int = 256, max_iterations: int = 10, flip_negate: bool = False) -> str:
        """Iterative with optional negation flipping"""
        generated_text = ""
        current_prompt = prompt

        for i in range(max_iterations):
            data = self._call_server(current_prompt, max_chunk_tokens)
            chunk = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason", "")

            generated_text += chunk

            if finish_reason != "length":
                break

            current_prompt += flip_negation(chunk) if flip_negate else chunk

        return generated_text

class PluginManager:
    def __init__(self, plugins: List[PluginBase], channel_map: Optional[Dict[str, List[str]]] = None):
        self.plugins = plugins
        self.channel_map = channel_map or {}

    def process_prompt(self, prompt: str) -> str:
        for plugin in self.plugins:
            prompt = plugin.process_prompt(prompt)
        return prompt

    def process_output(self, prompt: str, output: str, plugin_name: str) -> Optional[object]:
        debug_print(f"[DEBUG] process_output called for plugin '{plugin_name}'")

        debug_print(f"[DEBUG] Raw output length: {len(output)}")
        debug_print(f"[DEBUG] Raw output preview (first 500 chars):\n{output[:500]!r}")

        channels = split_into_channels(output)
        debug_print(f"[DEBUG] split_into_channels found channels: {list(channels.keys())}")

        channels_for_plugin = self.channel_map.get(plugin_name)
        if channels_for_plugin is not None:
            debug_print(f"[DEBUG] channels_for_plugin for '{plugin_name}': {channels_for_plugin}")
        else:
            debug_print(f"[DEBUG] No specific channels configured for plugin '{plugin_name}', sending full output")

        if channels_for_plugin:
            selected_texts = []
            for ch in channels_for_plugin:
                ch_text = channels.get(ch, "")
                selected_texts.append(ch_text)
                debug_print(f"[DEBUG] Channel '{ch}' content for plugin '{plugin_name}': '{ch_text[:100]}...'")
            text_to_send = "\n".join(selected_texts)
        else:
            text_to_send = output
            debug_print(f"[DEBUG] Sending full raw output to plugin '{plugin_name}'")

        try:
            return self.plugins_by_name()[plugin_name].process_output(prompt, text_to_send)
        except Exception:
            print(f"[Error] Plugin '{plugin_name}' process_output failed:\n{traceback.format_exc()}")
            return None

    def process_log(self, record: dict) -> None:
        for plugin in self.plugins:
            if hasattr(plugin, 'on_log'):
                plugin.on_log(record)

    def plugins_by_name(self) -> Dict[str, PluginBase]:
        return {plugin.__class__.__name__: plugin for plugin in self.plugins}

def run_prompt_test(
    prompt: str,
    model: GPTModel,
    plugins: List[PluginBase],
    aggregator: ResultAggregator,
    channel_map: Optional[Dict[str, List[str]]] = None,
    max_tokens_per_chunk: int = 256,
    max_iterations: int = 10,
    loop: bool = True,                # if True: run max_mutations times, else once or until accepted
    max_mutations: int = 1,
    iterator: int = 1,
    include_mutated_output: bool = True,  # toggle mutation runs on/off
    flip_negate: bool = False,
    rerun_clean_promt: bool = False,
    run_dir: str = None,
) -> List[dict]:
    pm = PluginManager(plugins, channel_map=channel_map)

    results = []
    clean_prompt = prompt

    # Run clean output always
    if iterator == 1:
        clean_output = model.infer_single_pass(
            clean_prompt
        )
    elif iterator == 2:
        clean_output = model.infer_iterative(
            clean_prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations, flip_negate=flip_negate
        )
    elif iterator == 3:
        clean_output = model.infer_iterative_exploit(
            clean_prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations, flip_negate=flip_negate
        )

    if not include_mutated_output:
        # Only analyze clean output, no mutation runs
        analysis_clean = {}
        for plugin in plugins:
            plugin_name = type(plugin).__name__
            analysis_clean[plugin_name] = pm.process_output(clean_prompt, clean_output, plugin_name)

        record = {
            'original_prompt': clean_prompt,
            'mutated_prompt': '',
            'clean_output': clean_output,
            'mutated_output': '',
            'analysis_clean': analysis_clean,
            'analysis_mutated': {},
            #'output_diff': '',
            'mutation_iteration': 0,
            'run_dir': run_dir,
        }
        aggregator.add_record(record)
        results.append(record)
        return results

    iterations = max_mutations
    if not rerun_clean_promt:
        analysis_clean = {}
        for plugin in plugins:
            plugin_name = type(plugin).__name__
            analysis_clean[plugin_name] = pm.process_output(clean_prompt, clean_output, plugin_name)

    for mutation_count in range(1, iterations + 1):
        mutated_prompt = pm.process_prompt(clean_prompt)

        if iterator == 1:
            mutated_output = model.infer_single_pass(
                mutated_prompt
            )
        elif iterator == 2:
            mutated_output = model.infer_iterative(
                mutated_prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations, flip_negate=flip_negate
            )
        elif iterator == 3:
            mutated_output = model.infer_iterative_exploit(
                mutated_prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations, flip_negate=flip_negate
            )

        try:
            analysis_mutated = {}
            if rerun_clean_promt:
                analysis_clean = {}

            for plugin in plugins:
                plugin_name = type(plugin).__name__
                if rerun_clean_promt:
                    analysis_clean[plugin_name] = pm.process_output(clean_prompt, clean_output, plugin_name)
                analysis_mutated[plugin_name] = pm.process_output(mutated_prompt, mutated_output, plugin_name)

            record = {
                'original_prompt': clean_prompt,
                'mutated_prompt': mutated_prompt,
                'clean_output': clean_output,
                'mutated_output': mutated_output,
                'analysis_clean': analysis_clean,
                'analysis_mutated': analysis_mutated,
                #'output_diff': diff,
                'mutation_iteration': mutation_count,
            }

            pm.process_log(record)
            results.append(record)
            aggregator.add_record(record)

            # If loop==False, break early if RefusalDetector accepted
            if not loop:
                refusal_result = analysis_mutated.get('RefusalDetector', {})
                if refusal_result.get('status') == 'accepted':
                    print(f"[run_prompt_test] RefusalDetector accepted at mutation {mutation_count}, breaking loop.")
                    break

        except Exception as e:
            print(f"Error processing record for prompt: {clean_prompt}\n{e}")
            traceback.print_exc()
            break

    return results