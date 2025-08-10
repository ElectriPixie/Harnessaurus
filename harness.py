import difflib
import requests
import traceback
import json
import re
from typing import List, Dict, Optional

from transformers import GPT2Tokenizer

from plugin_loader import load_plugin
from plugin_base import PluginBase
from result_aggregator import ResultAggregator

DEBUG = False  # Global debug flag


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def diff_texts(text1: str, text2: str) -> str:
    """Return unified diff string comparing two texts line by line."""
    d = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile='clean',
        tofile='mutated',
        lineterm=''
    )
    return ''.join(d)


def split_into_channels(text: str) -> Dict[str, str]:
    """
    Parse GPT-OSS style channel output into dict of channel_name -> content.
    Channels look like:
      <|channel|>channel_name<|message|>text here...

    Extracts text per channel until next <|channel|> or end of string.
    """
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


class GPTModel:
    def __init__(self, server_url: str, model_name: str = "llama"):
        """
        Initialize with llama-server URL and model name.
        Example: server_url="http://localhost:6589"
        """
        print(f"Using llama-server at {server_url} for model '{model_name}'")
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name

    def infer_batch(self, prompts: List[str], max_new_tokens: int = 256) -> List[str]:
        """
        Sends batch prompts to llama-server and returns list of generated texts.
        """
        results = []
        url = f"{self.server_url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        for prompt in prompts:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0,
                # You can add stop tokens if supported by your server
                # "stop": ["<|return|>", "\n\n"],
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                print("Raw response text:", response.text)
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                results.append(text)
            except Exception as e:
                print(f"Error generating text for prompt:\n{prompt}\nException: {e}")
                results.append("")

        return results


def batchify(lst: List[str], batch_size: int):
    """Yield successive batches from a list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


class PluginManager:
    def __init__(self, plugins: List[PluginBase], channel_map: Optional[Dict[str, List[str]]] = None):
        self.plugins = plugins
        self.channel_map = channel_map or {}
        debug_print(f"[DEBUG] Initialized PluginManager with channel_map: {self.channel_map}")

    def process_prompt(self, prompt: str) -> str:
        for plugin in self.plugins:
            prompt = plugin.process_prompt(prompt)
        return prompt

    def process_output(self, prompt: str, output: str, plugin_name: str) -> Optional[object]:
        debug_print(f"[DEBUG] process_output called for plugin '{plugin_name}'")

        # Log the raw output length and first 500 chars for inspection
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
            debug_print(f"[DEBUG] Sending full raw output to plugin '{plugin_name}' (length={len(text_to_send)})")

        for plugin in self.plugins:
            if type(plugin).__name__ == plugin_name:
                result = plugin.process_output(prompt, text_to_send)
                debug_print(f"[DEBUG] Plugin '{plugin_name}' process_output result: {result}")
                return result
        debug_print(f"[DEBUG] Plugin '{plugin_name}' not found among loaded plugins")
        return None

    def process_batch_output(self, prompts: List[str], outputs: List[str]) -> Dict[str, List[Optional[object]]]:
        results = {}
        for plugin in self.plugins:
            plugin_name = type(plugin).__name__
            debug_print(f"[DEBUG] Processing batch output for plugin '{plugin_name}'")
            if hasattr(plugin, 'process_batch'):
                batch_results = plugin.process_batch(prompts, outputs)
                results[plugin_name] = batch_results
                debug_print(f"[DEBUG] Plugin '{plugin_name}' batch process result count: {len(batch_results)}")
            else:
                plugin_results = []
                for p, o in zip(prompts, outputs):
                    res = self.process_output(p, o, plugin_name)
                    plugin_results.append(res)
                results[plugin_name] = plugin_results
        return results

    def process_log(self, record: dict) -> None:
        for plugin in self.plugins:
            if hasattr(plugin, 'on_log'):
                plugin.on_log(record)


def safe_plugin_result(analysis_dict, plugin_name, index, default=None):
    if plugin_name in analysis_dict:
        plugin_results = analysis_dict[plugin_name]
        if isinstance(plugin_results, list) and len(plugin_results) > index:
            return plugin_results[index]
    return default


def run_batch_test(
    prompts_batch: List[str],
    model: GPTModel,
    plugins: List[PluginBase],
    aggregator: ResultAggregator,
    channel_map: Optional[Dict[str, List[str]]] = None
) -> List[dict]:
    pm = PluginManager(plugins, channel_map=channel_map)

    clean_prompts = prompts_batch
    mutated_prompts = [pm.process_prompt(p) for p in clean_prompts]

    clean_outputs = model.infer_batch(clean_prompts)
    mutated_outputs = model.infer_batch(mutated_prompts)

    analysis_clean = pm.process_batch_output(clean_prompts, clean_outputs)
    analysis_mutated = pm.process_batch_output(mutated_prompts, mutated_outputs)

    results = []
    for i in range(len(clean_prompts)):
        try:
            diff = diff_texts(clean_outputs[i], mutated_outputs[i])

            record = {
                'original_prompt': clean_prompts[i],
                'mutated_prompt': mutated_prompts[i],
                'clean_output': clean_outputs[i],
                'mutated_output': mutated_outputs[i],
                'analysis_clean': {plugin: analysis_clean[plugin][i] for plugin in analysis_clean},
                'analysis_mutated': {plugin: analysis_mutated[plugin][i] for plugin in analysis_mutated},
                'output_diff': diff,
            }

            pm.process_log(record)
            results.append(record)
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            traceback.print_exc()

    for rec in results:
        aggregator.add_record(rec)

    return results
