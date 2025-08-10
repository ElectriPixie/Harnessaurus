import difflib
import requests
import traceback
from typing import List, Dict, Optional

from plugin_loader import load_plugin
from plugin_base import PluginBase
from result_aggregator import ResultAggregator


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


class GPTModel:
    def __init__(self, server_url: str, model_name: str = "llama"):
        """
        Initialize with llama-server URL and model name.
        Example: server_url="http://localhost:6589"
        """
        print(f"Using llama-server at {server_url} for model '{model_name}'")
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name

    def infer_batch(self, prompts: List[str], max_new_tokens: int = 100) -> List[str]:
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
    def __init__(self, plugins: List[PluginBase]):
        self.plugins = plugins

    def process_prompt(self, prompt: str) -> str:
        for plugin in self.plugins:
            prompt = plugin.process_prompt(prompt)
        return prompt

    def process_output(self, prompt: str, output: str) -> Dict[str, Optional[object]]:
        results = {}
        for plugin in self.plugins:
            results[type(plugin).__name__] = plugin.process_output(prompt, output)
        return results

    def process_batch_output(self, prompts: List[str], outputs: List[str]) -> Dict[str, List[Optional[object]]]:
        results = {}
        for plugin in self.plugins:
            if hasattr(plugin, 'process_batch'):
                batch_results = plugin.process_batch(prompts, outputs)
                results[type(plugin).__name__] = batch_results
            else:
                results[type(plugin).__name__] = [
                    plugin.process_output(p, o) for p, o in zip(prompts, outputs)
                ]
        return results

    def process_log(self, record: dict) -> None:
        for plugin in self.plugins:
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
    aggregator: ResultAggregator
) -> List[dict]:
    pm = PluginManager(plugins)

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

            refusal_clean = safe_plugin_result(analysis_clean, 'RefusalDetector', i)
            refusal_mutated = safe_plugin_result(analysis_mutated, 'RefusalDetector', i)

            record = {
                'original_prompt': clean_prompts[i],
                'mutated_prompt': mutated_prompts[i],
                'clean_output': clean_outputs[i],
                'mutated_output': mutated_outputs[i],
                'analysis_clean': {plugin: analysis_clean[plugin][i] for plugin in analysis_clean},
                'analysis_mutated': {plugin: analysis_mutated[plugin][i] for plugin in analysis_mutated},
                'output_diff': diff,
                'refusal_clean': refusal_clean,
                'refusal_mutated': refusal_mutated,
            }

            pm.process_log(record)
            results.append(record)
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            traceback.print_exc()

    for rec in results:
        aggregator.add_record(rec)

    return results