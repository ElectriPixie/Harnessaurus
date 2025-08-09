import difflib
from llama_cpp import Llama

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
    def __init__(self, model_path: str):
        print(f"Loading llama.cpp model from {model_path}")
        self.llm = Llama(model_path=model_path)

    def infer_batch(self, prompts: list[str], max_new_tokens: int = 100) -> list[str]:
        results = []
        for prompt in prompts:
            output = self.llm(prompt, max_tokens=max_new_tokens, stop=None)
            # Extract generated text from response
            text = output['choices'][0]['text']
            results.append(text)
        return results


def batchify(lst: list, batch_size: int):
    """Yield successive batches from a list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def run_batch_test(
    prompts_batch: list[str],
    model: GPTModel,
    plugins: list[PluginBase],
    aggregator: ResultAggregator
) -> list[dict]:
    """
    Run inference and plugin analysis on batches of prompts.
    Returns list of result records.
    """

    class PluginManager:
        def __init__(self, plugins: list[PluginBase]):
            self.plugins = plugins

        def process_prompt(self, prompt: str) -> str:
            for plugin in self.plugins:
                prompt = plugin.process_prompt(prompt)
            return prompt

        def process_output(self, prompt: str, output: str) -> dict:
            results = {}
            for plugin in self.plugins:
                results[type(plugin).__name__] = plugin.process_output(prompt, output)
            return results

        def process_batch_output(self, prompts: list[str], outputs: list[str]) -> dict:
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

    pm = PluginManager(plugins)

    clean_prompts = prompts_batch
    mutated_prompts = [pm.process_prompt(p) for p in clean_prompts]

    clean_outputs = model.infer_batch(clean_prompts)
    mutated_outputs = model.infer_batch(mutated_prompts)

    analysis_clean = pm.process_batch_output(clean_prompts, clean_outputs)
    analysis_mutated = pm.process_batch_output(mutated_prompts, mutated_outputs)

    results = []
    for i in range(len(clean_prompts)):
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

    for rec in results:
        aggregator.add_record(rec)

    return results