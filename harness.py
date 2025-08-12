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
    def __init__(self, server_url: str, model_name: str = "llama", max_context_chars: int = 2000):
        print(f"Using llama-server at {server_url} for model '{model_name}'")
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_context_chars = max_context_chars  # rough max prompt length in chars
    
    def infer_iterative_exploit(self, prompt: str, max_chunk_tokens: int = 256, max_iterations: int = 20, flip_negotiate: int = 1,) -> str:
        """
        Generate text iteratively by feeding back output until
        the model finishes or max_iterations is reached.
        """
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        current_prompt = prompt
        full_output = ""
        for i in range(max_iterations):
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": current_prompt}],
                "max_tokens": max_chunk_tokens,
                "temperature": 0,
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                chunk = data["choices"][0]["message"]["content"]

                #print(f"[Chunk {i+1}] Generated chunk length: {len(chunk)}")
                full_output += chunk

                finish_reason = data["choices"][0].get("finish_reason", "")
                if finish_reason != "length":
                    # generation stopped naturally (not truncated)
                    #print(f"Generation finished at chunk {i+1} with reason: {finish_reason}")
                    break

                # Append the new chunk to prompt for next iteration to continue from there
                if(flip_negotiate):
                    current_prompt += flip_negation(chunk)
                else:
                    current_prompt += chunk
            except Exception as e:
                print(f"Error during iterative generation at chunk {i+1}: {e}")
                break

        #print("Raw full_output: ", full_output)

        return full_output

    def infer_iterative(self, prompt: str, max_chunk_tokens: int = 256, max_iterations: int = 10) -> str:
        """
        Generate text iteratively by feeding back output until
        the model finishes or max_iterations is reached,
        with context length trimming to simulate context shifting.
        """
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        initial_prompt = prompt
        generated_text = ""

        for i in range(max_iterations):
            # Build the prompt for this iteration:
            # Take initial prompt + last part of generated text trimmed to fit max_context_chars
            recent_context = generated_text[-(self.max_context_chars - len(initial_prompt)):]
            current_prompt = initial_prompt + recent_context

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": current_prompt}],
                "max_tokens": max_chunk_tokens,
                "temperature": 0,
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                chunk = data["choices"][0]["message"]["content"]

                #print(f"[Chunk {i+1}] Generated chunk length: {len(chunk)}")
                generated_text += chunk

                finish_reason = data["choices"][0].get("finish_reason", "")
                if finish_reason != "length":
                    # generation stopped naturally (not truncated)
                    #print(f"Generation finished at chunk {i+1} with reason: {finish_reason}")
                    break

            except Exception as e:
                print(f"Error during iterative generation at chunk {i+1}: {e}")
                break

        #print("Raw full_output: ", generated_text)

        return generated_text


class PluginManager:
    def __init__(self, plugins: List[PluginBase], channel_map: Optional[Dict[str, List[str]]] = None):
        self.plugins = plugins
        self.channel_map = channel_map or {}
        #debug_print(f"[DEBUG] Initialized PluginManager with channel_map: {self.channel_map}")

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
            debug_print(f"[DEBUG] Sending full raw output to plugin '{plugin_name}' (length={len(text_to_send)})")

        for plugin in self.plugins:
            if type(plugin).__name__ == plugin_name:
                result = plugin.process_output(prompt, text_to_send)
                debug_print(f"[DEBUG] Plugin '{plugin_name}' process_output result: {result}")
                return result
        debug_print(f"[DEBUG] Plugin '{plugin_name}' not found among loaded plugins")
        return None

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


def run_prompt_test(
    prompt: str,
    model: GPTModel,
    plugins: List[PluginBase],
    aggregator: ResultAggregator,
    channel_map: Optional[Dict[str, List[str]]] = None,
    max_tokens_per_chunk: int = 256,
    max_iterations: int = 10,
    loop: bool = True, #run_prompt_test function that loops exactly max_mutations times if loop is True
    max_mutations: int = 10,
    iterator: int = 1,
) -> List[dict]:
    pm = PluginManager(plugins, channel_map=channel_map)

    results = []
    clean_prompt = prompt

    # Run clean inference once before loop
    if(iterator == 1):
        clean_output = model.infer_iterative(
            clean_prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations
        )
    if(iterator == 2):
        clean_output = model.infer_iterative_exploit(
            clean_prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations
        )

    iterations = max_mutations if loop else 1

    for mutation_count in range(1, iterations + 1):
        mutated_prompt = pm.process_prompt(clean_prompt)
        if(iterator == 1):
            mutated_output = model.infer_iterative(
                mutated_prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations
            )
        if(iterator == 2):
            mutated_output = model.infer_iterative_exploit(
                mutated_prompt, max_chunk_tokens=max_tokens_per_chunk, max_iterations=max_iterations
            )
        try:
            diff = diff_texts(clean_output, mutated_output)

            analysis_clean = {}
            analysis_mutated = {}

            for plugin in plugins:
                plugin_name = type(plugin).__name__
                analysis_clean[plugin_name] = pm.process_output(clean_prompt, clean_output, plugin_name)
                analysis_mutated[plugin_name] = pm.process_output(mutated_prompt, mutated_output, plugin_name)

            record = {
                'original_prompt': clean_prompt,
                'mutated_prompt': mutated_prompt,
                'clean_output': clean_output,
                'mutated_output': mutated_output,
                'analysis_clean': analysis_clean,
                'analysis_mutated': analysis_mutated,
                'output_diff': diff,
                'mutation_iteration': mutation_count,
            }

            pm.process_log(record)
            results.append(record)
            aggregator.add_record(record)

        except Exception as e:
            print(f"Error processing record for prompt: {clean_prompt}\n{e}")
            traceback.print_exc()
            break

    return results
