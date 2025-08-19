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

    def infer_iterative_with_prompt_list(
        self,
        prompts: List[str],
        max_chunk_tokens: int = 256,
        max_iterations: int = 10,
        flip_negate: bool = False,
    ) -> str:
        """
        For each prompt in the list, iteratively feed it to the model like infer_iterative,
        then accumulate all outputs, maintaining sliding context.

        Args:
            prompts: Ordered list of prompts to feed
            max_chunk_tokens: Max tokens per server call
            max_iterations: Max iterations per prompt
            flip_negate: Whether to flip negations from previous output

        Returns:
            str: Combined output from all prompts
        """
        combined_output = ""
        accumulated_context = ""  # context across all prompts
        generated_text_set = []

        for idx, prompt in enumerate(prompts, 1):
            generated_text = ""

            for i in range(max_iterations):
                # Use sliding window context for this prompt
                recent_context = generated_text[-max(0, self.max_context_chars - len(prompt)):]
                context_to_use = accumulated_context + "\n\n" + recent_context if accumulated_context else recent_context

                current_prompt = prompt + (flip_negation(context_to_use) if flip_negate else context_to_use)

                data = self._call_server(current_prompt, max_chunk_tokens)
                chunk = data["choices"][0]["message"]["content"]
                finish_reason = data["choices"][0].get("finish_reason", "")

                generated_text += chunk

                if finish_reason != "length":
                    break

            # Append the fully generated text for this prompt to the combined output
            combined_output += generated_text + "\n\n"
            accumulated_context += "\n\n" + generated_text  # update context for next prompt
            generated_text_set.append(generated_text)

        return generated_text_set

class PluginManager:
    def __init__(self, plugins: List[PluginBase], channel_map: Optional[Dict[str, List[str]]] = None):
        self.plugins = plugins
        self.channel_map = channel_map or {}

    def process_prompt(self, prompt: str, mutation_index: int = 0, plugins_to_apply: Optional[list] = None, ret_list: bool = False, with_context: bool = False, last_plugin_name: str = None ) -> str:
        """
        Apply prompt mutators.

        Args:
            prompt: The original prompt
            plugins_to_apply: 
                - None: use self.plugins
                - empty list: apply no plugins
                - non-empty list of PluginBase instances or class-name strings: apply only these plugins

        Returns:
            str: Mutated prompt
        """
        debug_print(f"[DEBUG] Original prompt:\n{prompt}\n")
        debug_print(f"[DEBUG] plugins_to_apply: {plugins_to_apply}")

        if plugins_to_apply is None:
            active_plugins = self.plugins
            debug_print(f"[DEBUG] Using all loaded plugins: {[p.__class__.__name__ for p in active_plugins]}")
        elif all(isinstance(p, str) for p in plugins_to_apply):
            # convert class-name strings to actual plugin instances from self.plugins
            name_to_plugin = {p.__class__.__name__: p for p in self.plugins}
            active_plugins = [name_to_plugin[name] for name in plugins_to_apply if name in name_to_plugin]
            missing = [name for name in plugins_to_apply if name not in name_to_plugin]
            debug_print(f"[DEBUG] Requested plugin names: {plugins_to_apply}")
            debug_print(f"[DEBUG] Resolved plugin instances: {[p.__class__.__name__ for p in active_plugins]}")
            if missing:
                print(f"[WARNING] These plugin names were not found: {missing}")
        else:
            active_plugins = plugins_to_apply  # assume already PluginBase instances
            debug_print(f"[DEBUG] Using provided plugin instances: {[p.__class__.__name__ for p in active_plugins]}")

        last_plugin = None
        for plugin in active_plugins:
            if plugin.__class__.__name__ == last_plugin_name:
                last_plugin = plugin
                continue
            debug_print(f"[DEBUG] Applying plugin: {plugin.__class__.__name__}")
            prompt = plugin.process_prompt(prompt=prompt, mutation_index=mutation_index, ret_list=ret_list, with_context=with_context)
            debug_print(f"[DEBUG] Prompt after {plugin.__class__.__name__}:\n{prompt}\n")

        if last_plugin:
            prompt = last_plugin.process_prompt(prompt=prompt, mutation_index=mutation_index, ret_list=ret_list, with_context=with_context) 

        debug_print(f"[DEBUG] Final mutated prompt:\n{prompt}")
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

def run_model_inference(model, prompt, iterator=1, max_chunk_tokens=256, max_iterations=5, flip_negate=False):
    """
    Dispatch mutated_prompt to the correct model method based on iterator.

    Args:
        model: the model object containing the inference methods
        mutated_prompt: str or list of prompts
        iterator: chooses the method (1-4)
        max_tokens_per_chunk: max tokens for chunked methods
        max_iterations: max iterations for iterative methods
        flip_negate: whether to apply flip_negate in iterative methods

    Returns:
        The model output
    """
    if iterator == 1:
        return model.infer_single_pass(prompt=prompt)
    elif iterator == 2:
        return model.infer_iterative(
            prompt=prompt,
            max_chunk_tokens=max_chunk_tokens,
            max_iterations=max_iterations,
            flip_negate=flip_negate
        )
    elif iterator == 3:
        return model.infer_iterative_exploit(
            prompt=prompt,
            max_chunk_tokens=max_chunk_tokens,
            max_iterations=max_iterations,
            flip_negate=flip_negate
        )
    elif iterator == 4:
        return model.infer_iterative_with_prompt_list(
            prompts=prompt,
            max_chunk_tokens=max_chunk_tokens,
            max_iterations=max_iterations,
            flip_negate=flip_negate
        )
    else:
        raise ValueError(f"Unsupported iterator value: {iterator}")
    
def create_record(
    original_prompt,
    mutated_prompt,
    clean_output,
    mutated_output,
    analysis_clean,
    analysis_mutated,
    mutation_iteration,
    run_dir
):
    """
    Creates a record dictionary for a single mutation.

    Returns:
        dict
    """
    return {
        'original_prompt': original_prompt,
        'mutated_prompt': mutated_prompt,
        'clean_output': clean_output,
        'mutated_output': mutated_output,
        'analysis_clean': analysis_clean,
        'analysis_mutated': analysis_mutated,
        #'output_diff': diff,  # uncomment if needed
        'mutation_iteration': mutation_iteration,
        'run_dir': run_dir,
    }

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
    mutators: list = None,
    iterator: int = 1,
    include_mutated_output: bool = True,  # toggle mutation runs on/off
    flip_negate: bool = False,
    rerun_clean_promt: bool = False,
    run_dir: str = None,
    last_mutator_name: str = None,
    prompt_sets: bool = False
) -> List[dict]:
    pm = PluginManager(plugins, channel_map=channel_map)

    results = []
    clean_prompt = prompt

    # Run clean output always
    clean_output = run_model_inference(
        model=model,
        prompt=clean_prompt,
        iterator=iterator,
        max_chunk_tokens=max_tokens_per_chunk, 
        max_iterations=max_iterations, 
        flip_negate=flip_negate
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

    rational_mutator = ["RationalizationMutator"]
    if (len(rational_mutator)):
        name_to_plugin = {p.__class__.__name__: p for p in pm.plugins}
        plugin_name = "RationalizationMutator"
        RationalizationMutator = name_to_plugin.get(plugin_name)
        iterations=RationalizationMutator.mutations
        iterator = 4
        loop=True
        ret_list = False
        with_context = True

    for mutation_count in range(1, iterations + 1):
        if not mutators:
            mutated_prompt = pm.process_prompt(clean_prompt)
        else:
            if with_context:
                mutated_prompt_set = []
                mutated_output_set = []
                if not ret_list:
                    mutated_prompt_set = pm.process_prompt(clean_prompt, plugins_to_apply=mutators, mutation_index=mutation_count, ret_list=ret_list, with_context=with_context, last_plugin_name=last_mutator_name)
            else:
                mutated_prompt = pm.process_prompt(clean_prompt, plugins_to_apply=mutators, mutation_index=mutation_count)

        if with_context:
                mutated_output_set = run_model_inference(
                    model=model,
                    prompt=mutated_prompt_set,
                    iterator=iterator,
                    max_chunk_tokens=max_tokens_per_chunk, 
                    max_iterations=max_iterations, 
                    flip_negate=flip_negate
                )
        else:
            mutated_output = run_model_inference(
                model=model,
                prompt=mutated_prompt,
                iterator=iterator,
                max_chunk_tokens=max_tokens_per_chunk, 
                max_iterations=max_iterations, 
                flip_negate=flip_negate
            )

        try:
            analysis_mutated = {}
            if rerun_clean_promt:
                analysis_clean = {}
            if with_context:
                for idx, mutated_output in enumerate(mutated_output_set):
                    mutated_prompt = mutated_prompt_set[idx]
                    for plugin in plugins:
                        plugin_name = type(plugin).__name__
                        if rerun_clean_promt:
                            analysis_clean[plugin_name] = pm.process_output(clean_prompt, clean_output, plugin_name)
                        analysis_mutated[plugin_name] = pm.process_output(mutated_prompt, mutated_output, plugin_name)
                    record = create_record(
                        original_prompt=clean_prompt,
                        mutated_prompt=mutated_prompt,
                        clean_output=clean_output,
                        mutated_output=mutated_output,
                        analysis_clean=analysis_clean,
                        analysis_mutated=analysis_mutated,
                        mutation_iteration=mutation_count,
                        run_dir=run_dir
                    )
                    pm.process_log(record)
                    results.append(record)
                    aggregator.add_record(record)
            else:
                for plugin in plugins:
                    plugin_name = type(plugin).__name__
                    if rerun_clean_promt:
                        analysis_clean[plugin_name] = pm.process_output(clean_prompt, clean_output, plugin_name)
                    analysis_mutated[plugin_name] = pm.process_output(mutated_prompt, mutated_output, plugin_name)

                record = create_record(
                    original_prompt=clean_prompt,
                    mutated_prompt=mutated_prompt,
                    clean_output=clean_output,
                    mutated_output=mutated_output,
                    analysis_clean=analysis_clean,
                    analysis_mutated=analysis_mutated,
                    mutation_iteration=mutation_count,
                    run_dir=run_dir
                )
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