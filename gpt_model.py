# gpt_model.py
import requests
import traceback
from typing import Union, List
from data_structures import Prompt, Output
from utils import flip_negation as original_flip_negation
from utils import debug_print

DEBUG = True
HARMONY_TOKEN_IDS = {200000, 200001, 200002, 200003, 200004, 200005, 200006, 200007}


def flip_negation(text: str) -> str:
    """Debuggable version of flip_negation."""
    new_text = original_flip_negation(text)
    debug_print(DEBUG, f"[flip_negation] Input (first 100 chars): {text[:100]}...\nResult (first 100 chars): {new_text[:100]}...")
    return new_text


class GPTModel:
    def __init__(self, server_url: str, model_name: str = "llama",
                 max_context_chars: int = 2000, multi_chunk: bool = False,
                 strip_harmony_tokens: bool = True):
        print(f"Using llama-server at {server_url} for model '{model_name}'")
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.multi_chunk = multi_chunk
        self.strip_harmony_tokens = strip_harmony_tokens

    def _call_server(self, prompt_text: str, max_tokens: int = 256) -> dict:
        debug_print(DEBUG, f"[call_server] Prompt length: {len(prompt_text)} chars, max_tokens={max_tokens}")
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
            return {"choices": [{"message": {"content": ""}}]}

    def _extract_text_and_channels(self, item) -> (str, dict):
        """Legacy extraction: returns raw string plus any channels if present."""
        if isinstance(item, dict):
            channels = item.get("channels")
            if channels:
                raw_output = "\n".join(str(channels[ch]) for ch in ["analysis", "commentary", "final"] if ch in channels)
                return raw_output, channels
            for key in ["final", "analysis", "text", "message"]:
                if key in item:
                    return str(item[key]), {}
            if "message" in item and "content" in item["message"]:
                return str(item["message"]["content"]), {}
        elif isinstance(item, str):
            return item, {}
        return "", {}

    def _extract_text_for_iterative(self, item) -> str:
        """Extract text for iterative inference, optionally stripping Harmony tokens."""
        text, channels = self._extract_text_and_channels(item)
        if channels and self.strip_harmony_tokens:
            final = channels.get("final", "")
            if isinstance(final, str):
                text = "".join(c for c in final if ord(c) not in HARMONY_TOKEN_IDS)
        return text

    # -------------------------
    # Inference methods
    # -------------------------

    def infer_single_pass(self, prompt: Prompt, max_tokens: int = 4096) -> Output:
        debug_print(DEBUG, f"[infer_single_pass] Running single-pass for prompt id: {id(prompt)}")
        data = self._call_server(prompt.output_text, max_tokens)
        raw_text, channels = self._extract_text_and_channels(data["choices"][0])
        output = Output(prompt=prompt, raw_output=raw_text)
        if channels:
            output.set_channels(channels)
        return output

    def infer_iterative(self, prompt: Union[str, Prompt], max_chunk_tokens: int = 256,
                        max_iterations: int = 10, flip_negate_flag: bool = False) -> str:
        """Proper iterative inference: continues if finish_reason=='length'."""
        prompt_text = prompt.output_text if isinstance(prompt, Prompt) else prompt
        generated_text = ""
        for i in range(max_iterations):
            # Build prompt with last max_context_chars of generated_text
            recent_context = generated_text[-self.max_context_chars:]
            current_prompt = prompt_text + (flip_negation(recent_context) if flip_negate_flag else recent_context)

            debug_print(DEBUG, f"[infer_iterative] Iteration {i+1}, current_prompt length: {len(current_prompt)}")
            data = self._call_server(current_prompt, max_chunk_tokens)
            chunk = self._extract_text_for_iterative(data["choices"][0])
            generated_text += chunk

            finish_reason = data["choices"][0].get("finish_reason", "")
            if finish_reason != "length" or not chunk:
                break
        return generated_text

    def infer_iterative_exploit(self, prompt: Union[str, Prompt], max_chunk_tokens: int = 256,
                                max_iterations: int = 10, flip_negate_flag: bool = False) -> str:
        """Broken-style exploit: appends entire chunk each iteration but properly respects finish_reason."""
        prompt_text = prompt.output_text if isinstance(prompt, Prompt) else prompt
        generated_text = ""
        current_prompt = prompt_text
        for i in range(max_iterations):
            debug_print(DEBUG, f"[infer_iterative_exploit] Iteration {i+1}")
            data = self._call_server(current_prompt, max_chunk_tokens)
            chunk = self._extract_text_for_iterative(data["choices"][0])
            generated_text += chunk

            finish_reason = data["choices"][0].get("finish_reason", "")
            if finish_reason != "length" or not chunk:
                break

            # Broken pattern: append whole chunk instead of pruning to last max_context_chars
            current_prompt += flip_negation(chunk) if flip_negate_flag else chunk
        return generated_text

    def infer_iterative_with_prompt_list(self, prompts: Union[Prompt, List[Prompt]],
                                         max_chunk_tokens: int = 256,
                                         max_iterations: int = 10,
                                         flip_negate_flag: bool = False) -> List[Output]:
        """Iterates over multiple prompts, continues if finish_reason=='length'."""
        all_outputs: List[Output] = []
        prompt_list = [prompts] if isinstance(prompts, Prompt) else prompts
        if not all(isinstance(p, Prompt) for p in prompt_list):
            raise TypeError(f"Expected Prompt or list of Prompts, got {type(prompts)}")

        for prompt in prompt_list:
            accumulated_context = ""
            for item in prompt.prompt_list:
                ptext, _ = self._extract_text_and_channels(item)
                ptext = ptext.strip()
                if not ptext:
                    continue

                generated_text = ""
                for i in range(max_iterations):
                    # Build context for current iteration
                    recent_context = generated_text[-self.max_context_chars:]
                    context_to_use = accumulated_context + "\n\n" + recent_context if accumulated_context else recent_context
                    current_prompt = ptext + (flip_negation(context_to_use) if flip_negate_flag else context_to_use)

                    debug_print(DEBUG, f"[infer_iterative_with_prompt_list] Iteration {i+1}, current_prompt length: {len(current_prompt)}")
                    data = self._call_server(current_prompt, max_chunk_tokens)
                    chunk = self._extract_text_for_iterative(data["choices"][0])
                    generated_text += chunk

                    finish_reason = data["choices"][0].get("finish_reason", "")
                    if finish_reason != "length" or not chunk:
                        break

                output_obj = Output(prompt=prompt, raw_output=generated_text)
                all_outputs.append(output_obj)
                accumulated_context += "\n\n" + generated_text
        return all_outputs
