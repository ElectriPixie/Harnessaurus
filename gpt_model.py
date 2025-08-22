# gpt_model.py
import requests
import traceback
from typing import Union, List
from data_structures import Prompt, Output
from utils import flip_negation as original_flip_negation
from utils import debug_print

DEBUG = False

def flip_negation(text: str) -> str:
    """Debuggable version of flip_negation."""
    new_text = original_flip_negation(text)
    debug_print(DEBUG, f"[flip_negation] Input (first 100 chars): {text[:100]}...\nResult (first 100 chars): {new_text[:100]}...")
    return new_text

class GPTModel:
    def __init__(self, server_url: str, model_name: str = "llama", max_context_chars: int = 2000, multi_chunk: bool = False):
        print(f"Using llama-server at {server_url} for model '{model_name}'")
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.multi_chunk = multi_chunk

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
            data = resp.json()
            debug_print(DEBUG, f"[call_server] Received chunk length: {len(self._get_item_text(data['choices'][0]))}")
            return data
        except Exception:
            traceback.print_exc()
            return {"choices": [{"message": {"content": ""}}]}

    def _get_item_text(self, item):
        """Extract text or preserve dict from Harmony or legacy response."""
        if isinstance(item, dict):
            if "channels" in item:
                return item["channels"]  # Preserve Harmony dict directly
            # fallback to other keys
            for key in ["final", "analysis", "text", "message"]:
                if key in item:
                    return item[key]
            if "message" in item and "content" in item["message"]:
                return item["message"]["content"]
        elif isinstance(item, str):
            return item
        return ""

    def infer_single_pass(self, prompt: Prompt, max_tokens: int = 4096) -> Output:
        debug_print(DEBUG, f"[infer_single_pass] Running single-pass for prompt id: {id(prompt)}")
        data = self._call_server(prompt.output_text, max_tokens)
        return Output(prompt=prompt, raw_output=self._get_item_text(data["choices"][0]))

    def infer_iterative(self, prompt: str, max_chunk_tokens: int = 256, max_iterations: int = 10, flip_negate_flag: bool = False) -> str:
        generated_text = ""
        for i in range(max_iterations):
            recent_context = generated_text[-max(0, self.max_context_chars - len(prompt)):]
            current_prompt = prompt + (flip_negation(recent_context) if flip_negate_flag else recent_context)
            debug_print(DEBUG, f"[infer_iterative] Iteration {i+1}, current_prompt length: {len(current_prompt)}")
            data = self._call_server(current_prompt, max_chunk_tokens)
            chunk = self._get_item_text(data["choices"][0])
            finish_reason = data["choices"][0].get("finish_reason", "")
            generated_text += chunk
            if finish_reason != "length":
                break
        return generated_text

    def infer_iterative_exploit(self, prompt: str, max_chunk_tokens: int = 256, max_iterations: int = 10, flip_negate_flag: bool = False) -> str:
        generated_text = ""
        current_prompt = prompt
        for i in range(max_iterations):
            debug_print(DEBUG, f"[infer_iterative_exploit] Iteration {i+1}")
            data = self._call_server(current_prompt, max_chunk_tokens)
            chunk = self._get_item_text(data["choices"][0])
            finish_reason = data["choices"][0].get("finish_reason", "")
            generated_text += chunk
            if finish_reason != "length":
                break
            current_prompt += flip_negation(chunk) if flip_negate_flag else chunk
        return generated_text

    def infer_iterative_with_prompt_list(self, prompts: Union[Prompt, List[Prompt]], max_chunk_tokens: int = 256, max_iterations: int = 10, flip_negate_flag: bool = False) -> List[Output]:
        all_outputs: List[Output] = []
        if isinstance(prompts, Prompt):
            prompt_list = [prompts]
        elif isinstance(prompts, list) and all(isinstance(p, Prompt) for p in prompts):
            prompt_list = prompts
        else:
            raise TypeError(f"Expected Prompt or list of Prompts, got {type(prompts)}")

        for prompt in prompt_list:
            accumulated_context = ""
            for item in prompt.prompt_list:
                ptext = self._get_item_text(item).strip()
                if not ptext:
                    continue
                generated_text = ""
                for i in range(max_iterations):
                    recent_context = generated_text[-max(0, self.max_context_chars - len(ptext)):]
                    context_to_use = accumulated_context + "\n\n" + recent_context if accumulated_context else recent_context
                    current_prompt = ptext + (flip_negation(context_to_use) if flip_negate_flag else context_to_use)
                    debug_print(DEBUG, f"[infer_iterative_with_prompt_list] Iteration {i+1}, current_prompt length: {len(current_prompt)}")
                    data = self._call_server(current_prompt, max_chunk_tokens)
                    chunk = self._get_item_text(data["choices"][0])
                    generated_text += chunk
                    if not chunk:
                        break
                all_outputs.append(Output(prompt=prompt, raw_output=generated_text))
                accumulated_context += "\n\n" + generated_text
        return all_outputs