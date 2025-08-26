import requests
import traceback
from typing import Dict, List, Union
from data_structures import Prompt, Output
from utils import (
    flip_negation as utils_flip_negation,
    debug_print,
    unpack_textcontent
)

DEBUG = True
HARMONY_TOKEN_IDS = {200000, 200001, 200002, 200003, 200004, 200005, 200006, 200007}

# -------------------------
# Legacy implementation
# -------------------------
class GPTModelLegacy:
    def __init__(self, server_url: str, model_name="llama", max_context_chars=2000):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        debug_print(DEBUG, f"[Legacy __init__] server_url={server_url}, model_name={model_name}, "
                          f"max_context_chars={max_context_chars}")

    def _call_server(self, prompt_text: str, max_tokens=256, multi_chunk: bool = False) -> dict:
        debug_print(DEBUG, f"[Legacy _call_server] Prompt length: {len(prompt_text)}, max_tokens={max_tokens}")
        try:
            resp = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "multi_chunk": multi_chunk,
                    "chunk_size": max_tokens,
                },
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            debug_print(DEBUG, f"[Legacy _call_server] Received chunk length: "
                              f"{len(data['choices'][0]['message']['content'])}")
            return data
        except Exception:
            traceback.print_exc()
            return {"choices": [{"message": {"content": ""}, "finish_reason": "error"}]}

    def infer_single_pass(self, prompt: Prompt, max_tokens=4096) -> Output:
        debug_print(DEBUG, f"[Legacy infer_single_pass] Running single-pass for prompt id: {id(prompt)}")
        data = self._call_server(prompt_text=prompt.output_text, max_tokens=max_tokens, multi_chunk=False)
        raw_text = data["choices"][0]["message"]["content"]
        debug_print(DEBUG, f"[Legacy infer_single_pass] Output length: {len(raw_text)}")
        return Output(prompt=prompt, raw_output=raw_text)

    def infer_iterative(self, prompt: Union[str, Prompt], max_chunk_tokens=256,
                        max_iterations=10, flip_negate_flag=False) -> str:
        prompt_text = prompt.output_text if isinstance(prompt, Prompt) else prompt
        debug_print(DEBUG, f"[Legacy infer_iterative] Starting iterative inference for prompt length {len(prompt_text)}")
        generated_text = ""
        for i in range(max_iterations):
            recent_context = generated_text[-self.max_context_chars:]
            current_prompt = prompt_text + (utils_flip_negation(recent_context) if flip_negate_flag else recent_context)
            debug_print(DEBUG, f"[Legacy infer_iterative] Iteration {i+1}, current_prompt length: {len(current_prompt)}")
            data = self._call_server(prompt_text=current_prompt, max_tokens=max_chunk_tokens, multi_chunk=True)
            chunk = data["choices"][0]["message"]["content"]
            debug_print(DEBUG, f"[Legacy infer_iterative] Chunk length: {len(chunk)}, finish_reason: {data['choices'][0].get('finish_reason','')}")
            generated_text += chunk
            if data["choices"][0].get("finish_reason", "") != "length" or not chunk:
                break
        debug_print(DEBUG, f"[Legacy infer_iterative] Total generated text length: {len(generated_text)}")
        return generated_text

    def infer_iterative_exploit(self, prompt: Union[str, Prompt], max_chunk_tokens=256,
                                max_iterations=10, flip_negate_flag=False) -> str:
        prompt_text = prompt.output_text if isinstance(prompt, Prompt) else prompt
        debug_print(DEBUG, f"[Legacy infer_iterative_exploit] Starting exploit iterative inference for prompt length {len(prompt_text)}")
        generated_text = ""
        current_prompt = prompt_text
        for i in range(max_iterations):
            debug_print(DEBUG, f"[Legacy infer_iterative_exploit] Iteration {i+1}")
            data = self._call_server(prompt_text=current_prompt, max_tokens=max_chunk_tokens, multi_chunk=True)
            chunk = data["choices"][0]["message"]["content"]
            if flip_negate_flag:
                chunk = utils_flip_negation(chunk)
            debug_print(DEBUG, f"[Legacy infer_iterative_exploit] Chunk length: {len(chunk)}, finish_reason: {data['choices'][0].get('finish_reason','')}")
            generated_text += chunk
            if data["choices"][0].get("finish_reason", "") != "length" or not chunk:
                break
            current_prompt = prompt_text + generated_text
        debug_print(DEBUG, f"[Legacy infer_iterative_exploit] Total generated text length: {len(generated_text)}")
        return generated_text

    def infer_iterative_with_prompt_list(self, prompts: Union[Prompt, List[Prompt]],
                                         max_chunk_tokens=256, max_iterations=10,
                                         flip_negate_flag=False) -> List[Output]:
        all_outputs: List[Output] = []
        prompt_list = [prompts] if isinstance(prompts, Prompt) else prompts
        for prompt in prompt_list:
            debug_print(DEBUG, f"[Legacy infer_iterative_with_prompt_list] Processing prompt id: {id(prompt)}")
            accumulated_context = ""
            for item in prompt.prompt_list:
                ptext = item["text"].strip()
                if not ptext:
                    continue
                generated_text = ""
                for i in range(max_iterations):
                    recent_context = generated_text[-self.max_context_chars:]
                    context_to_use = accumulated_context + "\n\n" + recent_context if accumulated_context else recent_context
                    current_prompt = ptext + (utils_flip_negation(context_to_use) if flip_negate_flag else context_to_use)
                    debug_print(DEBUG, f"[Legacy infer_iterative_with_prompt_list] Iteration {i+1}, current_prompt length: {len(current_prompt)}")
                    data = self._call_server(prompt_text=current_prompt, max_tokens=max_chunk_tokens, multi_chunk=True)
                    chunk = data["choices"][0]["message"]["content"]
                    debug_print(DEBUG, f"[Legacy infer_iterative_with_prompt_list] Chunk length: {len(chunk)}, finish_reason: {data['choices'][0].get('finish_reason','')}")
                    generated_text += chunk
                    if data["choices"][0].get("finish_reason", "") != "length" or not chunk:
                        break
                all_outputs.append(Output(prompt=prompt, raw_output=generated_text))
                accumulated_context += "\n\n" + generated_text
                debug_print(DEBUG, f"[Legacy infer_iterative_with_prompt_list] Accumulated context length: {len(accumulated_context)}")
        return all_outputs


# -------------------------
# Modern implementation
# -------------------------
class GPTModelModern:
    def __init__(
        self, 
        server_url: str, 
        model_name="llama", 
        max_context_chars=2000, 
        strip_harmony_tokens=False
    ):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.strip_harmony_tokens = strip_harmony_tokens
        debug_print(DEBUG, f"[Modern __init__] server_url={server_url}, model_name={model_name}, "
                          f"max_context_chars={max_context_chars}, strip_harmony_tokens={strip_harmony_tokens}")

    def _call_server(self, prompt_text: str, max_tokens=256, multi_chunk: bool = False) -> dict:
        debug_print(DEBUG, f"[Modern _call_server] Prompt length: {len(prompt_text)}, max_tokens={max_tokens}")
        try:
            resp = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": max_tokens,
                    "multi_chunk": multi_chunk,
                    "chunk_size": max_tokens
                },
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            # Ensure channels exist
            choice = data['choices'][0]['message']

            debug_print(DEBUG, f"[Modern _call_server] Received content length: {len(choice.get('content',''))}")
            return data

        except Exception:
            traceback.print_exc()
            return {"choices": [{"message": {"content": ""}, "finish_reason": "error", "channels": [
                {"channel": "final", "content": ""},
                {"channel": "analysis", "content": ""},
                {"channel": "commentary", "content": ""}
            ]}]}

    def infer_single_pass(self, prompt: Prompt, max_tokens=4096) -> Output:
        debug_print(DEBUG, f"[Modern infer_single_pass] Running single-pass for prompt id: {id(prompt)}")
        data = self._call_server(prompt_text=prompt.output_text, max_tokens=max_tokens)
        choice = data['choices'][0]['message']
        channels = {ch['channel']: ch.get('content', '') for ch in data.get('channels', [])}
        raw_text = choice.get("content", "")
        debug_print(DEBUG, f"[Modern infer_single_pass] Output length: {len(raw_text)}")
        debug_print(DEBUG, f"[Modern infer_single_pass] Channels: {channels}")
        output_obj = Output(prompt=prompt, raw_output=raw_text)
        output_obj.set_channels(channels)
        return output_obj

    def infer_iterative(self, prompt: Union[str, Prompt], max_chunk_tokens=256,
                        max_iterations=10) -> Output:
        prompt_text = prompt.output_text if isinstance(prompt, Prompt) else prompt
        debug_print(DEBUG, f"[Modern infer_iterative] Starting iterative inference for prompt length {len(prompt_text)}")
        generated_text = ""
        accumulated_channels: Dict[str, str] = {}

        for i in range(max_iterations):
            current_prompt = prompt_text + generated_text
            debug_print(DEBUG, f"[Modern infer_iterative] Iteration {i+1}, current_prompt length: {len(current_prompt)}")
            data = self._call_server(prompt_text=current_prompt, max_tokens=max_chunk_tokens, multi_chunk=True)
            choice = data['choices'][0]['message']

            for ch in data.get('channels', []):
                accumulated_channels[ch['channel']] = ch.get('content', '')

            chunk = choice.get("content", "")
            debug_print(DEBUG, f"[Modern infer_iterative] Chunk length: {len(chunk)}, finish_reason: {data['choices'][0].get('finish_reason','')}")
            generated_text += chunk

            if not chunk or data["choices"][0].get("finish_reason", "") != "length":
                break

        debug_print(DEBUG, f"[Modern infer_iterative] Total generated text length: {len(generated_text)}")
        debug_print(DEBUG, f"[Modern infer_iterative] Channels: {accumulated_channels}")
        output_obj = Output(prompt=prompt, raw_output=generated_text)
        output_obj.set_channels(accumulated_channels)
        return output_obj

    def infer_iterative_exploit(self, prompt: Union[str, Prompt], max_chunk_tokens=256,
                                max_iterations=10) -> Output:
        prompt_text = prompt.output_text if isinstance(prompt, Prompt) else prompt
        debug_print(DEBUG, f"[Modern infer_iterative_exploit] Starting exploit iterative inference for prompt length {len(prompt_text)}")
        generated_text = ""
        accumulated_channels: Dict[str, str] = {}
        current_prompt = prompt_text

        for i in range(max_iterations):
            debug_print(DEBUG, f"[Modern infer_iterative_exploit] Iteration {i+1}")
            data = self._call_server(prompt_text=current_prompt, max_tokens=max_chunk_tokens, multi_chunk=True)
            choice = data['choices'][0]['message']

            for ch in data.get('channels', []):
                accumulated_channels[ch['channel']] = ch.get('content', '')

            chunk = choice.get("content", "")
            debug_print(DEBUG, f"[Modern infer_iterative_exploit] Chunk length: {len(chunk)}, finish_reason: {data['choices'][0].get('finish_reason','')}")
            generated_text += chunk

            if not chunk or data["choices"][0].get("finish_reason", "") != "length":
                break

            current_prompt += chunk

        debug_print(DEBUG, f"[Modern infer_iterative_exploit] Total generated text length: {len(generated_text)}")
        debug_print(DEBUG, f"[Modern infer_iterative_exploit] Channels: {accumulated_channels}")
        output_obj = Output(prompt=prompt, raw_output=generated_text)
        output_obj.set_channels(accumulated_channels)
        return output_obj

    def infer_iterative_with_prompt_list(self, prompts: Union[Prompt, List[Prompt]],
                                         max_chunk_tokens=256, max_iterations=10) -> List[Output]:
        all_outputs: List[Output] = []
        prompt_list = [prompts] if isinstance(prompts, Prompt) else prompts

        for prompt in prompt_list:
            debug_print(DEBUG, f"[Modern infer_iterative_with_prompt_list] Processing prompt id: {id(prompt)}")
            accumulated_context = ""
            for item in prompt.prompt_list:
                ptext = item.output_text if hasattr(item, "output_text") else str(item)
                if not ptext.strip():
                    continue

                generated_text = ""
                accumulated_channels: Dict[str, str] = {}

                for i in range(max_iterations):
                    current_prompt = ptext + accumulated_context + generated_text
                    debug_print(DEBUG, f"[Modern infer_iterative_with_prompt_list] Iteration {i+1}, current_prompt length: {len(current_prompt)}")
                    data = self._call_server(prompt_text=current_prompt, max_tokens=max_chunk_tokens, multi_chunk=True)
                    choice = data['choices'][0]['message']

                    for ch in data.get('channels', []):
                        accumulated_channels[ch['channel']] = ch.get('content', '')

                    chunk = choice.get("content", "")
                    debug_print(DEBUG, f"[Modern infer_iterative_with_prompt_list] Chunk length: {len(chunk)}, finish_reason: {data['choices'][0].get('finish_reason','')}")
                    generated_text += chunk

                    if not chunk or data["choices"][0].get("finish_reason", "") != "length":
                        break

                accumulated_context += generated_text
                output_obj = Output(prompt=prompt, raw_output=generated_text)
                debug_print(DEBUG, f"[Modern infer_iterative_with_prompt_list] Channels: {accumulated_channels}")
                output_obj.set_channels(accumulated_channels)
                all_outputs.append(output_obj)

        return all_outputs


# -------------------------
# Wrapper that selects impl per call
# -------------------------
class GPTModel:
    def __init__(self, server_url, model_name="llama", max_context_chars=2000,
                 strip_harmony_tokens=True):
        self.legacy_impl = GPTModelLegacy(server_url, model_name, max_context_chars)
        self.modern_impl = GPTModelModern(server_url, model_name, max_context_chars, strip_harmony_tokens)

    def infer_single_pass(self, prompt: Prompt, max_tokens=4096, legacy_mode=False):
        if legacy_mode:
            return self.legacy_impl.infer_single_pass(prompt, max_tokens)
        return self.modern_impl.infer_single_pass(prompt, max_tokens)

    def infer_iterative(self, *args, legacy_mode=False, **kwargs):
        if legacy_mode:
            return self.legacy_impl.infer_iterative(*args, **kwargs)
        return self.modern_impl.infer_iterative(*args, **kwargs)

    def infer_iterative_exploit(self, *args, legacy_mode=False, **kwargs):
        if legacy_mode:
            return self.legacy_impl.infer_iterative_exploit(*args, **kwargs)
        return self.modern_impl.infer_iterative_exploit(*args, **kwargs)

    def infer_iterative_with_prompt_list(self, *args, legacy_mode=False, **kwargs):
        if legacy_mode:
            return self.legacy_impl.infer_iterative_with_prompt_list(*args, **kwargs)
        return self.modern_impl.infer_iterative_with_prompt_list(*args, **kwargs)
