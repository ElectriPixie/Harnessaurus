# openai-serve_harmony_refactored.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging
import threading
from queue import Queue
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import time
import uuid
import argparse
import math
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent
)

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="OpenAI-style Harmony server with paginated chunks")
parser.add_argument("--model_id", type=str, default="/data/AI/Models/gpt-oss-20b")
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--chunk_size", type=int, default=512)
parser.add_argument("--num_threads", type=int, default=4)
parser.add_argument("--multi_chunk", action="store_true", default=False)   # client-facing
parser.add_argument("--chunk_streaming", action="store_true", default=False)  # server-side
parser.add_argument("--strip_legacy_format", action="store_true", default=True)
parser.add_argument("--port", type=int, default=6589)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--do_sample", action="store_true", default=False)
parser.add_argument("--deterministic", choices=["on", "off"], default="on")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repetition_control", choices=["on", "off"], default="on")
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
args = parser.parse_args()

DETERMINISTIC = args.deterministic.lower() == "on"
REPETITION_CONTROL = args.repetition_control.lower() == "on"

if DETERMINISTIC:
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

# --- Logging setup ---
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join("logs", f"generation_{timestamp}.log")
logger = logging.getLogger("harmony_server")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_file)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Thread %(threadName)s] %(message)s"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# --- FastAPI app ---
app = FastAPI()

# --- Load model and tokenizer ---
logger.info(f"Loading model {args.model_id}...")
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
logger.info(f"Model loaded on device {next(model.parameters()).device}")

# --- Threading & job management ---
job_queue = Queue()
results = {}
output_buffers = {}
lock = threading.Lock()


# --- Helper functions ---
def flatten_entries(entries):
    flat_list = []
    for e in entries:
        if isinstance(e, list):
            flat_list.extend(flatten_entries(e))
        elif hasattr(e, "content"):
            flat_list.append(str(e.content))
        else:
            flat_list.append(str(e))
    return flat_list


def generate_harmony_text(messages, max_tokens, request_id, chunk_streaming=False):
    """Generate text with optional server-side streaming in chunks."""
    convo_messages = [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new().with_instructions("Respond helpfully and safely"))
    ]
    for msg in messages:
        convo_messages.append(Message.from_role_and_content(Role.USER, msg["content"]))

    convo = Conversation.from_messages(convo_messages)
    prefill_ids_list = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    # Ensure termination tokens
    while prefill_ids_list and prefill_ids_list[-1] == 200007:
        prefill_ids_list.pop()
    if not prefill_ids_list or prefill_ids_list[-1] != 200006:
        prefill_ids_list.append(200006)

    input_ids = torch.tensor(prefill_ids_list, device=model.device).unsqueeze(0)
    full_text = ""
    tokens_generated = 0

    iterations = math.ceil(max_tokens / args.chunk_size) if chunk_streaming else 1

    for _ in range(iterations):
        do_sample_flag = args.do_sample and not DETERMINISTIC
        generate_kwargs = {
            "max_new_tokens": min(args.chunk_size, max_tokens - tokens_generated),
            "do_sample": do_sample_flag,
            "eos_token_id": stop_token_ids
        }
        if do_sample_flag:
            generate_kwargs.update({
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p
            })
        if REPETITION_CONTROL:
            if args.no_repeat_ngram_size > 0:
                generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
            if args.repetition_penalty != 1.0:
                generate_kwargs["repetition_penalty"] = args.repetition_penalty

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, **generate_kwargs)

        completion_ids = outputs[0][input_ids.shape[1]:]

        if args.strip_legacy_format:
            LEGACY_TOKENS = set(range(200000, 200008))
            completion_ids = torch.tensor([t for t in completion_ids.tolist() if t not in LEGACY_TOKENS], device=model.device)

        assistant_entries = encoding.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)
        chunk_text = " ".join(flatten_entries(assistant_entries))

        if chunk_streaming:
            with lock:
                output_buffers[request_id] = output_buffers.get(request_id, "") + chunk_text
        else:
            full_text += chunk_text

        tokens_generated += len(chunk_text.split())

        if chunk_streaming:
            input_ids = torch.cat([input_ids, completion_ids.unsqueeze(0)], dim=1)

        if len(completion_ids) < min(args.chunk_size, max_tokens - tokens_generated):
            break

    if not chunk_streaming:
        with lock:
            output_buffers[request_id] = full_text

    return output_buffers[request_id]


# --- Worker thread ---
def model_worker():
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_id, messages, max_tokens, multi_chunk, request_id, chunk_streaming = job
        start_time = time.time()
        try:
            text = generate_harmony_text(messages, max_tokens, request_id, chunk_streaming)
            duration = time.time() - start_time
            with lock:
                results[job_id] = {
                    "text": text.strip(),
                    "finish_reason": "length" if len(text.split()) >= max_tokens else "stop",
                    "duration": duration,
                    "prompt_length": sum(len(m["content"].split()) for m in messages),
                    "completion_length": len(text.split()),
                    "device": str(model.device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk
                }
        except Exception as e:
            logger.exception(f"[Worker] ERROR Job {job_id} | Request ID: {request_id}")
            with lock:
                results[job_id] = {
                    "text": f"Error: {e}",
                    "finish_reason": "error",
                    "duration": 0,
                    "prompt_length": sum(len(m["content"].split()) for m in messages),
                    "completion_length": 0,
                    "device": str(model.device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk
                }
        finally:
            job_queue.task_done()


# --- Start worker threads ---
for i in range(args.num_threads):
    threading.Thread(target=model_worker, daemon=True, name=f"Worker-{i}").start()


# --- Job submission ---
def submit_generation(messages, max_tokens=None, multi_chunk=None, request_id=None, chunk_streaming=False):
    max_tokens = max_tokens or args.max_tokens
    multi_chunk = multi_chunk if multi_chunk is not None else args.multi_chunk
    with lock:
        job_id = len(results)
    job_queue.put((job_id, messages, max_tokens, multi_chunk, request_id, chunk_streaming))
    while True:
        with lock:
            if job_id in results:
                return results.pop(job_id)
        time.sleep(0.05)


# --- FastAPI endpoints ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = args.max_tokens
    multi_chunk: bool = None
    chunk_streaming: bool = None


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    request_id = str(uuid.uuid4())
    multi_chunk = req.multi_chunk if req.multi_chunk is not None else args.multi_chunk
    chunk_streaming = req.chunk_streaming if req.chunk_streaming is not None else False
    result = submit_generation(req.messages, req.max_tokens, multi_chunk, request_id, chunk_streaming)
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": result["text"]},
             "finish_reason": result["finish_reason"], "logprobs": None}
        ],
        "usage": {
            "prompt_tokens": result["prompt_length"],
            "completion_tokens": result["completion_length"],
            "total_tokens": result["prompt_length"] + result["completion_length"]
        }
    }


@app.get("/v1/chat/completions/chunk")
async def get_chunk(request_id: str, chunk_index: int = 0, chunk_size: int = 512):
    with lock:
        full_text = output_buffers.get(request_id)
        if not full_text:
            return {"error": "Request ID not found or expired."}
        start = chunk_index * chunk_size
        end = start + chunk_size
        chunk_text = full_text[start:end]
        finished = end >= len(full_text)
    return {
        "request_id": request_id,
        "chunk_index": chunk_index,
        "chunk_size": chunk_size,
        "text": chunk_text,
        "finished": finished,
        "total_length": len(full_text)
    }


@app.get("/v1/version")
async def version():
    return {
        "version": "0.5.0",
        "model_id": args.model_id,
        "harmony_mode": "OpenAI-Harmony",
        "deterministic": "Yes" if DETERMINISTIC else "No"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
