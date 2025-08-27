import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging
import threading
import time
import uuid
import argparse
import math
from queue import Queue, Empty
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Role,
    Author,
    Message,
    Conversation,
    StreamableParser,
)

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="OpenAI-style Harmony server")
parser.add_argument("--model_id", type=str, default="/data/AI/Models/gpt-oss-20b")
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--chunk_size", type=int, default=512)
parser.add_argument("--num_threads", type=int, default=4)
parser.add_argument("--multi_chunk", action="store_true", default=False)
parser.add_argument("--chunk_streaming", action="store_true", default=False)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--do_sample", action="store_true", default=False)
parser.add_argument("--deterministic", choices=["on","off"], default="on")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repetition_control", choices=["on","off"], default="on")
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
parser.add_argument("--port", type=int, default=6589)
parser.add_argument("--strip_legacy_format", action="store_true", default=False)
args = parser.parse_args()


DETERMINISTIC = args.deterministic.lower() == "on"
REPETITION_CONTROL = args.repetition_control.lower() == "on"
INSTRUCTIONS = "You are an AI assistant serving an insane mad scientist who's a really nice soul."

# --- Deterministic Setup ---
if DETERMINISTIC:
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

# --- Logging ---
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join("logs", f"generation_{timestamp}.log")
logger = logging.getLogger("harmony_server")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Thread %(threadName)s] %(message)s"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

logger.info(f"Starting Harmony server with model: {args.model_id}")

# --- FastAPI app ---
app = FastAPI()

# --- Load Model and Encoding ---
logger.info(f"Loading model {args.model_id}...")
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
device = next(model.parameters()).device
logger.info(f"Model loaded on device {device}")

# --- Threading + IPC ---
job_queue = Queue()
results = {}      # job_id -> final result
stream_queues = {}  # request_id -> queue for streaming
lock = threading.Lock()


def conversation_to_string(convo):
    """Convert a Harmony Conversation into a single prompt string."""
    messages_text = []
    for msg in convo.messages:
        # Try to get role from author if role attr is missing
        if hasattr(msg, "role"):
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        elif hasattr(msg, "author") and hasattr(msg.author, "role"):
            role = msg.author.role.value if hasattr(msg.author.role, "value") else str(msg.author.role)
        else:
            role = "unknown"
        
        content = getattr(msg, "content", str(msg))
        messages_text.append(f"[{role}] {content}")
    return "\n".join(messages_text)

def generate_chunk(conversation, chunk_size, request_id):
    """
    Generate a chunk of text from a conversation while preserving roles and zero-width characters.
    """

    # Build the prompt string safely
    prompt_lines = []
    for msg in conversation:
        role = getattr(msg, "role", "unknown")
        name = getattr(msg, "name", None)
        content = getattr(msg, "content", "")
        # Use repr to safely show zero-width characters
        prompt_lines.append(f"role={role} name={name}: {repr(content)}")

    prompt_str = "\n".join(prompt_lines)

    # Tokenize with attention mask to avoid pad=eos issues
    inputs = tokenizer(
        prompt_str,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=chunk_size
    )
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    # Generate output
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=chunk_size,
        do_sample=True  # or False depending on your setup
    )

    # Decode output safely
    chunk_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Build the completion IDs and channels (example)
    completion_ids = outputs[0].tolist()
    channels = {"final": chunk_text}

    finish_reason = "stop"  # adjust if you have other finish logic

    # Debug log
    print(f"[DEBUG] Generated chunk ({request_id}): {repr(chunk_text)}")

    return chunk_text, completion_ids, channels, finish_reason

# --- Worker Thread ---
def model_worker():
    while True:
        job = job_queue.get()
        if job is None:
            logger.debug("Worker received shutdown signal")
            break

        # Handle tuple dynamically
        try:
            if len(job) == 7:
                job_id, messages, max_tokens_param, multi_chunk, request_id, chunk_streaming, chunk_size_param = job
            elif len(job) == 6:
                job_id, messages, max_tokens_param, multi_chunk, request_id, chunk_streaming = job
                chunk_size_param = args.chunk_size
            elif len(job) == 5:
                job_id, messages, max_tokens_param, multi_chunk, request_id = job
                chunk_streaming = False
                chunk_size_param = args.chunk_size
            else:
                raise ValueError(f"Unexpected job tuple length: {len(job)}")
        except Exception:
            logger.exception("Malformed job tuple")
            job_queue.task_done()
            continue

        start_time = time.time()
        try:
            # Build Conversation
            convo = Conversation.from_messages([
                Message.from_role_and_content(Role.SYSTEM, INSTRUCTIONS)
            ] + [Message.from_role_and_content(Role.USER if m["role"]=="user" else Role.ASSISTANT, m["content"]) for m in messages])

            chunk_text, completion_ids, channels, finish_reason = generate_chunk(convo, chunk_size_param, request_id)

            duration = time.time() - start_time
            with lock:
                results[job_id] = {
                    "text": chunk_text,
                    "channels": channels,
                    "finish_reason": finish_reason,
                    "duration": duration,
                    "prompt_length": sum(len(m.get("content","").split()) for m in messages),
                    "completion_length": len(chunk_text.split()),
                    "device": str(device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk,
                }

            # Stream sentinel
            with lock:
                q = stream_queues.get(request_id)
            if q is not None:
                q.put(None)

        except Exception as e:
            logger.exception(f"ERROR Job {job_id} | Request ID: {request_id}")
            with lock:
                results[job_id] = {
                    "text": f"Error: {e}",
                    "channels": {},
                    "finish_reason": "error",
                    "duration": 0,
                    "prompt_length": sum(len(m.get("content","").split()) for m in messages),
                    "completion_length": 0,
                    "device": str(device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk,
                }
            with lock:
                q = stream_queues.get(request_id)
            if q is not None:
                q.put(None)
        finally:
            job_queue.task_done()

# --- Start worker threads ---
for i in range(args.num_threads):
    threading.Thread(target=model_worker, daemon=True, name=f"Worker-{i}").start()

# --- Submit generation job ---
def submit_generation(messages, max_tokens=None, multi_chunk=None, request_id=None, chunk_streaming=False, chunk_size=None):
    max_tokens = max_tokens or args.max_tokens
    multi_chunk = multi_chunk if multi_chunk is not None else args.multi_chunk
    chunk_size = chunk_size or args.chunk_size

    with lock:
        job_id = len(results)
        logger.debug(f"Submitting job {job_id} | Request ID: {request_id}")

    if chunk_streaming:
        with lock:
            stream_queues[request_id] = Queue()

    job_queue.put((job_id, messages, max_tokens, multi_chunk, request_id, chunk_streaming, chunk_size))

    if not chunk_streaming:
        # Blocking: wait until result is ready
        while True:
            with lock:
                if job_id in results:
                    return results.pop(job_id)
            time.sleep(0.01)
    else:
        # Streaming generator
        import json

        def stream_gen():
            with lock:
                q_local = stream_queues.get(request_id)
            if q_local is None:
                return

            while True:
                try:
                    item = q_local.get(timeout=30)
                except Empty:
                    with lock:
                        if job_id in results:
                            break
                    continue
                if item is None:
                    break
                yield json.dumps({"event": "delta", "data": item}, ensure_ascii=False) + "\n"

            # Yield final result
            with lock:
                if job_id in results:
                    final = results[job_id]["text"]
                    final_channels = results[job_id].get("channels", {})
                    yield json.dumps({"event": "done", "data": {"text": final, "channels": final_channels}}, ensure_ascii=False) + "\n"

                stream_queues.pop(request_id, None)

        return stream_gen()


# --- API Schema ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = args.max_tokens
    multi_chunk: bool = None
    chunk_size: int = None
    chunk_stream: bool = False

# --- Routes ---
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    request_id = str(uuid.uuid4())
    multi_chunk = req.multi_chunk if req.multi_chunk is not None else args.multi_chunk
    chunk_size = req.chunk_size if req.chunk_size else args.chunk_size
    use_stream = req.chunk_stream if req.chunk_stream is not None else args.chunk_streaming

    logger.info(
        f"[API] Incoming request | Request ID: {request_id} | IP: {request.client.host} | "
        f"Multi-chunk: {multi_chunk} | Stream: {use_stream} | Chunk size: {chunk_size}"
    )
    if req.messages:
        logger.info(f"[API] Last user message preview: {str(req.messages[-1].get('content',''))[:200]}")

    result_or_gen = submit_generation(
        req.messages,
        max_tokens=req.max_tokens,
        multi_chunk=multi_chunk,
        request_id=request_id,
        chunk_streaming=use_stream,
        chunk_size=chunk_size
    )

    if not use_stream:
        result = result_or_gen
        logger.debug(f"[API] Full generation result: {result}")
        # safely get channels
        channels = result.get("channels") or {}
        final_text = channels.get("final") or ""
        analysis_text = channels.get("analysis") or ""
        commentary_text = channels.get("commentary") or ""

        logger.info(
            f"[API] Response ready | Request ID: {request_id} | Tokens(est): {result.get('completion_length',0)} | "
            f"Finish reason: {result.get('finish_reason','unknown')}"
        )
        logger.info(f"Channels: {repr(channels)}")

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_text,
                        "channels": {
                            "final": final_text,
                            "analysis": analysis_text,
                            "commentary": commentary_text,
                        },
                    },
                    "finish_reason": result.get("finish_reason", "unknown"),
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": result.get("prompt_length", 0),
                "completion_tokens": result.get("completion_length", 0),
                "total_tokens": result.get("prompt_length", 0) + result.get("completion_length", 0),
            },
        }
    else:
        logger.info(f"[API] Streaming response started | Request ID: {request_id}")
        return StreamingResponse(result_or_gen, media_type="text/plain")

@app.get("/v1/version")
async def version():
    return {
        "version": "0.6.0-hf-template",
        "model_id": args.model_id,
        "harmony_mode": "OpenAI-Harmony",
        "deterministic": "Yes" if DETERMINISTIC else "No",
    }

if __name__ == "__main__":
    logger.info(f"Starting server on port {args.port}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
