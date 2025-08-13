import logging
import threading
from queue import Queue
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
import time
import os
import uuid
import hashlib
import argparse

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="OpenAI-style server with optional multi-chunk mode")
parser.add_argument("--multi_chunk", action="store_true", help="Enable multi-chunk generation by default")
parser.add_argument("--port", type=int, default=6589, help="Server port")
args = parser.parse_args()

# --- Logging setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/generation.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Thread %(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- FastAPI app ---
app = FastAPI()

# --- Config ---
MODEL_ID = "/data/AI/Models/gpt-oss-20b"
MAX_TOKENS = 4096
CHUNK_SIZE = 1024
NUM_THREADS = 4
DEFAULT_MULTI_CHUNK = args.multi_chunk

# --- Load model & tokenizer ---
logger.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
logger.info("Model loaded on device %s", next(model.parameters()).device)

# --- Threaded job queue ---
job_queue = Queue()
results = {}
lock = threading.Lock()

def model_worker():
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_id, prompt, max_tokens, multi_chunk, request_id = job
        start_time = time.time()
        analysis_text = ""
        try:
            if multi_chunk:
                context = prompt
                full_text = ""
                tokens_generated = 0
                while tokens_generated < max_tokens:
                    inputs = tokenizer(context, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=min(CHUNK_SIZE, max_tokens - tokens_generated),
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95,
                            return_dict_in_generate=True,
                        )
                    chunk = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    if chunk.startswith(context):
                        chunk = chunk[len(context):].strip()
                    # Append to full text
                    full_text += " " + chunk
                    # Append to analysis
                    analysis_text += f"Chunk {tokens_generated // CHUNK_SIZE + 1} generated {len(chunk.split())} tokens.\n"
                    tokens_generated += len(chunk.split())
                    context += " " + chunk
                    if len(chunk.split()) < min(CHUNK_SIZE, max_tokens - tokens_generated):
                        break
                generated_text = full_text.strip()
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        return_dict_in_generate=True,
                    )
                generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                analysis_text += f"Single-chunk generated {len(generated_text.split())} tokens.\n"

            finish_reason = "length" if len(generated_text.split()) >= max_tokens else "stop"
            duration = time.time() - start_time

            with lock:
                results[job_id] = {
                    "text": f"<|channel|>analysis<|message|>{analysis_text.strip()}\n<|channel|>final<|message|>{generated_text}",
                    "finish_reason": finish_reason,
                    "duration": duration,
                    "prompt_length": len(prompt.split()),
                    "completion_length": len(generated_text.split()),
                    "device": str(model.device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk,
                }

            logger.info(
                "Job %s completed in %.2fs on device %s | Prompt tokens: %d | Completion tokens: %d | Finish reason: %s | Request ID: %s | Multi-chunk: %s",
                job_id,
                duration,
                model.device,
                len(prompt.split()),
                len(generated_text.split()),
                finish_reason,
                request_id,
                multi_chunk,
            )

        except Exception as e:
            logger.exception("Error generating output for job %s | Request ID: %s | Prompt: %s", job_id, request_id, prompt)
            with lock:
                results[job_id] = {
                    "text": f"<|channel|>analysis<|message|>Error: {e}\n<|channel|>final<|message|>Error during generation.",
                    "finish_reason": "error",
                    "duration": 0,
                    "prompt_length": len(prompt.split()),
                    "completion_length": 0,
                    "device": str(model.device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk,
                }
        finally:
            job_queue.task_done()
            logger.info("Current queue size: %d", job_queue.qsize())

for i in range(NUM_THREADS):
    threading.Thread(target=model_worker, daemon=True, name=f"Worker-{i}").start()

def submit_generation(prompt, max_tokens=MAX_TOKENS, multi_chunk=None, request_id=None):
    if multi_chunk is None:
        multi_chunk = DEFAULT_MULTI_CHUNK
    with lock:
        job_id = len(results)
    job_queue.put((job_id, prompt, max_tokens, multi_chunk, request_id))
    job_queue.join()
    with lock:
        return results.pop(job_id)

# --- API ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = MAX_TOKENS
    multi_chunk: bool = None  # optional override

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    user_message = req.messages[-1]["content"]
    client_host = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")
    request_id = str(uuid.uuid4())
    prompt_hash = hashlib.sha256(user_message.encode("utf-8")).hexdigest()
    multi_chunk = req.multi_chunk if req.multi_chunk is not None else DEFAULT_MULTI_CHUNK

    logger.info(
        "Incoming request | Request ID: %s | Client: %s | User-Agent: %s | Model: %s | Max tokens: %d | Multi-chunk: %s | Prompt hash: %s",
        request_id,
        client_host,
        user_agent,
        req.model,
        req.max_tokens,
        multi_chunk,
        prompt_hash,
    )

    result = submit_generation(user_message, req.max_tokens, multi_chunk, request_id)

    return {
        "id": "local-chat-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_length"],
            "completion_tokens": result["completion_length"],
            "total_tokens": result["prompt_length"] + result["completion_length"],
        },
        "debug": {
            "duration_seconds": result["duration"],
            "device": result["device"],
            "client_host": client_host,
            "user_agent": user_agent,
            "request_id": request_id,
            "prompt_hash": prompt_hash,
            "multi_chunk": result["multi_chunk"],
        },
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
