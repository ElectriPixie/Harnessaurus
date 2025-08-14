import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import logging
import threading
from queue import Queue
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
import time
import uuid
import hashlib
import argparse

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="OpenAI-style local server with multi-chunk + harmony output")
parser.add_argument("--multi_chunk", action="store_true", help="Enable multi-chunk generation by default")
parser.add_argument("--harmony", choices=["off", "on", "both"], default="Both",
                    help="Harmony output mode: off (no tags), on (tags in content), both (normal + tags in debug)")
parser.add_argument("--port", type=int, default=6589, help="Server port")
parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for reproducible outputs")
parser.add_argument("--top_k", type=int, default=1, help="Top-k sampling")
parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
parser.add_argument("--do_sample", action="store_true", default=False,
                    help="Enable sampling (default False for deterministic output)")
parser.add_argument("--seed", type=int, default=42, help="Manual random seed for reproducibility")
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.use_deterministic_algorithms(True)

#if args.do_sample:
#    temperature = args.temperature if args.temperature > 0 else 1.0
#    top_k = args.top_k
#    top_p = args.top_p
#else:
#    # deterministic
#    temperature = 1.0  # ignored
#    top_k = 1          # ignored
#    top_p = 1.0        # ignored

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
HARMONY_MODE = args.harmony
TEMPERATURE = args.temperature
TOP_K = args.top_k
TOP_P = args.top_p

# --- Load model & tokenizer ---
logger.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
logger.info("Model loaded successfully on device %s", next(model.parameters()).device)

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
                            do_sample=args.do_sample,
                            temperature=TEMPERATURE,
                            top_k=TOP_K,
                            top_p=TOP_P,
                            return_dict_in_generate=True,
                        )
                    chunk = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    if chunk.startswith(context):
                        chunk = chunk[len(context):].strip()
                    full_text += " " + chunk
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
                        temperature=TEMPERATURE,
                        top_k=TOP_K,
                        top_p=TOP_P,
                        return_dict_in_generate=True,
                    )
                generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

            # Harmony tags
            harmony_output = f"<|channel|>analysis<|message|>Generated {len(generated_text.split())} tokens.\n" \
                             f"<|channel|>final<|message|>{generated_text}"

            finish_reason = "length" if len(generated_text.split()) >= max_tokens else "stop"
            duration = time.time() - start_time

            with lock:
                results[job_id] = {
                    "text": generated_text,
                    "harmony": harmony_output,
                    "finish_reason": finish_reason,
                    "duration": duration,
                    "prompt_length": len(prompt.split()),
                    "completion_length": len(generated_text.split()),
                    "device": str(model.device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk,
                }

            logger.info(
                "Job %s completed in %.2fs | Prompt tokens: %d | Completion tokens: %d | Finish reason: %s | Request ID: %s | Multi-chunk: %s",
                job_id,
                duration,
                len(prompt.split()),
                len(generated_text.split()),
                finish_reason,
                request_id,
                multi_chunk,
            )

        except Exception as e:
            logger.exception("Error generating job %s | Request ID: %s | Prompt: %s", job_id, request_id, prompt)
            with lock:
                results[job_id] = {
                    "text": f"Error during generation: {e}",
                    "harmony": f"<|channel|>analysis<|message|>Error occurred\n<|channel|>final<|message|>Error: {e}",
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
    multi_chunk: bool = None

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    user_message = req.messages[-1]["content"]
    client_host = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")
    request_id = str(uuid.uuid4())
    prompt_hash = hashlib.sha256(user_message.encode("utf-8")).hexdigest()
    multi_chunk = req.multi_chunk if req.multi_chunk is not None else DEFAULT_MULTI_CHUNK

    result = submit_generation(user_message, req.max_tokens, multi_chunk, request_id)

    if HARMONY_MODE == "on":
        content_out = result["harmony"]
    elif HARMONY_MODE == "both":
        content_out = result["text"]
    else:
        content_out = result["text"]

    response = {
        "id": "local-chat-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content_out},
             "finish_reason": result["finish_reason"]}
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

    if HARMONY_MODE == "both":
        response["debug"]["harmony_raw"] = result["harmony"]

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
