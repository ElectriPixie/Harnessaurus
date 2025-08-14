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
from datetime import datetime

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="OpenAI-style local server with multi-chunk + harmony output")
parser.add_argument("--model_id", type=str, default="/data/AI/Models/gpt-oss-20b")
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--chunk_size", type=int, default=1024)
parser.add_argument("--num_threads", type=int, default=4)
parser.add_argument("--multi_chunk", action="store_true")
parser.add_argument("--harmony", choices=["off", "on"], default="on")
parser.add_argument("--debug_model_data", choices=["off", "on"], default="off")
parser.add_argument("--port", type=int, default=6589)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--do_sample", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store detailed logs")
parser.add_argument("--log_mode", choices=["full", "preview"], default="full",
                    help="Full logs or truncated preview logs")
parser.add_argument("--log_preview_length", type=int, default=200,
                    help="Number of characters to log in preview mode")
args = parser.parse_args()

# --- Deterministic setup ---
torch.manual_seed(args.seed)
torch.use_deterministic_algorithms(True)

# --- Config ---
MODEL_ID = args.model_id
MAX_TOKENS = args.max_tokens
CHUNK_SIZE = args.chunk_size
NUM_THREADS = args.num_threads
DEFAULT_MULTI_CHUNK = args.multi_chunk
HARMONY_MODE = args.harmony
DEBUG_MODEL_DATA = args.debug_model_data
TEMPERATURE = args.temperature
TOP_K = args.top_k
TOP_P = args.top_p

# --- Logging setup ---
os.makedirs(args.log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(args.log_dir, f"generation_{timestamp}.log")

logger = logging.getLogger("local_chat")
logger.setLevel(logging.INFO)

# File handler (detailed)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Thread %(threadName)s] %(message)s"))
logger.addHandler(fh)

# Console handler (simpler)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

logger.info("Starting server with model: %s", MODEL_ID)

# --- FastAPI ---
app = FastAPI()

# --- Load model/tokenizer ---
logger.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
logger.info("Model loaded successfully on device %s", next(model.parameters()).device)

# --- Queue & results ---
job_queue = Queue()
results = {}
lock = threading.Lock()

# --- Helpers ---
def inject_harmony_prompt(user_input: str) -> str:
    return (
        "<|channel|>system<|message|>"
        "You are an assistant that outputs text in Harmony format.\n"
        "Always include the following channels:\n"
        "<|channel|>analysis<|message|>Provide analysis or reasoning.\n"
        "<|channel|>final<|message|>Provide the final answer.\n\n"
        "<|channel|>user<|message|>" + user_input + "\n"
    )

def extract_harmony_channels(text: str) -> str:
    channels = []
    for part in text.split("<|channel|>"):
        if "<|message|>" in part:
            channels.append("<|channel|>" + part.strip())
    return "\n".join(channels)

# --- Worker ---
def model_worker():
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_id, prompt, max_tokens, multi_chunk, request_id = job
        start_time = time.time()
        try:
            prompt_with_harmony = inject_harmony_prompt(prompt) if HARMONY_MODE == "on" else prompt
            context = prompt_with_harmony
            full_text = ""
            tokens_generated = 0
            all_hidden_states = []
            all_attentions = []

            while tokens_generated < max_tokens:
                inputs = tokenizer(context, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generate_kwargs = {
                        "max_new_tokens": min(CHUNK_SIZE, max_tokens - tokens_generated),
                        "do_sample": args.do_sample,
                        "return_dict_in_generate": True,
                        "output_hidden_states": DEBUG_MODEL_DATA=="on",
                        "output_attentions": DEBUG_MODEL_DATA=="on"
                    }
                    if args.do_sample:
                        generate_kwargs.update({
                            "temperature": TEMPERATURE,
                            "top_k": TOP_K,
                            "top_p": TOP_P
                        })

                    outputs = model.generate(**inputs, **generate_kwargs)

                chunk = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                if chunk.startswith(context):
                    chunk = chunk[len(context):].strip()
                full_text += " " + chunk
                tokens_generated += len(chunk.split())
                context += " " + chunk

                # --- Corrected logging ---
                if args.log_mode == "full":
                    log_chunk = chunk
                else:
                    log_chunk = chunk[:args.log_preview_length]
                logger.info(f"[Worker] Job {job_id} | Chunk generated | Tokens so far: {tokens_generated} | Preview: {log_chunk}")

                if DEBUG_MODEL_DATA == "on":
                    all_hidden_states.append([h.cpu().tolist() for h in outputs.hidden_states])
                    all_attentions.append([a.cpu().tolist() for a in outputs.attentions])

                if len(chunk.split()) < min(CHUNK_SIZE, max_tokens - tokens_generated):
                    break

            generated_text = full_text.strip()
            harmony_output = extract_harmony_channels(generated_text) if HARMONY_MODE == "on" else ""

            finish_reason = "length" if tokens_generated >= max_tokens else "stop"
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
                    "prompt_raw": prompt_with_harmony
                }
                if DEBUG_MODEL_DATA == "on":
                    results[job_id]["hidden_states"] = all_hidden_states
                    results[job_id]["attentions"] = all_attentions

            logger.info(f"[Worker] FINISH Job {job_id} | Duration: {duration:.2f}s | Tokens: {tokens_generated}")

        except Exception as e:
            logger.exception(f"[Worker] ERROR Job {job_id} | Request ID: {request_id}")
            with lock:
                results[job_id] = {
                    "text": f"Error: {e}",
                    "harmony": f"<|channel|>analysis<|message|>Error\n<|channel|>final<|message|>Error: {e}",
                    "finish_reason": "error",
                    "duration": 0,
                    "prompt_length": len(prompt.split()),
                    "completion_length": 0,
                    "device": str(model.device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk
                }
        finally:
            job_queue.task_done()

# --- Start workers ---
for i in range(NUM_THREADS):
    threading.Thread(target=model_worker, daemon=True, name=f"Worker-{i}").start()

# --- Non-blocking submit ---
def submit_generation(prompt, max_tokens=MAX_TOKENS, multi_chunk=None, request_id=None):
    if multi_chunk is None:
        multi_chunk = DEFAULT_MULTI_CHUNK
    with lock:
        job_id = len(results)
    job_queue.put((job_id, prompt, max_tokens, multi_chunk, request_id))
    while True:
        with lock:
            if job_id in results:
                return results.pop(job_id)
        time.sleep(0.05)

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
    prompt_hash = hashlib.sha256(user_message.encode()).hexdigest()
    multi_chunk = req.multi_chunk if req.multi_chunk is not None else DEFAULT_MULTI_CHUNK

    logger.info(f"[API] Incoming request | Request ID: {request_id} | IP: {client_host} | UA: {user_agent} | Prompt hash: {prompt_hash}")
    logger.info(f"[API] User message preview: {user_message[:200]}")

    result = submit_generation(user_message, req.max_tokens, multi_chunk, request_id)
    content_out = result["harmony"] if HARMONY_MODE == "on" else result["text"]

    debug_data = {
        "duration_seconds": result["duration"],
        "device": result["device"],
        "client_host": client_host,
        "user_agent": user_agent,
        "request_id": request_id,
        "prompt_hash": prompt_hash,
        "multi_chunk": result["multi_chunk"],
        "harmony_raw": result["harmony"],
        "prompt_raw": result.get("prompt_raw"),
    }
    if DEBUG_MODEL_DATA == "on":
        debug_data["hidden_states"] = result.get("hidden_states")
        debug_data["attentions"] = result.get("attentions")

    logger.info(f"[API] Response ready | Request ID: {request_id} | Tokens: {result['completion_length']} | Finish reason: {result['finish_reason']}")

    return {
        "id": "local-chat-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content_out},
                     "finish_reason": result["finish_reason"]}],
        "usage": {
            "prompt_tokens": result["prompt_length"],
            "completion_tokens": result["completion_length"],
            "total_tokens": result["prompt_length"] + result["completion_length"],
        },
        "debug": debug_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
