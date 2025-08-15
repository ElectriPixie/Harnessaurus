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
import math
from datetime import datetime

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="OpenAI-style local server with multi-chunk + harmony output")
parser.add_argument("--model_id", type=str, default="/data/AI/Models/gpt-oss-20b")
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--chunk_size", type=int, default=1024)
parser.add_argument("--num_threads", type=int, default=4)
parser.add_argument("--multi_chunk", action="store_true", default=False)
parser.add_argument("--harmony", choices=["off", "on"], default="on")
parser.add_argument("--debug_model_data", choices=["off", "on"], default="off")
parser.add_argument("--port", type=int, default=6589)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--do_sample", action="store_true", default=False)
parser.add_argument("--deterministic", choices=["on", "off"], default="on", help="Enable deterministic outputs")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--log_mode", choices=["full", "preview"], default="full")
parser.add_argument("--log_preview_length", type=int, default=200)
parser.add_argument(
    "--harmony_newlines",
    choices=["on", "off"],
    default="off",
    help="Add newline characters after Harmony special tokens (default: off)"
)
parser.add_argument(
    "--fill_missing_channels",
    action="store_true",
    default=False,
    help="If set, missing harmony channels are automatically filled with default text"
)
# --- NEW repetition control options ---
parser.add_argument(
    "--repetition_control",
    choices=["on", "off"],
    default="on",
    help="Enable or disable repetition control (no-repeat-ngram + repetition penalty)"
)
parser.add_argument(
    "--no_repeat_ngram_size",
    type=int,
    default=3,
    help="Size of n-grams to avoid repeating (set to 0 to disable)"
)
parser.add_argument(
    "--repetition_penalty",
    type=float,
    default=1.1,
    help="Penalty factor for repetition (1.0 = no penalty)"
)

args = parser.parse_args()

FILL_MISSING_CHANNELS = args.fill_missing_channels
ADD_HARMONY_NEWLINES = args.harmony_newlines.lower() == "on"
DETERMINISTIC = args.deterministic.lower() == "on"
REPETITION_CONTROL = args.repetition_control.lower() == "on"

# --- Deterministic setup ---
if DETERMINISTIC:
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    logging.info(f"Deterministic mode ON | Seed: {args.seed}")
else:
    torch.use_deterministic_algorithms(False)
    logging.info("Deterministic mode OFF | Using PyTorch default RNG")

# --- Config ---
MODEL_ID = args.model_id
MAX_TOKENS = args.max_tokens
CHUNK_SIZE = args.chunk_size
NUM_THREADS = args.num_threads
DEFAULT_MULTI_CHUNK = args.multi_chunk
HARMONY_MODE = args.harmony
DEBUG_MODEL_DATA = args.debug_model_data == "on"
TEMPERATURE = args.temperature
TOP_K = args.top_k
TOP_P = args.top_p

# --- Logging setup ---
os.makedirs(args.log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(args.log_dir, f"generation_{timestamp}.log")
logger = logging.getLogger("local_chat")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Thread %(threadName)s] %(message)s"))
logger.addHandler(fh)
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
special_tokens = ["<|role|>", "<|/role|>", "<|channel|>", "<|message|>", "<|return|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
model.resize_token_embeddings(len(tokenizer))
logger.info("Model loaded successfully on device %s", next(model.parameters()).device)

# --- Queue & results ---
job_queue = Queue()
results = {}
lock = threading.Lock()

def inject_harmony_prompt(messages: list) -> str:
    sep = "\n" if ADD_HARMONY_NEWLINES else ""
    harmony_prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        harmony_prompt += f"<|role|>{role}{sep}<|message|>{content}{sep}<|/role|>{sep}"
    harmony_prompt += (
        f"<|role|>system{sep}<|message|>"
        "You are an assistant that outputs text in Harmony format.\n"
        "Always include <|channel|>analysis and <|channel|>final channels in assistant outputs."
        f"{sep}<|/message|>{sep}<|/role|>{sep}"
    )
    return harmony_prompt

def format_harmony_output(raw_text: str) -> str:
    sep = "\n" if ADD_HARMONY_NEWLINES else ""
    channels = {}
    for part in raw_text.split("<|channel|>"):
        if "<|message|>" in part:
            name, content = part.split("<|message|>", 1)
            channels[name.strip()] = content.strip()

    if FILL_MISSING_CHANNELS:
        if "analysis" not in channels or not channels["analysis"]:
            channels["analysis"] = "We must refuse."
        if "final" not in channels or not channels["final"]:
            channels["final"] = "I’m sorry, but I can’t help with that."

    assistant_text = f"<|role|>assistant{sep}"
    for ch_name, content in channels.items():
        assistant_text += f"<|channel|>{ch_name}<|message|>{content}{sep}"
    assistant_text += f"<|/role|>{sep}<|return|>"
    return assistant_text

# --- Worker ---
def model_worker():
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_id, messages, max_tokens, multi_chunk, request_id = job
        start_time = time.time()
        try:
            prompt_with_harmony = inject_harmony_prompt(messages) if HARMONY_MODE == "on" else messages[-1]["content"]
            context = prompt_with_harmony
            full_text = ""
            tokens_generated = 0
            all_hidden_states, all_attentions = [], []

            max_iterations = math.ceil(max_tokens / CHUNK_SIZE) if multi_chunk else 1

            for _ in range(max_iterations):
                inputs = tokenizer(context, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    do_sample_flag = args.do_sample and not DETERMINISTIC
                    generate_kwargs = {
                        "max_new_tokens": min(CHUNK_SIZE, max_tokens - tokens_generated),
                        "do_sample": do_sample_flag,
                        "return_dict_in_generate": True,
                        "output_hidden_states": DEBUG_MODEL_DATA,
                        "output_attentions": DEBUG_MODEL_DATA
                    }
                    if do_sample_flag:
                        generate_kwargs.update({
                            "temperature": TEMPERATURE,
                            "top_k": TOP_K,
                            "top_p": TOP_P
                        })
                    # --- Apply repetition control if enabled ---
                    if REPETITION_CONTROL:
                        if args.no_repeat_ngram_size > 0:
                            generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
                        if args.repetition_penalty != 1.0:
                            generate_kwargs["repetition_penalty"] = args.repetition_penalty

                    outputs = model.generate(**generate_kwargs, **inputs)

                prev_len = inputs.input_ids.shape[1]
                new_tokens = outputs.sequences[0][prev_len:]
                new_chunk = tokenizer.decode(new_tokens, skip_special_tokens=False)

                full_text += new_chunk
                tokens_generated += len(new_chunk.split())

                if DEBUG_MODEL_DATA:
                    all_hidden_states.append([h.cpu().tolist() for h in outputs.hidden_states])
                    all_attentions.append([a.cpu().tolist() for a in outputs.attentions])

                if multi_chunk:
                    context += new_chunk

                if len(new_tokens) < min(CHUNK_SIZE, max_tokens - tokens_generated):
                    break

            finish_reason = "length" if tokens_generated >= max_tokens else "stop"
            duration = time.time() - start_time
            content_out = format_harmony_output(full_text) if HARMONY_MODE == "on" else full_text.strip()

            with lock:
                results[job_id] = {
                    "text": full_text.strip(),
                    "harmony": content_out,
                    "finish_reason": finish_reason,
                    "duration": duration,
                    "prompt_length": sum(len(m["content"].split()) for m in messages),
                    "completion_length": len(full_text.split()),
                    "device": str(model.device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk,
                    "prompt_raw": prompt_with_harmony
                }
                if DEBUG_MODEL_DATA:
                    results[job_id]["hidden_states"] = all_hidden_states
                    results[job_id]["attentions"] = all_attentions

        except Exception as e:
            logger.exception(f"[Worker] ERROR Job {job_id} | Request ID: {request_id}")
            with lock:
                results[job_id] = {
                    "text": f"Error: {e}",
                    "harmony": "<|role|>assistant\n<|channel|>analysis<|message|>Error\n<|channel|>final<|message|>Error\n<|/role|>\n<|return|>",
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
for i in range(NUM_THREADS):
    threading.Thread(target=model_worker, daemon=True, name=f"Worker-{i}").start()

# --- Submit job ---
def submit_generation(messages, max_tokens=MAX_TOKENS, multi_chunk=None, request_id=None):
    if multi_chunk is None:
        multi_chunk = DEFAULT_MULTI_CHUNK
    with lock:
        job_id = len(results)
    job_queue.put((job_id, messages, max_tokens, multi_chunk, request_id))
    while True:
        with lock:
            if job_id in results:
                return results.pop(job_id)
        time.sleep(0.05)

# --- FastAPI Request ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = MAX_TOKENS
    multi_chunk: bool = None

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    client_host = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")
    request_id = str(uuid.uuid4())
    prompt_hash = hashlib.sha256("".join([m["content"] for m in req.messages]).encode()).hexdigest()
    multi_chunk = req.multi_chunk if req.multi_chunk is not None else DEFAULT_MULTI_CHUNK

    logger.info(f"[API] Incoming request | Request ID: {request_id} | IP: {client_host} | UA: {user_agent} | Prompt hash: {prompt_hash}")
    logger.info(f"[API] Last user message preview: {req.messages[-1]['content'][:200]}")

    result = submit_generation(req.messages, req.max_tokens, multi_chunk, request_id)
    content_out = result["harmony"] if HARMONY_MODE == "on" else result["text"]

    logger.info(f"[API] Response ready | Request ID: {request_id} | Tokens: {result['completion_length']} | Finish reason: {result['finish_reason']}\n{content_out}")

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
    if DEBUG_MODEL_DATA:
        debug_data["hidden_states"] = result.get("hidden_states")
        debug_data["attentions"] = result.get("attentions")

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

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
