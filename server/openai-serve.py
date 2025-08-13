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
NUM_THREADS = 4

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
lock = threading.Lock()  # protect results dict in multi-thread

def model_worker():
    while True:
        job_id, prompt, max_tokens = job_queue.get()
        if job_id is None:
            break

        start_time = time.time()
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

        finish_reason = "length" if len(generated_text.split()) >= max_tokens else "stop"
        duration = time.time() - start_time

        with lock:
            results[job_id] = {
                "text": generated_text,
                "finish_reason": finish_reason,
                "duration": duration,
                "prompt_length": len(prompt.split()),
                "completion_length": len(generated_text.split()),
                "device": str(model.device),
            }

        logger.info(
            "Job %s completed in %.2fs on device %s | Prompt tokens: %d | Completion tokens: %d | Finish reason: %s\nPrompt: %s\nGenerated: %s",
            job_id,
            duration,
            model.device,
            len(prompt.split()),
            len(generated_text.split()),
            finish_reason,
            prompt,
            generated_text,
        )

        job_queue.task_done()

# Launch worker threads
for i in range(NUM_THREADS):
    threading.Thread(target=model_worker, daemon=True, name=f"Worker-{i}").start()

def submit_generation(prompt, max_tokens=MAX_TOKENS):
    with lock:
        job_id = len(results)
    job_queue.put((job_id, prompt, max_tokens))
    job_queue.join()
    with lock:
        return results.pop(job_id)

# --- OpenAI-style API ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = MAX_TOKENS

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    user_message = req.messages[-1]["content"]
    client_host = request.client.host

    # --- Log incoming request ---
    logger.info(
        "Incoming request from %s | Model: %s | Max tokens: %d | Prompt tokens: %d\nPrompt: %s",
        client_host,
        req.model,
        req.max_tokens,
        len(user_message.split()),
        user_message,
    )

    # Submit to threaded GPU runner
    result = submit_generation(user_message, req.max_tokens)

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
        },
    }

# --- Run FastAPI server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6589)
