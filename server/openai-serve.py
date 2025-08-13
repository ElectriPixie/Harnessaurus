from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import threading
from queue import Queue
import uvicorn

# --- FastAPI app ---
app = FastAPI()

# --- Config ---
MODEL_ID = "/data/AI/Models/gpt-oss-20b"
MAX_TOKENS = 8096
NUM_THREADS = 4  # adjust for GPU saturation

# --- Load model and tokenizer ---
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

# --- Threaded job queue system ---
job_queue = Queue()
results = {}

def model_worker():
    while True:
        job_id, prompt, max_tokens = job_queue.get()
        if job_id is None:
            break

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate output with attentions & hidden states
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Decode text
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        # Save results
        results[job_id] = {
            "text": generated_text,
            "attentions": [tuple(a.shape) for a in outputs.attentions[0]],
            "hidden_states": [tuple(h.shape) for h in outputs.hidden_states[0]],
        }
        job_queue.task_done()

# Launch worker threads
for _ in range(NUM_THREADS):
    threading.Thread(target=model_worker, daemon=True).start()

# --- Submit prompt to queue and wait ---
def submit_generation(prompt, max_tokens=MAX_TOKENS):
    job_id = len(results)
    job_queue.put((job_id, prompt, max_tokens))
    job_queue.join()
    return results.pop(job_id)

# --- OpenAI-style request model ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = MAX_TOKENS

# --- API endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    user_message = req.messages[-1]["content"]

    # Submit to threaded GPU runner
    result = submit_generation(user_message, req.max_tokens)

    # Determine finish reason
    finish_reason = "length" if len(result["text"].split()) >= req.max_tokens else "stop"

    return {
        "id": "local-chat-1",
        "object": "chat.completion",
        "created": 0,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": finish_reason,
            }
        ],
        "analysis": {
            "attention_shapes": result["attentions"],
            "hidden_state_shapes": result["hidden_states"],
        },
        "usage": {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(result["text"].split()),
            "total_tokens": len(user_message.split()) + len(result["text"].split()),
        },
    }

# --- Run FastAPI ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)