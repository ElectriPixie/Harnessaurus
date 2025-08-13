from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from transformers import pipeline
import torch
import gradio as gr
import threading
import uvicorn

# --- Initialize FastAPI application ---
app = FastAPI()

# Path to your local GPT-OSS model
model_id = "/data/AI/Models/gpt-oss-20b"

print("Loading model...")

# --- Load model and tokenizer via HuggingFace pipeline ---
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",  # Automatically chooses a safe dtype based on your GPU/hardware.
    device_map="auto",   # Automatically maps model layers to available devices (GPU/CPU).
)

# --- Define request model for OpenAI-style API ---
class ChatCompletionRequest(BaseModel):
    model: str               # Model name / identifier from the request
    messages: list           # List of messages in OpenAI chat format
    max_tokens: int = 256    # Max tokens for the response (default 256)

# --- API endpoint to mimic OpenAI /v1/chat/completions ---
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # Take the last user message from the messages array
    user_message = req.messages[-1]["content"]

    # Generate text using the pipeline safely
    # Sampling parameters help avoid NaNs and keep generation coherent:
    #   do_sample=True allows stochastic generation
    #   temperature=0.7 controls randomness (lower = more deterministic)
    #   top_k=50 and top_p=0.95 limit extreme logits for stability
    outputs = pipe(
        user_message,
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    # Extract generated text
    generated_text = outputs[0]["generated_text"]

    # If the model repeated the prompt, trim it out
    if generated_text.startswith(user_message):
        generated_text = generated_text[len(user_message):].strip()

    # Return a JSON response in OpenAI-compatible format
    return {
        "id": "local-chat-1",
        "object": "chat.completion",
        "created": 0,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_message.split()),              # Rough token estimate
            "completion_tokens": len(generated_text.split()),       # Rough token estimate
            "total_tokens": len(user_message.split()) + len(generated_text.split()),
        },
    }

# --- Redirect root path to Gradio UI for convenience ---
@app.get("/")
async def root():
    return RedirectResponse(url="/gradio")

# --- Gradio UI for interactive text generation ---
def generate_text(prompt):
    # Generate text similarly to the API endpoint
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs[0]["generated_text"]

# Define Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=5, label="Enter prompt"),
    outputs=gr.Textbox(lines=15, label="Generated text"),
    title="GPT-OSS-20B Text Generation",
)

# --- Run Gradio in a separate thread ---
# This allows FastAPI to serve API requests simultaneously
def run_gradio():
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        prevent_thread_lock=True  # Avoids blocking the main thread
    )

threading.Thread(target=run_gradio, daemon=True).start()

# --- Run FastAPI server via uvicorn ---
if __name__ == "__main__":
    # FastAPI will listen on port 8000 for API calls
    uvicorn.run(app, host="0.0.0.0", port=8000)