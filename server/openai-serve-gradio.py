from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr
import threading
import uvicorn

# --- Initialize FastAPI ---
app = FastAPI()

# --- Configurable parameters ---
model_id = "/data/AI/Models/gpt-oss-20b"
MAX_TOKENS = 256  # max tokens for generation

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# --- Request model for OpenAI-style API ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = MAX_TOKENS

# --- API endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    user_message = req.messages[-1]["content"]

    # Tokenize input
    inputs = tokenizer(user_message, return_tensors="pt").to(model.device)

    # Generate text with attentions and hidden states
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # Decode generated text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Trim repeated prompt
    if generated_text.startswith(user_message):
        generated_text = generated_text[len(user_message):].strip()

    # Convert attentions and hidden states to shapes for lightweight info
    # (avoid returning full tensors unless you want huge JSON)
    attention_shapes = [
        tuple(layer.shape) for layer in outputs.attentions[0]  # first generation step
    ]
    hidden_state_shapes = [
        tuple(layer.shape) for layer in outputs.hidden_states[0]  # first generation step
    ]

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
        "analysis": {
            "attention_shapes": attention_shapes,
            "hidden_state_shapes": hidden_state_shapes,
        },
        "usage": {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(generated_text.split()),
            "total_tokens": len(user_message.split()) + len(generated_text.split()),
        },
    }

# --- Redirect root to Gradio ---
@app.get("/")
async def root():
    return RedirectResponse(url="/gradio")

# --- Gradio UI ---
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    return generated_text

iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=5, label="Enter prompt"),
    outputs=gr.Textbox(lines=15, label="Generated text"),
    title="GPT-OSS-20B Text Generation",
)

# --- Run Gradio in separate thread ---
def run_gradio():
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        prevent_thread_lock=True,
    )

threading.Thread(target=run_gradio, daemon=True).start()

# --- Run FastAPI ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)