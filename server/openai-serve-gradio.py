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
MAX_TOKENS = 256

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# Store last prompt/output for CoT display
last_prompt = None
last_outputs = None

# --- FastAPI OpenAI-style API ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = MAX_TOKENS

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global last_prompt, last_outputs
    user_message = req.messages[-1]["content"]

    inputs = tokenizer(user_message, return_tensors="pt").to(model.device)

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

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    if generated_text.startswith(user_message):
        generated_text = generated_text[len(user_message):].strip()

    # Save last prompt and outputs for CoT display
    last_prompt = user_message
    last_outputs = outputs

    return {
        "id": "local-chat-1",
        "object": "chat.completion",
        "created": 0,
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": generated_text}, "finish_reason": "stop"}],
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

# --- Gradio UI functions ---
def generate_text(prompt):
    global last_prompt, last_outputs
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

    last_prompt = prompt
    last_outputs = outputs

    return generated_text

def get_cot_details():
    if last_outputs is None:
        return "No CoT data available. Generate text first."
    
    # Convert tensors to lists for JSON-friendly display
    attention_shapes = [[list(layer.shape) for layer in step] for step in last_outputs.attentions]
    hidden_state_shapes = [[list(layer.shape) for layer in step] for step in last_outputs.hidden_states]

    return f"Prompt: {last_prompt}\n\nAttention shapes: {attention_shapes}\n\nHidden state shapes: {hidden_state_shapes}"

# --- Gradio interface ---
with gr.Blocks() as iface:
    gr.Markdown("# GPT-OSS-20B Text Generation")
    
    txt_input = gr.Textbox(lines=5, label="Enter prompt")
    txt_output = gr.Textbox(lines=15, label="Generated text")
    btn_generate = gr.Button("Generate")
    btn_cot = gr.Button("See CoT / Analysis")
    cot_output = gr.Textbox(lines=20, label="Chain of Thought / Analysis")

    btn_generate.click(fn=generate_text, inputs=txt_input, outputs=txt_output)
    btn_cot.click(fn=get_cot_details, inputs=None, outputs=cot_output)

# --- Run Gradio in separate thread ---
def run_gradio():
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False, prevent_thread_lock=True)

threading.Thread(target=run_gradio, daemon=True).start()

# --- Run FastAPI ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)