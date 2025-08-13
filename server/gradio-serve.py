import gradio as gr
import requests

# --- Config ---
SERVER_URL = "http://localhost:8000"
MODEL_NAME = "gpt-oss-20b"
MAX_TOKENS = 8096

# --- Helper to call the FastAPI server ---
def query_api(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
    }
    try:
        resp = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        generated_text = data["choices"][0]["message"]["content"]
        return generated_text
    except Exception as e:
        return f"Error: {e}"

# --- Gradio UI ---
def generate_text_ui(prompt):
    return query_api(prompt)

iface = gr.Interface(
    fn=generate_text_ui,
    inputs=gr.Textbox(lines=5, label="Enter prompt"),
    outputs=gr.Textbox(lines=15, label="Generated text"),
    title="GPT-OSS-20B Text Generation",
)

# --- Launch Gradio ---
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)