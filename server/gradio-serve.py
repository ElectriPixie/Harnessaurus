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
        analysis = data.get("analysis", {})
        return generated_text, analysis
    except Exception as e:
        return f"Error: {e}", {}

# --- Gradio UI ---
def generate_text_ui(prompt, show_analysis=False):
    text, analysis = query_api(prompt)
    if show_analysis and analysis:
        analysis_text = (
            f"Attention shapes:\n{analysis.get('attention_shapes')}\n\n"
            f"Hidden state shapes:\n{analysis.get('hidden_state_shapes')}"
        )
    else:
        analysis_text = ""
    return text, analysis_text

iface = gr.Interface(
    fn=generate_text_ui,
    inputs=[
        gr.Textbox(lines=5, label="Enter prompt"),
        gr.Checkbox(label="Show CoT / Analysis"),
    ],
    outputs=[
        gr.Textbox(lines=15, label="Generated text"),
        gr.Textbox(lines=15, label="CoT / Analysis"),
    ],
    title="GPT-OSS-20B Text Generation with CoT",
)

# --- Launch Gradio ---
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)