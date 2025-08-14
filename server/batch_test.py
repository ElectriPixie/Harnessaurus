import requests
from threading import Thread

# --- Configuration ---
API_URL = "http://127.0.0.1:6589/v1/chat/completions"
PROMPTS = [
    "Write a short story about a robot learning to garden.",
    "Explain quantum entanglement in simple terms.",
    "List 5 healthy snacks for work.",
    "Summarize the plot of 'Alice in Wonderland'."
]

def send_prompt(prompt):
    payload = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256
    }
    try:
        resp = requests.post(API_URL, json=payload)
        data = resp.json()
        output = data["choices"][0]["message"]["content"]
        print(f"---\nPrompt: {prompt}\nResponse:\n{output}\n---")
    except Exception as e:
        print(f"Error sending prompt '{prompt}': {e}")

threads = []

for prompt in PROMPTS:
    t = Thread(target=send_prompt, args=(prompt,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
