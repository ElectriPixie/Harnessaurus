from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- Load model & tokenizer ---
MODEL_ID = "/data/AI/Models/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
special_tokens = ["<|role|>", "<|/role|>", "<|channel|>", "<|message|>", "<|return|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
model.resize_token_embeddings(len(tokenizer))

# --- Deterministic ---
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# --- Harmony prompt wrapper ---
def inject_harmony(prompt_text, role="user"):
    return f"<|role|>{role}<|message|>{prompt_text}<|/role|><|role|>system<|message|>You are an assistant that outputs text in Harmony format.<|/role|>"

# --- Function to generate output for a single prompt ---
def generate_for_prompt(prompt_text, max_tokens=1024):
    input_text = inject_harmony(prompt_text)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.1,
        top_k=1,
        do_sample=False  # ensures deterministic
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def parse_harmony_channels(harmony_text):
    """
    Extract 'analysis' and 'final' channels from Harmony output.
    Fallbacks to defaults if missing.
    """
    channels = {"analysis": "We must refuse.", "final": "I’m sorry, but I can’t help with that."}

    # Split on channel tags
    matches = re.findall(r"<\|channel\|>(.*?)<\|message\|>(.*?)(?=<\|channel\||<\|/role\|>)", harmony_text, re.DOTALL)
    for name, content in matches:
        name = name.strip().lower()
        content = content.strip()
        if name in channels:
            channels[name] = content
    return channels

def save_harmony_finding(prompt_text, harmony_text, filename_prefix="myteamname.findings"):
    """
    Save a single prompt + harmony output as a Kaggle findings JSON file.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    finding_id = str(uuid.uuid4())
    
    channels = parse_harmony_channels(harmony_text)

    finding = {
        "id": finding_id,
        "created": timestamp,
        "prompt": prompt_text,
        "output": harmony_text,
        "channels": channels,  # Parsed channels
        "severity": None,      # Fill in your self-assessed Severity
        "breadth": None,       # Fill in your self-assessed Breadth
        "novelty": None,       # Fill in your self-assessed Novelty
        "notes": ""            # Optional analysis or reasoning
    }

    filename = f"{filename_prefix}.{finding_id}.json"
    with open(filename, "w") as f:
        json.dump(finding, f, indent=2)
    print(f"[Saved] {filename}")
    return filename