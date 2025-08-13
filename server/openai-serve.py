from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "/data/AI/Models/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

prompt = "Explain quantum mechanics simply."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Enable attentions and hidden states for analysis
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,  # <- important to get the internals
    )

generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

# You can now access:
attentions = outputs.attentions          # list of attention tensors per layer
hidden_states = outputs.hidden_states    # list of hidden state tensors per layer

print("Generated:", generated_text)
print("Attention shape (layer 0):", attentions[0].shape)
print("Hidden state shape (layer 0):", hidden_states[0].shape)