from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Requantize and save a model in 4-bit using bitsandbytes")
    parser.add_argument("--model_name", required=True, help="Path or Hugging Face model repo ID")
    parser.add_argument("--save_dir", required=True, help="Directory to save the quantized model")
    args = parser.parse_args()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)

    print(f"Loading model from {args.model_name} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Saving quantized model to {args.save_dir}...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    print("Done.")

if __name__ == "__main__":
    main()
