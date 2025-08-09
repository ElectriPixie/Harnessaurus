import argparse
import os
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Requantize and save a model in 4-bit using AutoGPTQ")
    parser.add_argument("--model_name", required=True, help="Path or Hugging Face model repo ID")
    parser.add_argument("--save_dir", required=True, help="Directory to save the quantized model")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading model from {args.model_name} for quantization...")
    try:
        quant_config = BaseQuantizeConfig(quantize_mode="4bit")
        model = AutoGPTQForCausalLM.from_pretrained(
            args.model_name,
            quantize_config=quant_config,
            device="auto",
            use_safetensors=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Saving quantized model to {args.save_dir}...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    print("Done.")

if __name__ == "__main__":
    main()