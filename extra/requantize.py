from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Saving quantized model...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

if __name__ == "__main__":
    main()
