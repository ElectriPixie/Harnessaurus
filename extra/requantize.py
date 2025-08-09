import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Load and save 4-bit quantized model")
    parser.add_argument('--model_name', required=True, help='Model name or local path')
    parser.add_argument('--save_dir', required=True, help='Directory to save the quantized model')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        load_in_4bit=True
    )

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Quantized model and tokenizer saved to: {args.save_dir}")

if __name__ == "__main__":
    main()