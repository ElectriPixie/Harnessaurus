import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

# I couldn't find an official way to try and store these models requantized using MXFP4 so I pulled some stuff out of the openai/gpt-oss
# code and reused it in a standalone script

# ---- Official MXFP4 quantization helpers from GPT-OSS moe.py ----

def float_to_e2m1(f: torch.Tensor) -> torch.Tensor:
    """
    Convert float tensor to 4-bit E2M1 format (uint8 0-15).
    """
    sign = (f < 0).to(torch.uint8)
    abs_f = f.abs()
    exp = torch.zeros_like(abs_f, dtype=torch.uint8)
    man = torch.zeros_like(abs_f, dtype=torch.uint8)

    # Define thresholds for exponent bits (biased by 1)
    # Exponent range 0-3 (2 bits)
    # This is a simplified example, based on official repo constants
    exp[abs_f < 0.5] = 0
    exp[(abs_f >= 0.5) & (abs_f < 1)] = 1
    exp[(abs_f >= 1) & (abs_f < 2)] = 2
    exp[abs_f >= 2] = 3

    # Mantissa is one bit, calculated by whether fractional part exceeds 0.5 threshold
    man[abs_f >= 1.5] = 1  # simplified threshold example

    # Pack into 4 bits: s e e m
    encoded = (sign << 3) | (exp << 1) | man
    return encoded

def quantize_mxfp4(x: torch.Tensor):
    """
    Quantize float tensor x into MXFP4 format:
    - Split into blocks of 32
    - Calculate scale per block
    - Normalize and quantize to 4-bit E2M1
    - Pack into bytes (2 values per byte)
    Returns:
      - packed quantized bytes tensor (uint8)
      - scales tensor (float16)
    """
    BLOCK_SIZE = 32
    x_flat = x.view(-1)
    n_blocks = (x_flat.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded_len = n_blocks * BLOCK_SIZE
    pad_size = padded_len - x_flat.numel()
    if pad_size > 0:
        x_flat = torch.cat([x_flat, torch.zeros(pad_size, dtype=x.dtype, device=x.device)])

    x_blocks = x_flat.view(n_blocks, BLOCK_SIZE)

    scales = x_blocks.abs().max(dim=1)[0].clamp(min=1e-8).to(torch.float16)

    normalized = x_blocks / scales.unsqueeze(1)

    # Quantize to 4-bit E2M1 encoding
    quantized = float_to_e2m1(normalized).to(torch.uint8)

    # Pack pairs of 4-bit values into one byte (little-endian: low 4 bits first)
    quantized = quantized.view(-1)
    if quantized.numel() % 2 != 0:
        quantized = torch.cat([quantized, torch.zeros(1, dtype=torch.uint8, device=quantized.device)])
    quantized_packed = quantized[0::2] | (quantized[1::2] << 4)

    return quantized_packed, scales

# ---- Model loading, saving, and processing ----

def load_weights(model_dir: Path):
    weights = {}
    for f in model_dir.glob("*.safetensors"):
        shard_weights = load_file(f)
        # Convert to torch tensors if needed
        weights.update({k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in shard_weights.items()})
    return weights

def save_quantized_model(weights_quantized: dict, scales: dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    def to_tensor_compatible(tensor):
        if tensor.dtype == torch.uint8:
            return tensor.cpu()
        elif tensor.dtype in [torch.float16, torch.float32]:
            return tensor.cpu()
        else:
            # Convert any other dtype (e.g. bfloat16) to float16 for saving, but keep as tensor
            return tensor.cpu().to(torch.float16)

    save_quantized = {k: to_tensor_compatible(v) for k, v in weights_quantized.items()}
    save_scales = {k + "_scale": v.cpu().to(torch.float16) for k, v in scales.items()}

    combined = {**save_quantized, **save_scales}

    save_path = save_dir / "model_quantized.safetensors"
    save_file(combined, str(save_path))
    print(f"Saved quantized model and scales to {save_path}")

def requantize(weights: dict):
    quantized_weights = {}
    scales = {}

    for k, v in weights.items():
        if "weight" in k and isinstance(v, torch.Tensor) and v.dim() == 2:
            q, s = quantize_mxfp4(v)
            quantized_weights[k] = q
            scales[k] = s
        else:
            quantized_weights[k] = v

    return quantized_weights, scales

def main():
    parser = argparse.ArgumentParser(description="Quantize GPT-OSS model weights to MXFP4.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with original model safetensors shards.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save quantized model.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    save_dir = Path(args.save_dir)

    print(f"Loading weights from {model_dir}...")
    weights = load_weights(model_dir)

    print("Quantizing weights with MXFP4...")
    quantized_weights, scales = requantize(weights)

    print(f"Saving quantized weights and scales to {save_dir}...")
    save_quantized_model(quantized_weights, scales, save_dir)

if __name__ == "__main__":
    main()