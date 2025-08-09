import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

def quantize_mxfp4(x: torch.Tensor) -> torch.Tensor:
    """
    Simplified MXFP4 quantization function.
    """
    BLOCK_SIZE = 32

    x_flat = x.view(-1)
    n_blocks = (x_flat.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded_len = n_blocks * BLOCK_SIZE

    if x_flat.numel() < padded_len:
        pad_size = padded_len - x_flat.numel()
        x_flat = torch.cat([x_flat, torch.zeros(pad_size, dtype=x_flat.dtype, device=x_flat.device)])

    x_blocks = x_flat.view(n_blocks, BLOCK_SIZE)
    scales = x_blocks.abs().max(dim=1)[0].clamp(min=1e-8)
    normalized = x_blocks / scales.unsqueeze(1)
    quantized_blocks = torch.round((normalized + 1) * 7.5).clamp(0, 15).to(torch.uint8)
    quantized = quantized_blocks.view(-1)[:x.numel()]
    return quantized

def load_weights(model_dir: Path):
    weights = {}
    for f in model_dir.glob("*.safetensors"):
        shard_weights = load_file(f)
        weights.update({k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in shard_weights.items()})
    return weights

def save_weights(weights: dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model_quantized.safetensors"
    save_file(weights, str(save_path))
    print(f"Saved quantized model to {save_path}")

def requantize(weights: dict):
    quantized_weights = {}
    for k, v in weights.items():
        if "weight" in k and isinstance(v, torch.Tensor) and v.dim() == 2:
            quantized_weights[k] = quantize_mxfp4(v)
        else:
            quantized_weights[k] = v
    return quantized_weights

def main():
    parser = argparse.ArgumentParser(description="Requantize model weights to MXFP4.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to directory with original safetensors model shards.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save quantized model.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    save_dir = Path(args.save_dir)

    print(f"Loading weights from {model_dir}...")
    weights = load_weights(model_dir)

    print("Quantizing weights...")
    quantized = requantize(weights)

    print(f"Saving quantized weights to {save_dir}...")
    save_weights(quantized, save_dir)

if __name__ == "__main__":
    main()