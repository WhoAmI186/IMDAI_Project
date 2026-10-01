import json
import glob
import os

p = glob.glob(os.path.expandvars(r'%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen3-8B\snapshots\*\config.json'))
cfg = json.load(open(p[0])) if p else {}

vocab_size = cfg.get("vocab_size", 151936)
hidden_size = cfg.get("hidden_size", 4096)
num_layers = cfg.get("num_hidden_layers", 36)
num_heads = cfg.get("num_attention_heads", 32)
num_kv_heads = cfg.get("num_key_value_heads", 8)
intermediate_size = cfg.get("intermediate_size", 12288)

# Embeddings: embed_tokens + lm_head
embed_params = vocab_size * hidden_size * 2  # 2 for embed + lm_head

# Per layer:
q_params = hidden_size * (num_heads * (hidden_size // num_heads))
k_params = hidden_size * (num_kv_heads * (hidden_size // num_heads))
v_params = hidden_size * (num_kv_heads * (hidden_size // num_heads))
o_params = hidden_size * hidden_size
gate_params = hidden_size * intermediate_size
up_params = hidden_size * intermediate_size
down_params = intermediate_size * hidden_size
layer_params = q_params + k_params + v_params + o_params + gate_params + up_params + down_params

total_linear_params = layer_params * num_layers
total_params = total_linear_params + embed_params

print("=" * 70)
print(f"MODEL ARCHITECTURE PARAMETER ANALYSIS FOR QWEN3-8B")
print("=" * 70)
print(f"Total Parameters: {total_params / 1e9:.2f} Billion")
print(f"  - Linear Layer Parameters: {total_linear_params / 1e9:.2f} Billion")
print(f"  - Embedding & LM-Head Parameters: {embed_params / 1e9:.2f} Billion")

# Memory footprint:
# In 16-bit (FP16/BF16):
fp16_bytes = total_params * 2
print(f"\n1. Full Precision (FP16/BF16):")
print(f"   Model Weights Size: {fp16_bytes / (1024**3):.2f} GB")

# In 4-bit (NF4 BitsAndBytes):
# Standard BitsAndBytes leaves embeddings and lm_head in FP16 (2 bytes) for numerical stability,
# and quantizes linear layers to 4-bit (0.5 bytes) + ~0.06 bytes double-quant overhead.
bnb_4bit_bytes = (total_linear_params * 0.56) + (embed_params * 2.0)
print(f"\n2. BitsAndBytes 4-bit Quantization (Linear layers 4-bit NF4, Embeddings FP16):")
print(f"   Model Weights VRAM: {bnb_4bit_bytes / (1024**3):.2f} GB")
print(f"   PyTorch CUDA Runtime / Context Overhead: ~0.50 GB")
print(f"   KV Cache (1024 tokens, batch=1): ~0.35 GB")
print(f"   Total Active VRAM Required: {(bnb_4bit_bytes / (1024**3)) + 0.85:.2f} GB")

print("\n" + "=" * 70)
print("GPU HARDWARE RESOURCE BUDGET (RTX 4050 Laptop)")
print("=" * 70)
print("Total Physical VRAM: 6.00 GB (6,141 MiB)")
print("WDDM OS / Desktop VRAM Allocation: ~0.87 GB")
print("Maximum Usable Dedicated VRAM: ~5.13 GB")
print(f"Deficit / Over-budget: {((bnb_4bit_bytes / (1024**3)) + 0.85) - 5.13:.2f} GB")
print("=" * 70)
