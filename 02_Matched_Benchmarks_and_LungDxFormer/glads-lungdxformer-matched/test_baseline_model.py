
import sys
from pathlib import Path
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/LungDxFormer/src")

import torch
from lungdxformer.models.lungdxformer import LungDxFormer

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = LungDxFormer(
    image_size=64,
    in_channels=1,
    num_classes=3,
    cnn_channels=(32, 64, 128),
    embed_dim=128,
    transformer_heads=4,
    transformer_layers=2,
    transformer_mlp_ratio=2.0,
    dropout=0.1,
    use_transformer=True,
    use_positional_encoding=True,
    use_spatial_attention=True,
    fusion_type="concat",
    classifier_hidden_dim=128,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Forward pass test with dummy batch of size 32
x = torch.randn(32, 1, 64, 64, device=device)
with torch.no_grad():
    out = model(x)

print("Forward pass successful!")
print("Logits shape:", out["logits"].shape)
print("Probs shape:", out["probs"].shape)
print("Attention map shape:", out["attention_map"].shape)
print("Fused features shape:", out["fused_features"].shape)
