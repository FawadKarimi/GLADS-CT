
import sys
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/LungDxFormer/src")
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")

import torch
from glads_slice_model import GLADS_CT_Slice

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = GLADS_CT_Slice(image_size=64).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"GLADS_CT_Slice total parameters: {total_params:,}")

x = torch.randn(32, 1, 64, 64, device=device)
with torch.no_grad():
    out = model(x)

print("Forward pass successful!")
print("Logits shape:", out["logits"].shape)
print("Gate map shape:", out["gate_map"].shape)
print("Attention map shape:", out["attention_map"].shape)
