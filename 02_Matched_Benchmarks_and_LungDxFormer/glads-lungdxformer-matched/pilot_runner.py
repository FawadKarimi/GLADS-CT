
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/LungDxFormer/src")
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")

from dataset import LIDCSliceDataset
from lungdxformer.models.lungdxformer import LungDxFormer
from glads_slice_model import GLADS_CT_Slice

# Ensure GPU 1 is selected
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Executing Pilot on device:", torch.cuda.get_device_name(0))

# Load split manifest
df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")

# Use split_slice for the matched slice experiment (or split_patient)
# Let's test on split_slice (2991 train, 641 val)
train_df = df[df["split_slice"] == "train"]
val_df = df[df["split_slice"] == "val"]

train_loader = DataLoader(LIDCSliceDataset(train_df, is_train=True), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(LIDCSliceDataset(val_df, is_train=False), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Calculate paper's inverse-frequency class weights
counts = train_df["label"].value_counts().sort_index().values
total_samples = len(train_df)
class_weights = torch.tensor([total_samples / (3.0 * c) for c in counts], dtype=torch.float32, device=device)
print(f"Train counts: {counts}, Class weights: {class_weights.tolist()}")

results = {}

for model_name, model_class in [("LungDxFormer_B0", LungDxFormer), ("GLADS_CT_Slice", GLADS_CT_Slice)]:
    print(f"\n=== Testing {model_name} (3 Epochs Pilot) ===")
    torch.cuda.reset_peak_memory_stats()
    
    if model_name == "LungDxFormer_B0":
        model = model_class(image_size=64, num_classes=3, fusion_type="concat").to(device)
    else:
        model = model_class(image_size=64, num_classes=3).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    epoch_times = []
    
    for epoch in range(1, 4):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            rt = batch["reader_target"].to(device, non_blocking=True)
            disag = batch["disagreement"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(x)
                logits = out["logits"]
                
                if model_name == "GLADS_CT_Slice":
                    # GLADS-CT Disagreement-Aware Supervision
                    # target = (1 - u) * one_hot + u * reader_target
                    u = disag.unsqueeze(1)
                    target = (1.0 - u) * F.one_hot(y, 3).float() + u * rt
                    mass = target * class_weights.unsqueeze(0)
                    log_d = F.log_softmax(logits, dim=1)
                    loss = -(mass * log_d).sum() / mass.sum().clamp_min(1e-8)
                else:
                    # Baseline Weighted Cross Entropy
                    loss = F.cross_entropy(logits, y, weight=class_weights)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * len(y)
            
        train_loss /= len(train_df)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    out = model(x)
                    logits = out["logits"]
                    loss = F.cross_entropy(logits, y, weight=class_weights)
                val_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
                
        val_loss /= len(val_df)
        val_acc = correct / total
        dt = time.time() - t0
        epoch_times.append(dt)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {dt:.2f}s")
        
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    results[model_name] = {
        "avg_epoch_seconds": float(np.mean(epoch_times)),
        "peak_vram_mb": peak_vram_mb,
        "final_val_acc": val_acc
    }
    print(f"{model_name} -> Avg Epoch Time: {np.mean(epoch_times):.2f}s | Peak VRAM: {peak_vram_mb:.1f} MiB")

print("\n=== PILOT SUMMARY ===")
print(json.dumps(results, indent=2))
with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/pilot_results.json", "w") as f:
    json.dump(results, f, indent=2)
