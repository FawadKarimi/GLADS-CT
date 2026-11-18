
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/LungDxFormer/src")
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")

from dataset import LIDCSliceDataset
from glads_slice_model import GLADS_CT_Slice
from lungdxformer.models.lungdxformer import LungDxFormer

device = torch.device("cuda:1")

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")

# Filter out indeterminate (label == 1), keep only Benign (0) and Malignant (2)
polar_df = df[df["label"] != 1].copy()
# Remap label: 0 -> 0 (Benign), 2 -> 1 (Malignant)
polar_df["label"] = (polar_df["label"] == 2).astype(int)

train_df = polar_df[polar_df["split_slice"] == "train"].reset_index(drop=True)
val_df = polar_df[polar_df["split_slice"] == "val"].reset_index(drop=True)
test_df = polar_df[polar_df["split_slice"] == "test"].reset_index(drop=True)

print(f"Polar Cohort: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Total: {len(polar_df)})")

train_loader = DataLoader(LIDCSliceDataset(train_df, is_train=True), batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(LIDCSliceDataset(val_df, is_train=False), batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(LIDCSliceDataset(test_df, is_train=False), batch_size=32, shuffle=False, num_workers=2)

# Train GLADS-CT on Polar (Binary)
model = GLADS_CT_Slice(image_size=64, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
scaler = torch.cuda.amp.GradScaler()

best_val_acc = 0.0
best_epoch = 0

print("Training GLADS-CT Polar Malignancy Model...")
for epoch in range(1, 26):
    model.train()
    tr_loss = 0.0
    for b in train_loader:
        x, y = b["image"].to(device), b["label"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = F.cross_entropy(out["logits"], y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item() * len(y)
    tr_loss /= len(train_df)
    scheduler.step()
    
    # Val
    model.eval()
    val_preds, val_y = [], []
    with torch.no_grad():
        for b in val_loader:
            x, y = b["image"].to(device), b["label"].to(device)
            with torch.cuda.amp.autocast():
                out = model(x)
            val_preds.extend(out["logits"].argmax(1).cpu().numpy())
            val_y.extend(y.cpu().numpy())
    val_acc = accuracy_score(val_y, val_preds)
    val_f1 = f1_score(val_y, val_preds, average="macro")
    
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints/GLADS_polar_best.pt")
        
    if epoch % 5 == 0 or is_best:
        print(f"Ep {epoch:02d} | TrLoss: {tr_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% {'[*BEST*]' if is_best else ''}")

# Evaluate best model on TEST with TTA
model.load_state_dict(torch.load("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints/GLADS_polar_best.pt"))
model.eval()

test_logits = []
test_y = []
with torch.no_grad():
    for b in test_loader:
        x, y = b["image"].to(device), b["label"].to(device)
        x_h = torch.flip(x, dims=[-1])
        x_v = torch.flip(x, dims=[-2])
        x_hv = torch.flip(x, dims=[-1, -2])
        with torch.cuda.amp.autocast():
            l0 = model(x)["logits"]
            l1 = model(x_h)["logits"]
            l2 = model(x_v)["logits"]
            l3 = model(x_hv)["logits"]
            l = (l0 + l1 + l2 + l3) / 4.0
        test_logits.append(l.cpu())
        test_y.extend(y.cpu().numpy())

test_logits = torch.cat(test_logits, dim=0)
test_probs = F.softmax(test_logits, dim=1).numpy()[:, 1]
test_preds = test_logits.argmax(1).numpy()
test_y = np.array(test_y)

acc = accuracy_score(test_y, test_preds)
f1 = f1_score(test_y, test_preds, average="macro")
f1s = f1_score(test_y, test_preds, average=None)
auc = roc_auc_score(test_y, test_probs)
cm = confusion_matrix(test_y, test_preds)

print("\n=======================================================")
print("=== FINAL POLAR TEST EVALUATION (BENIGN VS MALIGNANT) ===")
print("=======================================================")
print(f"Test Accuracy: {acc*100:.2f}% ({np.sum(test_preds == test_y)}/{len(test_y)})")
print(f"Test Macro-F1: {f1*100:.2f}%")
print(f"Benign F1:     {f1s[0]*100:.2f}%")
print(f"Malignant F1:  {f1s[1]*100:.2f}%")
print(f"ROC AUC:       {auc:.4f}")
print("Confusion Matrix (0: Benign, 1: Malignant):")
print(cm)
