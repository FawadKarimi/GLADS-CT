
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")
from dataset import LIDCSliceDataset

device = torch.device("cuda:1")
df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")

train_df = df[df["split_slice"] == "train"]
test_df = df[df["split_slice"] == "test"]

train_loader = DataLoader(LIDCSliceDataset(train_df, is_train=True), batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(LIDCSliceDataset(test_df, is_train=False), batch_size=32, shuffle=False, num_workers=2)

# Load ConvNeXt-Tiny
weights_path = "/home/chaidir/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth"
model = models.convnext_tiny(weights=None)
state_dict = torch.load(weights_path)
model.load_state_dict(state_dict)

# Modify first layer for 1-channel CT
w = model.features[0][0].weight.data
model.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
model.features[0][0].weight.data = w.mean(dim=1, keepdim=True)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 3)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.cuda.amp.GradScaler()

print("Training High-Capacity ConvNeXt Backbone...")
for epoch in range(1, 21):
    model.train()
    tr_loss = 0.0
    for b in train_loader:
        x, y = b["image"].to(device), b["label"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = F.cross_entropy(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item() * len(y)
    tr_loss /= len(train_df)
    scheduler.step()
    
    if epoch % 2 == 0 or epoch == 20:
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b in test_loader:
                x, y = b["image"].to(device), b["label"].to(device)
                with torch.cuda.amp.autocast():
                    out = model(x)
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        acc = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        print(f"Ep {epoch:02d} | TrLoss: {tr_loss:.4f} | Test Acc: {acc*100:.2f}% | Test Macro-F1: {macro_f1*100:.2f}%")
