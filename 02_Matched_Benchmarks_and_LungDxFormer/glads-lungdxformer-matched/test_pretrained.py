
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")
train_df = df[df["split_slice"] == "train"]
test_df = df[df["split_slice"] == "test"]

train_loader = DataLoader(LIDCSliceDataset(train_df, is_train=True), batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(LIDCSliceDataset(test_df, is_train=False), batch_size=32, shuffle=False, num_workers=2)

weights_path = "/home/chaidir/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
model = models.resnet18(weights=None)
state_dict = torch.load(weights_path)
model.load_state_dict(state_dict)

w = model.conv1.weight.data
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.conv1.weight.data = w.mean(dim=1, keepdim=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
scaler = torch.cuda.amp.GradScaler()

print("Testing Pretrained ResNet-18 Transfer Learning (10 Epochs Quick Test)...")
for epoch in range(1, 11):
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
    
    # Test eval
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
    print(f"Epoch {epoch:02d} | TrLoss: {tr_loss:.4f} | Test Acc: {acc*100:.2f}% | Test Macro-F1: {macro_f1*100:.2f}%")
