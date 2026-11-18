
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

device = torch.device("cuda:1")

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")
polar_df = df[df["label"] != 1].copy()
polar_df["label"] = (polar_df["label"] == 2).astype(int)

train_df = polar_df[polar_df["split_slice"] == "train"].reset_index(drop=True)
val_df = polar_df[polar_df["split_slice"] == "val"].reset_index(drop=True)
test_df = polar_df[polar_df["split_slice"] == "test"].reset_index(drop=True)

sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")
from dataset import LIDCSliceDataset

train_loader = DataLoader(LIDCSliceDataset(train_df, is_train=True), batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(LIDCSliceDataset(val_df, is_train=False), batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(LIDCSliceDataset(test_df, is_train=False), batch_size=32, shuffle=False, num_workers=2)

# Build Pretrained ResNet18 + GLADS Gated Transformer Head
class GLADS_Pretrained_Polar(nn.Module):
    def __init__(self):
        super().__init__()
        weights_path = "/home/chaidir/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
        base = models.resnet18(weights=None)
        base.load_state_dict(torch.load(weights_path))
        
        # Adapt first conv for 1-channel CT
        w = base.conv1.weight.data
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = w.mean(dim=1, keepdim=True)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        # Transformer token projection from layer4 (512 channels -> 128)
        self.proj = nn.Conv2d(512, 128, 1)
        self.pe = nn.Parameter(torch.randn(1, 4, 128) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3) # [B, 512, 2, 2]
        
        cnn_feat = self.proj(c4) # [B, 128, 2, 2]
        b, c, h, w = cnn_feat.shape
        tokens = cnn_feat.flatten(2).transpose(1, 2) + self.pe # [B, 4, 128]
        trans_tokens = self.transformer(tokens)
        trans_feat = trans_tokens.transpose(1, 2).reshape(b, c, h, w)
        
        # Gated fusion
        g = self.gate(torch.cat([cnn_feat, trans_feat], dim=1))
        fused = self.refine(g * cnn_feat + (1 - g) * trans_feat)
        logits = self.classifier(fused)
        return logits

model = GLADS_Pretrained_Polar().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.cuda.amp.GradScaler()

best_val_acc = 0.0
for epoch in range(1, 21):
    model.train()
    tr_loss = 0.0
    for b in train_loader:
        x, y = b["image"].to(device), b["label"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item() * len(y)
    tr_loss /= len(train_df)
    scheduler.step()
    
    model.eval()
    val_preds, val_y = [], []
    with torch.no_grad():
        for b in val_loader:
            x, y = b["image"].to(device), b["label"].to(device)
            with torch.cuda.amp.autocast():
                out = model(x)
            val_preds.extend(out.argmax(1).cpu().numpy())
            val_y.extend(y.cpu().numpy())
    val_acc = accuracy_score(val_y, val_preds)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints/GLADS_pretrained_polar_best.pt")
    if epoch % 5 == 0 or val_acc > best_val_acc:
        print(f"Ep {epoch:02d} | TrLoss: {tr_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

# Eval on Test with TTA
model.load_state_dict(torch.load("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints/GLADS_pretrained_polar_best.pt"))
model.eval()
test_logits, test_y = [], []
with torch.no_grad():
    for b in test_loader:
        x, y = b["image"].to(device), b["label"].to(device)
        x_h = torch.flip(x, dims=[-1])
        x_v = torch.flip(x, dims=[-2])
        x_hv = torch.flip(x, dims=[-1, -2])
        with torch.cuda.amp.autocast():
            l = (model(x) + model(x_h) + model(x_v) + model(x_hv)) / 4.0
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

print("\n=== TEST RESULTS: PRETRAINED GLADS-CT (POLAR) ===")
print(f"Test Accuracy: {acc*100:.2f}% ({np.sum(test_preds == test_y)}/{len(test_y)})")
print(f"Test Macro-F1: {f1*100:.2f}%")
print(f"Benign F1:     {f1s[0]*100:.2f}%")
print(f"Malignant F1:  {f1s[1]*100:.2f}%")
print(f"ROC AUC:       {auc:.4f}")
