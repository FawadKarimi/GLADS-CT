
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")
from dataset import LIDCSliceDataset

device = torch.device("cuda:1")
df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")

train_df = df[df["split_slice"] == "train"].reset_index(drop=True)
val_df = df[df["split_slice"] == "val"].reset_index(drop=True)
test_df = df[df["split_slice"] == "test"].reset_index(drop=True)

train_loader = DataLoader(LIDCSliceDataset(train_df, is_train=True), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(LIDCSliceDataset(val_df, is_train=False), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(LIDCSliceDataset(test_df, is_train=False), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

class GLADS_HighPerf(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        weights_path = "/home/chaidir/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth"
        base = models.convnext_tiny(weights=None)
        base.load_state_dict(torch.load(weights_path))
        
        # Modify first layer for 1-channel CT
        w = base.features[0][0].weight.data
        base.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        base.features[0][0].weight.data = w.mean(dim=1, keepdim=True)
        
        self.backbone = base.features # outputs [B, 768, 2, 2] for 64x64
        self.proj_cnn = nn.Conv2d(768, embed_dim, kernel_size=1)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        
        # Dynamic Spatial Attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # GLADS Coarse-to-Fine Lateral Gate
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        feat = self.backbone(x) # [B, 768, 2, 2]
        cnn_feat = self.proj_cnn(feat) # [B, 128, 2, 2]
        B, C, H, W = cnn_feat.shape
        
        tokens = cnn_feat.flatten(2).transpose(1, 2) + self.pos_embed # [B, 4, 128]
        trans_tokens = self.transformer(tokens)
        f_trans = trans_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        attn_map = self.spatial_attn(f_trans)
        f_trans_attn = f_trans * attn_map
        
        # Lateral gate
        cat_feat = torch.cat([cnn_feat, f_trans_attn], dim=1)
        g = self.gate(cat_feat)
        fused = self.refine(g * cnn_feat + (1.0 - g) * f_trans_attn)
        
        logits = self.classifier(fused)
        return logits

model = GLADS_HighPerf().to(device)

# Differential learning rates: smaller for pretrained backbone, higher for GLADS gate + transformer
backbone_params = list(model.backbone.parameters())
head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params, "lr": 5e-4}
], weight_decay=1e-2)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
scaler = torch.cuda.amp.GradScaler()

print("Training ConvNeXt-GLADS on Slice Split (25 Epochs)...")
for epoch in range(1, 26):
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for b in train_loader:
        x, y = b["image"].to(device), b["label"].to(device)
        rt = b["reader_target"].to(device)
        disag = b["disagreement"].to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            u = disag.unsqueeze(1)
            target = (1.0 - u) * F.one_hot(y, 3).float() + u * rt
            log_p = F.log_softmax(logits, dim=1)
            loss = -(target * log_p).sum(dim=1).mean()
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        tr_loss += loss.item() * len(y)
        tr_correct += (logits.argmax(1) == y).sum().item()
        tr_total += len(y)
        
    tr_loss /= tr_total
    tr_acc = tr_correct / tr_total
    scheduler.step()
    
    if epoch % 5 == 0 or epoch == 25:
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b in test_loader:
                x, y = b["image"].to(device), b["label"].to(device)
                with torch.cuda.amp.autocast():
                    out = model(x)
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        test_acc = accuracy_score(targets, preds)
        test_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        print(f"Ep {epoch:02d} | TrLoss: {tr_loss:.4f} | TrAcc: {tr_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1*100:.2f}%")

