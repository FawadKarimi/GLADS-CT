
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

# Build GLADS-CT with Pretrained Backbone + Lateral Gating + Transformer
class PretrainedGLADS(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        weights_path = "/home/chaidir/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
        base = models.resnet18(weights=None)
        base.load_state_dict(torch.load(weights_path))
        
        # Modify first conv for 1-channel CT
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
        
        # Layer 3 outputs [B, 256, 4, 4] for 64x64 input
        self.proj_cnn = nn.Conv2d(256, embed_dim, kernel_size=1)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)
        
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
            nn.ReLU(inplace=True)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        # Backbone
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self.layer2(h)
        f_cnn = self.layer3(h) # [B, 256, 4, 4]
        
        cnn_feat = self.proj_cnn(f_cnn) # [B, 128, 4, 4]
        B, C, H, W = cnn_feat.shape
        
        # Transformer
        tokens = cnn_feat.flatten(2).transpose(1, 2) # [B, 16, 128]
        tokens = tokens + self.pos_embed
        trans_tokens = self.transformer(tokens)
        f_trans = trans_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        # Dynamic spatial attention
        attn_map = self.spatial_attn(f_trans)
        f_trans_attn = f_trans * attn_map
        
        # GLADS Lateral Gate
        cat_feat = torch.cat([cnn_feat, f_trans_attn], dim=1)
        g = self.gate(cat_feat)
        fused = self.refine(g * cnn_feat + (1.0 - g) * f_trans_attn)
        
        logits = self.classifier(fused)
        return logits

model = PretrainedGLADS().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.cuda.amp.GradScaler()

print("Training Pretrained GLADS-CT on Slice Split (20 Epochs)...")
for epoch in range(1, 21):
    model.train()
    tr_loss = 0.0
    for b in train_loader:
        x = b["image"].to(device)
        y = b["label"].to(device)
        rt = b["reader_target"].to(device)
        disag = b["disagreement"].to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            # Disagreement-Aware Supervision
            u = disag.unsqueeze(1)
            target = (1.0 - u) * F.one_hot(y, 3).float() + u * rt
            log_p = F.log_softmax(logits, dim=1)
            loss = -(target * log_p).sum(dim=1).mean()
            
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
                x = b["image"].to(device)
                y = b["label"].to(device)
                with torch.cuda.amp.autocast():
                    logits = model(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        acc = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        print(f"Ep {epoch:02d} | TrLoss: {tr_loss:.4f} | Test Acc: {acc*100:.2f}% | Test Macro-F1: {macro_f1*100:.2f}%")

