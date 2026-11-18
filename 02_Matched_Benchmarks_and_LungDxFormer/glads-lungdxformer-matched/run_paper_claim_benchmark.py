
import os
import sys
import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from scipy.stats import binomtest

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("=== PROTOCOL-MATCHED 4,095 BENCHMARK RUNNER ===")
print("Executing on device:", device)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Load Benchmark Metadata
data_dir = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/data/benchmark_4095"
df = pd.read_csv(f"{data_dir}/metadata.csv")

LABEL_MAP = {"benign": 0, "indeterminate": 1, "malignant": 2}
df["label_id"] = df["label"].map(LABEL_MAP)

# Add soft targets and disagreement
def get_soft_target(label_id):
    if label_id == 0:
        return [0.92, 0.08, 0.0]
    elif label_id == 1:
        return [0.20, 0.60, 0.20]
    else:
        return [0.0, 0.08, 0.92]

def get_disagreement(label_id):
    return 0.35 if label_id == 1 else 0.05

df["reader_target"] = df["label_id"].apply(get_soft_target)
df["disagreement"] = df["label_id"].apply(get_disagreement)

train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "val"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Cohort Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Total: {len(df)})")

class BenchmarkDataset(Dataset):
    def __init__(self, df, data_dir, is_train=False):
        self.df = df
        self.data_dir = data_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_p = os.path.join(self.data_dir, row["image_path"])
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        if self.is_train:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            if np.random.rand() > 0.5:
                img = np.flipud(img)
            angle = np.random.uniform(-10, 10)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
        tensor_x = torch.from_numpy(img).unsqueeze(0).float()
        y = int(row["label_id"])
        rt = torch.tensor(row["reader_target"], dtype=torch.float32)
        disag = float(row["disagreement"])
        
        return {
            "image": tensor_x,
            "label": torch.tensor(y, dtype=torch.long),
            "reader_target": rt,
            "disagreement": torch.tensor(disag, dtype=torch.float32)
        }

train_loader = DataLoader(BenchmarkDataset(train_df, data_dir, is_train=True), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(BenchmarkDataset(val_df, data_dir, is_train=False), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(BenchmarkDataset(test_df, data_dir, is_train=False), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

WEIGHTS_PATH = "/home/chaidir/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"

# 1. LungDxFormer Architecture
class LungDxFormer_Model(nn.Module):
    def __init__(self, embed_dim=128, num_classes=3):
        super().__init__()
        base = models.resnet18(weights=None)
        base.load_state_dict(torch.load(WEIGHTS_PATH))
        
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
        
        self.proj_cnn = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, dropout=0.2, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.spatial_attn = nn.Sequential(nn.Conv2d(embed_dim, 1, kernel_size=1), nn.Sigmoid())
        
        # Concat Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        h = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        h = self.layer2(self.layer1(h))
        cnn_feat = self.proj_cnn(self.layer3(h))
        B, C, H, W = cnn_feat.shape
        
        tokens = cnn_feat.flatten(2).transpose(1, 2) + self.pos_embed
        trans_tokens = self.transformer(tokens)
        f_trans = trans_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        attn = self.spatial_attn(f_trans)
        fused = self.fusion(torch.cat([cnn_feat, f_trans * attn], dim=1))
        return self.classifier(fused)

# 2. GLADS-CT Architecture
class GLADS_CT_Model(nn.Module):
    def __init__(self, embed_dim=128, num_classes=3):
        super().__init__()
        base = models.resnet18(weights=None)
        base.load_state_dict(torch.load(WEIGHTS_PATH))
        
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
        
        self.proj_cnn = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.spatial_attn = nn.Sequential(nn.Conv2d(embed_dim, 1, kernel_size=1), nn.Sigmoid())
        
        # GLADS Coarse-to-Fine Lateral Gate
        self.gate = nn.Sequential(nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1), nn.Sigmoid())
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        h = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        h = self.layer2(self.layer1(h))
        cnn_feat = self.proj_cnn(self.layer3(h))
        B, C, H, W = cnn_feat.shape
        
        tokens = cnn_feat.flatten(2).transpose(1, 2) + self.pos_embed
        trans_tokens = self.transformer(tokens)
        f_trans = trans_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        attn = self.spatial_attn(f_trans)
        f_trans_attn = f_trans * attn
        
        # Lateral Gated Fusion
        cat_feat = torch.cat([cnn_feat, f_trans_attn], dim=1)
        g = self.gate(cat_feat)
        fused = self.refine(g * cnn_feat + (1.0 - g) * f_trans_attn)
        return self.classifier(fused)

def train_and_eval(model_name, model, is_disag=False, epochs=25):
    print(f"\n--- Training {model_name} (Epochs={epochs}) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_f1 = 0.0
    best_path = f"/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints/{model_name}_best.pt"
    
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for b in train_loader:
            x = b["image"].to(device)
            y = b["label"].to(device)
            rt = b["reader_target"].to(device)
            disag = b["disagreement"].to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                if is_disag:
                    u = disag.unsqueeze(1)
                    target = (1.0 - u) * F.one_hot(y, 3).float() + u * rt
                    loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                else:
                    loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tr_loss += loss.item() * len(y)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += len(y)
            
        tr_acc = tr_correct / tr_total
        scheduler.step()
        
        # Eval val
        model.eval()
        val_preds, val_y = [], []
        with torch.no_grad():
            for b in val_loader:
                x, y = b["image"].to(device), b["label"].to(device)
                with torch.cuda.amp.autocast():
                    l = model(x)
                val_preds.extend(l.argmax(1).cpu().numpy())
                val_y.extend(y.cpu().numpy())
        val_acc = accuracy_score(val_y, val_preds)
        val_f1 = f1_score(val_y, val_preds, average="macro")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            
        if ep % 5 == 0 or ep == epochs:
            print(f"Ep {ep:02d} | TrAcc: {tr_acc*100:.2f}% | ValAcc: {val_acc*100:.2f}% | ValMacroF1: {val_f1*100:.2f}%")
            
    # Final Test Eval with TTA
    model.load_state_dict(torch.load(best_path))
    model.eval()
    test_logits, test_y = [], []
    with torch.no_grad():
        for b in test_loader:
            x, y = b["image"].to(device), b["label"].to(device)
            x_h = torch.flip(x, dims=[-1])
            x_v = torch.flip(x, dims=[-2])
            with torch.cuda.amp.autocast():
                l = (model(x) + model(x_h) + model(x_v)) / 3.0
            test_logits.append(l.cpu())
            test_y.extend(y.cpu().numpy())
            
    test_logits = torch.cat(test_logits, dim=0)
    test_probs = F.softmax(test_logits, dim=1).numpy()
    preds = test_logits.argmax(1).numpy()
    targets = np.array(test_y)
    
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    f1_classes = f1_score(targets, preds, average=None)
    prec_classes = precision_score(targets, preds, average=None)
    rec_classes = recall_score(targets, preds, average=None)
    auc = roc_auc_score(np.eye(3)[targets], test_probs, average="macro")
    cm = confusion_matrix(targets, preds).tolist()
    
    print(f"\n=== TEST RESULTS: {model_name} ===")
    print(f"Test Accuracy: {acc*100:.2f}% ({np.sum(preds == targets)}/{len(targets)})")
    print(f"Test Macro-F1: {macro_f1*100:.2f}%")
    print(f"Benign F1:     {f1_classes[0]*100:.2f}% (P: {prec_classes[0]*100:.2f}%, R: {rec_classes[0]*100:.2f}%)")
    print(f"Indet F1:      {f1_classes[1]*100:.2f}% (P: {prec_classes[1]*100:.2f}%, R: {rec_classes[1]*100:.2f}%)")
    print(f"Malig F1:      {f1_classes[2]*100:.2f}% (P: {prec_classes[2]*100:.2f}%, R: {rec_classes[2]*100:.2f}%)")
    print(f"ROC AUC:       {auc:.4f}")
    print(f"Confusion Matrix:\n{np.array(cm)}")
    
    return {
        "model": model_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "f1_classes": [float(v) for v in f1_classes],
        "prec_classes": [float(v) for v in prec_classes],
        "rec_classes": [float(v) for v in rec_classes],
        "auc": float(auc),
        "confusion_matrix": cm,
        "preds": preds.tolist(),
        "targets": targets.tolist()
    }

print("Running Head-to-Head Training on the 4,095 Sample Benchmark...")
ldf_res = train_and_eval("LungDxFormer_4095", LungDxFormer_Model().to(device), is_disag=False, epochs=25)
glads_res = train_and_eval("GLADS_CT_4095", GLADS_CT_Model().to(device), is_disag=True, epochs=25)

# McNemar Test
t = np.array(glads_res["targets"])
p_l = np.array(ldf_res["preds"])
p_g = np.array(glads_res["preds"])
c_l = (p_l == t)
c_g = (p_g == t)
b = int(np.sum(~c_l & c_g))
c = int(np.sum(c_l & ~c_g))
p_val = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue) if (b + c) > 0 else 1.0

print(f"\n=== STATISTICAL SIGNIFICANCE TESTING ===")
print(f"McNemar Contingency: b (GLADS correct, LDF incorrect) = {b}")
print(f"                     c (LDF correct, GLADS incorrect) = {c}")
print(f"                     p-value = {p_val:.6f}")

out_json = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/paper_claim_benchmark_results.json"
with open(out_json, "w") as f:
    json.dump({"lungdxformer": ldf_res, "glads_ct": glads_res, "mcnemar": {"b": b, "c": c, "p_value": p_val}}, f, indent=2)
print("Saved benchmark report to:", out_json)

