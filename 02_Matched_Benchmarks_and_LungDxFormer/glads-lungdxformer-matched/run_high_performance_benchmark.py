
import os
import sys
import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from scipy.stats import binomtest

# Environment and Device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("=== HIGH-PERFORMANCE BENCHMARK RUNNER ===")
print("Executing on device:", device)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Load Manifest
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")
from dataset import LIDCSliceDataset

csv_path = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv"
df = pd.read_csv(csv_path)

train_df = df[df["split_slice"] == "train"].reset_index(drop=True)
val_df = df[df["split_slice"] == "val"].reset_index(drop=True)
test_df = df[df["split_slice"] == "test"].reset_index(drop=True)

print(f"Cohort Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Total: {len(df)})")

# Class weights for training
class_counts = train_df["label"].value_counts().sort_index().values
total_samples = len(train_df)
class_weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class Frequencies in Train:", class_counts)
print("Computed Class Weights:", class_weights)

train_loader = DataLoader(LIDCSliceDataset(train_df, is_train=True), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(LIDCSliceDataset(val_df, is_train=False), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(LIDCSliceDataset(test_df, is_train=False), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

WEIGHTS_PATH = "/home/chaidir/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"

# ==========================================
# 1. LungDxFormer Baseline Architecture (5.6M)
# ==========================================
class LungDxFormer_HighPerf(nn.Module):
    def __init__(self, embed_dim=128, num_classes=3):
        super().__init__()
        base = models.resnet18(weights=None)
        base.load_state_dict(torch.load(WEIGHTS_PATH))
        
        # 1-channel CT adaptation
        w = base.conv1.weight.data
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = w.mean(dim=1, keepdim=True)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3 # [B, 256, 4, 4] for 64x64
        
        self.proj_cnn = nn.Conv2d(256, embed_dim, kernel_size=1)
        
        # Learnable Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)
        
        # Transformer Encoder (2 layers, 4 heads, mlp_ratio=4.0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, dropout=0.2, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Dynamic Spatial Attention (DSA)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Concatenation Fusion Layer (Eq. 7 & Eq. 13 of Rao et al.)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classification Head (Dropout 0.5)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self.layer2(h)
        f_cnn_raw = self.layer3(h) # [B, 256, 4, 4]
        
        cnn_feat = self.proj_cnn(f_cnn_raw) # [B, 128, 4, 4]
        B, C, H, W = cnn_feat.shape
        
        tokens = cnn_feat.flatten(2).transpose(1, 2) + self.pos_embed # [B, 16, 128]
        trans_tokens = self.transformer(tokens)
        f_trans = trans_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        attn_map = self.spatial_attn(f_trans)
        f_trans_attn = f_trans * attn_map
        
        # Concatenation fusion
        cat_feat = torch.cat([cnn_feat, f_trans_attn], dim=1)
        fused = self.fusion_conv(cat_feat)
        
        logits = self.classifier(fused)
        return logits


# ==========================================
# 2. GLADS-CT Proposed Architecture (5.8M)
# ==========================================
class GLADS_CT_HighPerf(nn.Module):
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
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
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
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Diagnostic Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self.layer2(h)
        f_cnn_raw = self.layer3(h)
        
        cnn_feat = self.proj_cnn(f_cnn_raw)
        B, C, H, W = cnn_feat.shape
        
        tokens = cnn_feat.flatten(2).transpose(1, 2) + self.pos_embed
        trans_tokens = self.transformer(tokens)
        f_trans = trans_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        attn_map = self.spatial_attn(f_trans)
        f_trans_attn = f_trans * attn_map
        
        # GLADS-CT Gated Fusion
        cat_feat = torch.cat([cnn_feat, f_trans_attn], dim=1)
        g = self.gate(cat_feat)
        fused = self.refine(g * cnn_feat + (1.0 - g) * f_trans_attn)
        
        logits = self.classifier(fused)
        return logits


# ==========================================
# Training Function
# ==========================================
def train_model(model_name, model, is_disag=False, epochs=40, lr=3e-4):
    print(f"\n=======================================================")
    print(f"Starting Training: {model_name} (Epochs={epochs}, LR={lr})")
    print(f"=======================================================")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_macro_f1 = 0.0
    best_ckpt_path = f"/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints/{model_name}_best.pt"
    history = []
    
    start_t = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            rt = batch["reader_target"].to(device, non_blocking=True)
            disag = batch["disagreement"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                if is_disag:
                    # GLADS-CT Disagreement-Aware Supervision
                    u = disag.unsqueeze(1)
                    target = (1.0 - u) * F.one_hot(y, 3).float() + u * rt
                    mass = target * class_weights_t.unsqueeze(0)
                    log_p = F.log_softmax(logits, dim=1)
                    loss = -(mass * log_p).sum() / mass.sum().clamp_min(1e-8)
                else:
                    loss = F.cross_entropy(logits, y, weight=class_weights_t)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tr_loss += loss.item() * len(y)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += len(y)
            
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, weight=class_weights_t)
                val_loss += loss.item() * len(y)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_targets.extend(y.cpu().numpy())
                
        val_loss /= len(val_df)
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        
        f1_classes = f1_score(val_targets, val_preds, average=None, zero_division=0)
        f1_b, f1_i, f1_m = f1_classes[0], f1_classes[1], f1_classes[2]
        
        marker = ""
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_ckpt_path)
            marker = "[*BEST*]"
            
        history.append({
            "epoch": ep, "tr_loss": tr_loss, "tr_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_macro_f1,
            "f1_b": f1_b, "f1_i": f1_i, "f1_m": f1_m
        })
        
        if ep % 5 == 0 or marker != "" or ep == epochs:
            print(f"Ep {ep:02d} | TrLoss: {tr_loss:.4f} TrAcc: {tr_acc*100:.2f}% | ValLoss: {val_loss:.4f} ValAcc: {val_acc*100:.2f}% ValF1: {val_macro_f1*100:.2f}% (B/I/M: {f1_b*100:.1f}/{f1_i*100:.1f}/{f1_m*100:.1f}) {marker}")
            
    total_time = time.time() - start_t
    print(f"Finished {model_name} in {total_time:.1f}s. Best Val Macro-F1: {best_val_macro_f1*100:.2f}%")
    return best_ckpt_path, history


# ==========================================
# Evaluation Function with TTA
# ==========================================
def evaluate_model(model, ckpt_path, loader, tta=True):
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            
            if not tta:
                with torch.cuda.amp.autocast():
                    l = model(x)
            else:
                x_h = torch.flip(x, dims=[-1])
                x_v = torch.flip(x, dims=[-2])
                x_hv = torch.flip(x, dims=[-1, -2])
                with torch.cuda.amp.autocast():
                    l0 = model(x)
                    l1 = model(x_h)
                    l2 = model(x_v)
                    l3 = model(x_hv)
                l = (l0 + l1 + l2 + l3) / 4.0
                
            all_logits.append(l.cpu())
            all_targets.extend(y.cpu().numpy())
            
    logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(1).numpy()
    targets = np.array(all_targets)
    
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    prec_macro = precision_score(targets, preds, average="macro", zero_division=0)
    rec_macro = recall_score(targets, preds, average="macro", zero_division=0)
    
    f1_per_class = f1_score(targets, preds, average=None, zero_division=0)
    prec_per_class = precision_score(targets, preds, average=None, zero_division=0)
    rec_per_class = recall_score(targets, preds, average=None, zero_division=0)
    
    cm = confusion_matrix(targets, preds).tolist()
    
    # One-vs-rest AUC
    try:
        y_onehot = np.eye(3)[targets]
        auc_macro = roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr")
        auc_per_class = roc_auc_score(y_onehot, probs, average=None, multi_class="ovr").tolist()
    except Exception as e:
        auc_macro = 0.0
        auc_per_class = [0.0, 0.0, 0.0]
        
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "auc_macro_ovr": float(auc_macro),
        "f1_per_class": [float(x) for x in f1_per_class],
        "prec_per_class": [float(x) for x in prec_per_class],
        "rec_per_class": [float(x) for x in rec_per_class],
        "auc_per_class": auc_per_class,
        "confusion_matrix": cm,
        "preds": preds.tolist(),
        "targets": targets.tolist(),
        "probs": probs.tolist()
    }


# ==========================================
# Run Training for Both Models
# ==========================================
print("\n>>> STEP 1: TRAINING LUNGDXFORMER (REPRODUCED BASELINE)...")
ldf_model = LungDxFormer_HighPerf().to(device)
ldf_ckpt, ldf_hist = train_model("LungDxFormer_Benchmark", ldf_model, is_disag=False, epochs=40, lr=3e-4)

print("\n>>> STEP 2: TRAINING GLADS-CT (PROPOSED GATED + DISAGREEMENT)...")
glads_model = GLADS_CT_HighPerf().to(device)
glads_ckpt, glads_hist = train_model("GLADS_CT_Benchmark", glads_model, is_disag=True, epochs=40, lr=3e-4)

# ==========================================
# Step 3: Sealed Test Set Evaluation
# ==========================================
print("\n=======================================================")
print("=== FINAL EVALUATION ON SEALED TEST SET (N = 641) ===")
print("=======================================================")

ldf_test_raw = evaluate_model(ldf_model, ldf_ckpt, test_loader, tta=False)
ldf_test_tta = evaluate_model(ldf_model, ldf_ckpt, test_loader, tta=True)

glads_test_raw = evaluate_model(glads_model, glads_ckpt, test_loader, tta=False)
glads_test_tta = evaluate_model(glads_model, glads_ckpt, test_loader, tta=True)

# McNemar test
targets = np.array(glads_test_tta["targets"])
p_ldf = np.array(ldf_test_tta["preds"])
p_glads = np.array(glads_test_tta["preds"])

correct_ldf = (p_ldf == targets)
correct_glads = (p_glads == targets)

b = int(np.sum(~correct_ldf & correct_glads)) # LDF wrong, GLADS right
c = int(np.sum(correct_ldf & ~correct_glads)) # LDF right, GLADS wrong
mcnemar_p = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
chi2 = float((abs(b - c) - 1)**2 / (b + c)) if (b + c) > 0 else 0.0

print(f"\n--- Comparative Results ---")
print(f"LungDxFormer (Raw): Acc = {ldf_test_raw['accuracy']*100:.2f}% | Macro-F1 = {ldf_test_raw['macro_f1']*100:.2f}% | AUC = {ldf_test_raw['auc_macro_ovr']:.4f}")
print(f"LungDxFormer (TTA): Acc = {ldf_test_tta['accuracy']*100:.2f}% | Macro-F1 = {ldf_test_tta['macro_f1']*100:.2f}% | AUC = {ldf_test_tta['auc_macro_ovr']:.4f}")
print(f"GLADS-CT     (Raw): Acc = {glads_test_raw['accuracy']*100:.2f}% | Macro-F1 = {glads_test_raw['macro_f1']*100:.2f}% | AUC = {glads_test_raw['auc_macro_ovr']:.4f}")
print(f"GLADS-CT     (TTA): Acc = {glads_test_tta['accuracy']*100:.2f}% | Macro-F1 = {glads_test_tta['macro_f1']*100:.2f}% | AUC = {glads_test_tta['auc_macro_ovr']:.4f}")
print(f"\nGLADS-CT Class F1 (TTA): Benign = {glads_test_tta['f1_per_class'][0]*100:.2f}% | Indet = {glads_test_tta['f1_per_class'][1]*100:.2f}% | Malig = {glads_test_tta['f1_per_class'][2]*100:.2f}%")
print(f"McNemar Test: b={b}, c={c}, chi2={chi2:.4f}, p-value = {mcnemar_p:.6f}")

# Save Full JSON Report
final_report = {
    "protocol": "High-Performance Benchmark Matched Protocol",
    "cohort_size": len(df),
    "test_size": len(test_df),
    "models": {
        "LungDxFormer_Raw": ldf_test_raw,
        "LungDxFormer_TTA": ldf_test_tta,
        "GLADS_CT_Raw": glads_test_raw,
        "GLADS_CT_TTA": glads_test_tta
    },
    "mcnemar": {
        "b": b, "c": c, "chi2": chi2, "p_value": mcnemar_p
    },
    "checkpoints": {
        "LungDxFormer": ldf_ckpt,
        "GLADS_CT": glads_ckpt
    }
}

report_path = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/high_performance_benchmark_report.json"
with open(report_path, "w") as f:
    json.dump(final_report, f, indent=2)

print(f"Saved final evaluation report to: {report_path}")
print("ALL BENCHMARK RUNS COMPLETED SUCCESSFULLY.")

