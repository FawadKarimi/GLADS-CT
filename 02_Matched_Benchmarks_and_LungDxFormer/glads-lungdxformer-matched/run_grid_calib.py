
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/LungDxFormer/src")
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")

from dataset import LIDCSliceDataset
from glads_slice_model import GLADS_CT_Slice
from lungdxformer.models.lungdxformer import LungDxFormer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_DIR = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints"
df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")

val_df = df[df["split_slice"] == "val"].reset_index(drop=True)
test_df = df[df["split_slice"] == "test"].reset_index(drop=True)

val_loader = DataLoader(LIDCSliceDataset(val_df, is_train=False), batch_size=32, shuffle=False)
test_loader = DataLoader(LIDCSliceDataset(test_df, is_train=False), batch_size=32, shuffle=False)

def predict_tta(model, loader):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            x_h = torch.flip(x, dims=[-1])
            x_v = torch.flip(x, dims=[-2])
            x_hv = torch.flip(x, dims=[-1, -2])
            with torch.cuda.amp.autocast():
                l0 = model(x)["logits"]
                l1 = model(x_h)["logits"]
                l2 = model(x_v)["logits"]
                l3 = model(x_hv)["logits"]
                logits = (l0 + l1 + l2 + l3) / 4.0
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
    return torch.cat(all_logits, dim=0).numpy(), torch.cat(all_targets, dim=0).numpy()

# Load GLADS ensemble models (seeds 42, 43, 44)
val_logits_list = []
test_logits_list = []

for seed in [42, 43, 44]:
    ckpt = torch.load(f"{CKPT_DIR}/GLADS-Hard-seed{seed}_best.pt", map_location=device)
    m = GLADS_CT_Slice(image_size=64, num_classes=3).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    vl, vy = predict_tta(m, val_loader)
    tl, ty = predict_tta(m, test_loader)
    val_logits_list.append(vl)
    test_logits_list.append(tl)

val_ens = np.mean(val_logits_list, axis=0)
test_ens = np.mean(test_logits_list, axis=0)

# Uncalibrated baseline metrics
base_preds = np.argmax(test_ens, axis=1)
base_acc = accuracy_score(ty, base_preds)
base_f1 = f1_score(ty, base_preds, average="macro")
base_f1s = f1_score(ty, base_preds, average=None)

print(f"GLADS Ensemble (Raw TTA): Acc={base_acc*100:.2f}%, Macro-F1={base_f1*100:.2f}%, F1s={base_f1s*100}")

# Grid search for optimal bias vector (b0, b1, b2) on VALIDATION set
best_val_f1 = -1.0
best_bias = None

# Grid over b0, b1, b2
b_range = np.linspace(-1.0, 1.0, 21)
for b0 in b_range:
    for b1 in b_range:
        for b2 in b_range:
            bias = np.array([b0, b1, b2])
            adj_val = val_ens + bias
            preds = np.argmax(adj_val, axis=1)
            score = f1_score(vy, preds, average="macro", zero_division=0)
            if score > best_val_f1:
                best_val_f1 = score
                best_bias = bias

print(f"\nBest Val Macro-F1 found: {best_val_f1*100:.2f}% with bias: {best_bias}")

# Apply best bias to TEST set
opt_test = test_ens + best_bias
opt_preds = np.argmax(opt_test, axis=1)
opt_acc = accuracy_score(ty, opt_preds)
opt_macro_f1 = f1_score(ty, opt_preds, average="macro")
opt_f1s = f1_score(ty, opt_preds, average=None)

print(f"GLADS Ensemble (Calibrated on Val):")
print(f"Test Accuracy: {opt_acc*100:.2f}%")
print(f"Test Macro-F1: {opt_macro_f1*100:.2f}%")
print(f"Per-Class F1: Benign={opt_f1s[0]*100:.2f}%, Indeterminate={opt_f1s[1]*100:.2f}%, Malignant={opt_f1s[2]*100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(ty, opt_preds))

# Save results
out_calib = {
    "optimal_bias": best_bias.tolist(),
    "best_val_macro_f1": float(best_val_f1),
    "test_accuracy": float(opt_acc),
    "test_macro_f1": float(opt_macro_f1),
    "test_per_class_f1": opt_f1s.tolist(),
    "confusion_matrix": confusion_matrix(ty, opt_preds).tolist()
}
with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/calibrated_ensemble_results.json", "w") as f:
    json.dump(out_calib, f, indent=2)
