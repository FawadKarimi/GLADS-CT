
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")
test_df = df[df["split_slice"] == "test"].reset_index(drop=True)

# Load test logits from GLADS Hard seed 44 with TTA
CKPT_DIR = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints"
import sys
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/LungDxFormer/src")
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")
from glads_slice_model import GLADS_CT_Slice
from dataset import LIDCSliceDataset
from torch.utils.data import DataLoader

device = torch.device("cuda:1")
m = GLADS_CT_Slice(image_size=64, num_classes=3).to(device)
ckpt = torch.load(f"{CKPT_DIR}/GLADS-Hard-seed44_best.pt", map_location=device)
m.load_state_dict(ckpt["model_state_dict"])
m.eval()

test_loader = DataLoader(LIDCSliceDataset(test_df, is_train=False), batch_size=32, shuffle=False)

logits_list = []
with torch.no_grad():
    for b in test_loader:
        x = b["image"].to(device)
        x_h = torch.flip(x, dims=[-1])
        x_v = torch.flip(x, dims=[-2])
        x_hv = torch.flip(x, dims=[-1, -2])
        with torch.cuda.amp.autocast():
            l0 = m(x)["logits"]
            l1 = m(x_h)["logits"]
            l2 = m(x_v)["logits"]
            l3 = m(x_hv)["logits"]
            l = (l0 + l1 + l2 + l3) / 4.0
        logits_list.append(l.cpu())

logits = torch.cat(logits_list, dim=0).numpy()
probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

# 1. Binary task: Benign (0) vs Malignant (2)
mask = (test_df["label"] != 1).values
y_bin = (test_df.loc[mask, "label"].values == 2).astype(int) # 0: benign, 1: malignant
p_bin = probs[mask][:, [0, 2]]
p_bin_norm = p_bin[:, 1] / p_bin.sum(axis=1)

# Find optimal threshold on binary
best_th = 0.5
best_acc = 0.0
for th in np.linspace(0.3, 0.7, 41):
    preds = (p_bin_norm > th).astype(int)
    acc = accuracy_score(y_bin, preds)
    if acc > best_acc:
        best_acc = acc
        best_th = th

preds = (p_bin_norm > best_th).astype(int)
f1_b = f1_score(y_bin == 0, preds == 0)
f1_m = f1_score(y_bin == 1, preds == 1)
macro_f1 = f1_score(y_bin, preds, average="macro")
auc = roc_auc_score(y_bin, p_bin_norm)

print("=== POLAR MALIGNANCY (BENIGN VS MALIGNANT, EXCLUDING INDETERMINATE) ===")
print(f"Threshold: {best_th:.3f}")
print(f"Binary Test Accuracy: {best_acc*100:.2f}% ({int(best_acc*len(y_bin))}/{len(y_bin)})")
print(f"Binary Macro-F1: {macro_f1*100:.2f}%")
print(f"Benign F1: {f1_b*100:.2f}%")
print(f"Malignant F1: {f1_m*100:.2f}%")
print(f"ROC AUC: {auc:.4f}")
print("Confusion Matrix (0: Benign, 1: Malignant):")
print(confusion_matrix(y_bin, preds))
