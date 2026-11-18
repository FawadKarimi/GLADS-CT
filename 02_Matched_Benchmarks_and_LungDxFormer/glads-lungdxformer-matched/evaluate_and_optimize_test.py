
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from scipy.optimize import minimize
from scipy.stats import binomtest

sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/LungDxFormer/src")
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")

from dataset import LIDCSliceDataset
from lungdxformer.models.lungdxformer import LungDxFormer
from glads_slice_model import GLADS_CT_Slice

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_DIR = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints"
df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")

val_df = df[df["split_slice"] == "val"].reset_index(drop=True)
test_df = df[df["split_slice"] == "test"].reset_index(drop=True)

val_loader = DataLoader(LIDCSliceDataset(val_df, is_train=False), batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(LIDCSliceDataset(test_df, is_train=False), batch_size=32, shuffle=False, num_workers=2)

def predict_model(model, loader, tta=False):
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            if not tta:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    logits = out["logits"]
            else:
                # Test-Time Augmentation: 4 variations (original, hflip, vflip, rot180)
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

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    res = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "f1_benign": float(per_class_f1[0]),
        "f1_indeterminate": float(per_class_f1[1]),
        "f1_malignant": float(per_class_f1[2]),
        "confusion_matrix": cm
    }
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            res["auc_ovr"] = float(auc)
        except Exception:
            pass
    return res

runs = [
    ("LDF-B0-seed42", LungDxFormer, "concat"),
    ("LDF-B0-seed43", LungDxFormer, "concat"),
    ("LDF-B0-seed44", LungDxFormer, "concat"),
    ("GLADS-Hard-seed42", GLADS_CT_Slice, "gated"),
    ("GLADS-Hard-seed43", GLADS_CT_Slice, "gated"),
    ("GLADS-Hard-seed44", GLADS_CT_Slice, "gated"),
    ("GLADS-Disag-seed42", GLADS_CT_Slice, "gated"),
    ("GLADS-Disag-seed43", GLADS_CT_Slice, "gated"),
    ("GLADS-Disag-seed44", GLADS_CT_Slice, "gated"),
]

models = {}
for r_id, m_cls, f_type in runs:
    ckpt_path = os.path.join(CKPT_DIR, f"{r_id}_best.pt")
    if m_cls == LungDxFormer:
        m = m_cls(image_size=64, num_classes=3, fusion_type=f_type).to(device)
    else:
        m = m_cls(image_size=64, num_classes=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    models[r_id] = m

print("=== 1. RAW TEST EVALUATION (SINGLE RUNS) ===")
test_raw_results = {}
val_logits_dict = {}
test_logits_dict = {}

for r_id, m_cls, _ in runs:
    m = models[r_id]
    v_logits, v_y = predict_model(m, val_loader, tta=False)
    t_logits, t_y = predict_model(m, test_loader, tta=False)
    
    val_logits_dict[r_id] = v_logits
    test_logits_dict[r_id] = t_logits
    
    t_preds = np.argmax(t_logits, axis=1)
    t_prob = torch.softmax(torch.from_numpy(t_logits), dim=1).numpy()
    metrics = compute_metrics(t_y, t_preds, t_prob)
    test_raw_results[r_id] = metrics
    print(f"{r_id:20s} | Test Acc: {metrics['accuracy']*100:.2f}% | Test Macro-F1: {metrics['macro_f1']*100:.2f}% | F1(B/I/M): {metrics['f1_benign']*100:.1f}/{metrics['f1_indeterminate']*100:.1f}/{metrics['f1_malignant']*100:.1f}")

print("\n=== 2. TEST-TIME AUGMENTATION (TTA) EVALUATION ===")
test_tta_results = {}
val_tta_logits_dict = {}
test_tta_logits_dict = {}

for r_id, m_cls, _ in runs:
    m = models[r_id]
    v_logits_tta, _ = predict_model(m, val_loader, tta=True)
    t_logits_tta, _ = predict_model(m, test_loader, tta=True)
    
    val_tta_logits_dict[r_id] = v_logits_tta
    test_tta_logits_dict[r_id] = t_logits_tta
    
    t_preds_tta = np.argmax(t_logits_tta, axis=1)
    t_prob_tta = torch.softmax(torch.from_numpy(t_logits_tta), dim=1).numpy()
    metrics_tta = compute_metrics(t_y, t_preds_tta, t_prob_tta)
    test_tta_results[r_id] = metrics_tta
    print(f"{r_id:20s} + TTA | Test Acc: {metrics_tta['accuracy']*100:.2f}% | Test Macro-F1: {metrics_tta['macro_f1']*100:.2f}% | F1(B/I/M): {metrics_tta['f1_benign']*100:.1f}/{metrics_tta['f1_indeterminate']*100:.1f}/{metrics_tta['f1_malignant']*100:.1f}")

print("\n=== 3. ENSEMBLE EVALUATION ===")
# LungDxFormer 3-seed ensemble
ldf_val_ens = (val_tta_logits_dict["LDF-B0-seed42"] + val_tta_logits_dict["LDF-B0-seed43"] + val_tta_logits_dict["LDF-B0-seed44"]) / 3.0
ldf_test_ens = (test_tta_logits_dict["LDF-B0-seed42"] + test_tta_logits_dict["LDF-B0-seed43"] + test_tta_logits_dict["LDF-B0-seed44"]) / 3.0
ldf_ens_preds = np.argmax(ldf_test_ens, axis=1)
ldf_ens_probs = torch.softmax(torch.from_numpy(ldf_test_ens), dim=1).numpy()
ldf_ens_metrics = compute_metrics(t_y, ldf_ens_preds, ldf_ens_probs)
print(f"LDF-3Seed-Ensemble + TTA   | Test Acc: {ldf_ens_metrics['accuracy']*100:.2f}% | Test Macro-F1: {ldf_ens_metrics['macro_f1']*100:.2f}% | F1(B/I/M): {ldf_ens_metrics['f1_benign']*100:.1f}/{ldf_ens_metrics['f1_indeterminate']*100:.1f}/{ldf_ens_metrics['f1_malignant']*100:.1f}")

# GLADS-CT Gated Hard 3-seed ensemble
glads_hard_test_ens = (test_tta_logits_dict["GLADS-Hard-seed42"] + test_tta_logits_dict["GLADS-Hard-seed43"] + test_tta_logits_dict["GLADS-Hard-seed44"]) / 3.0
glads_hard_preds = np.argmax(glads_hard_test_ens, axis=1)
glads_hard_probs = torch.softmax(torch.from_numpy(glads_hard_test_ens), dim=1).numpy()
glads_hard_metrics = compute_metrics(t_y, glads_hard_preds, glads_hard_probs)
print(f"GLADS-Hard-Ensemble + TTA  | Test Acc: {glads_hard_metrics['accuracy']*100:.2f}% | Test Macro-F1: {glads_hard_metrics['macro_f1']*100:.2f}% | F1(B/I/M): {glads_hard_metrics['f1_benign']*100:.1f}/{glads_hard_metrics['f1_indeterminate']*100:.1f}/{glads_hard_metrics['f1_malignant']*100:.1f}")

# GLADS-CT Full Disagreement 3-seed ensemble
glads_disag_val_ens = (val_tta_logits_dict["GLADS-Disag-seed42"] + val_tta_logits_dict["GLADS-Disag-seed43"] + val_tta_logits_dict["GLADS-Disag-seed44"]) / 3.0
glads_disag_test_ens = (test_tta_logits_dict["GLADS-Disag-seed42"] + test_tta_logits_dict["GLADS-Disag-seed43"] + test_tta_logits_dict["GLADS-Disag-seed44"]) / 3.0
glads_disag_preds = np.argmax(glads_disag_test_ens, axis=1)
glads_disag_probs = torch.softmax(torch.from_numpy(glads_disag_test_ens), dim=1).numpy()
glads_disag_metrics = compute_metrics(t_y, glads_disag_preds, glads_disag_probs)
print(f"GLADS-Disag-Ensemble + TTA | Test Acc: {glads_disag_metrics['accuracy']*100:.2f}% | Test Macro-F1: {glads_disag_metrics['macro_f1']*100:.2f}% | F1(B/I/M): {glads_disag_metrics['f1_benign']*100:.1f}/{glads_disag_metrics['f1_indeterminate']*100:.1f}/{glads_disag_metrics['f1_malignant']*100:.1f}")

# 4. Threshold Calibration on Validation set
print("\n=== 4. THRESHOLD CALIBRATION (Fitted on VAL, Tested on TEST) ===")
# Learn class weight biases [w0, w1, w2] on validation ensemble to maximize macro F1
def loss_func(weights):
    w = np.array(weights)
    scaled_val = glads_disag_val_ens + w
    p = np.argmax(scaled_val, axis=1)
    return -f1_score(v_y, p, average="macro", zero_division=0)

opt = minimize(loss_func, [0.0, 0.0, 0.0], method="Nelder-Mead")
best_biases = opt.x
print(f"Optimal validation biases: {best_biases.tolist()}")

# Apply optimal validation biases to test set
calib_test_logits = glads_disag_test_ens + best_biases
calib_test_preds = np.argmax(calib_test_logits, axis=1)
calib_test_probs = torch.softmax(torch.from_numpy(calib_test_logits), dim=1).numpy()
calib_metrics = compute_metrics(t_y, calib_test_preds, calib_test_probs)
print(f"GLADS-CT Full (Calibrated) | Test Acc: {calib_metrics['accuracy']*100:.2f}% | Test Macro-F1: {calib_metrics['macro_f1']*100:.2f}% | F1(B/I/M): {calib_metrics['f1_benign']*100:.1f}/{calib_metrics['f1_indeterminate']*100:.1f}/{calib_metrics['f1_malignant']*100:.1f}")

# 5. Bootstrap 95% Confidence Intervals
print("\n=== 5. BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 resamples) ===")
n_boot = 1000
boot_accs = []
boot_macro_f1s = []
n_test = len(t_y)
np.random.seed(42)

for _ in range(n_boot):
    idx = np.random.choice(n_test, size=n_test, replace=True)
    b_true = t_y[idx]
    b_pred = calib_test_preds[idx]
    boot_accs.append(accuracy_score(b_true, b_pred))
    boot_macro_f1s.append(f1_score(b_true, b_pred, average="macro", zero_division=0))

ci_acc = (np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5))
ci_f1 = (np.percentile(boot_macro_f1s, 2.5), np.percentile(boot_macro_f1s, 97.5))

print(f"GLADS-CT Test Accuracy 95% CI: [{ci_acc[0]*100:.2f}%, {ci_acc[1]*100:.2f}%] (Mean: {np.mean(boot_accs)*100:.2f}%)")
print(f"GLADS-CT Test Macro-F1  95% CI: [{ci_f1[0]*100:.2f}%, {ci_f1[1]*100:.2f}%] (Mean: {np.mean(boot_macro_f1s)*100:.2f}%)")

# 6. McNemar Paired Statistical Test
table = confusion_matrix(ldf_ens_preds == t_y, calib_test_preds == t_y)

b = int(table[0, 1])
c = int(table[1, 0])
res_mcn = binomtest(b, n=b+c, p=0.5) if (b + c) > 0 else None
mcn_stat = float((abs(b - c) - 1)**2 / (b + c)) if (b + c) > 0 else 0.0
mcn_pval = float(res_mcn.pvalue) if res_mcn else 1.0
print(f"\nMcNemar test GLADS-CT vs LungDxFormer: b={b}, c={c}, p-value={mcn_pval:.4e}")

# Save full evaluation deliverable
final_deliverable = {
    "test_raw_results": test_raw_results,
    "test_tta_results": test_tta_results,
    "ensemble_results": {
        "lungdxformer_ensemble": ldf_ens_metrics,
        "glads_hard_ensemble": glads_hard_metrics,
        "glads_disag_ensemble": glads_disag_metrics,
        "glads_calibrated_ensemble": calib_metrics
    },
    "bootstrap_ci": {
        "accuracy_95ci": [float(ci_acc[0]), float(ci_acc[1])],
        "macro_f1_95ci": [float(ci_f1[0]), float(ci_f1[1])],
    },
    "mcnemar_test": {
        "statistic": mcn_stat,
        "pvalue": mcn_pval,
        "b_discordant": b,
        "c_discordant": c
    }
}

with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/test_evaluation_report.json", "w") as f:
    json.dump(final_deliverable, f, indent=2)

print("\nSaved evaluation deliverable to /experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/test_evaluation_report.json")
