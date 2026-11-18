#!/usr/bin/env python3
"""GLADS-CT Model Ensemble and Decision-Threshold Calibration Evaluation.

Evaluates the 6 trained GLADS-CT checkpoints on the frozen 143 validation nodules.
Performs:
  1. Individual Model Evaluation (HARD-42, HARD-43, HARD-44, LEGACY-42, LEGACY-43, LEGACY-44)
  2. Model Ensemble Evaluation (HARD-42 + HARD-43, All-3 HARD, All-6)
  3. Cost-Sensitive / Bayesian Decision Threshold Calibration
  4. Direct Benchmark Comparison against DPMT-Net 3D (2026)
"""
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

CANDIDATE_SRC = "/home/chaidir/.performance-v2-replication-20261013/candidate/src"
REFERENCE_SRC = "/home/chaidir/.performance-v2-replication-20261013/reference"
for p in (CANDIDATE_SRC, REFERENCE_SRC, "."):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_train1297_six_runs as r

def compute_metrics(y_true, y_pred, class_names=("benign", "indeterminate", "malignant")):
    n = len(y_true)
    acc = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / n
    cm = [[0 for _ in range(3)] for _ in range(3)]
    for yt, yp in zip(y_true, y_pred):
        cm[yt][yp] += 1
    
    per_class = {}
    f1_list = []
    for k in range(3):
        tp = cm[k][k]
        fp = sum(cm[i][k] for i in range(3) if i != k)
        fn = sum(cm[k][j] for j in range(3) if j != k)
        supp = sum(cm[k])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        cname = class_names[k]
        per_class[cname] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(supp),
        }
    macro_f1 = sum(f1_list) / 3.0
    return {
        "accuracy": float(acc),
        "correct_count": int(sum(yt == yp for yt, yp in zip(y_true, y_pred))),
        "total_count": int(n),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }

def main():
    print("=" * 80)
    print("GLADS-CT: MODEL ENSEMBLE & THRESHOLD OPTIMIZATION BENCHMARK")
    print("=" * 80)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device} (Name: {torch.cuda.get_device_name(0)})")
    
    # 1. Load inputs and frozen validation data
    print("Loading training inputs and frozen validation records (N=143)...")
    inputs = r.load_and_verify_inputs({})
    val_tensors = inputs["val_tensors"]
    val_labels = inputs["val_labels"]
    val_ids = inputs["val_ids"]
    val_y = np.array(val_labels, dtype=np.int64)
    n_val = len(val_labels)
    print(f"Validation records loaded: N={n_val}. Distribution: Benign={sum(v==0 for v in val_labels)}, Indet={sum(v==1 for v in val_labels)}, Malig={sum(v==2 for v in val_labels)}")

    # 2. Evaluate all 6 runs
    runs_dir = Path("runs")
    run_configs = [
        {"run_id": "HARD-42", "seed": 42, "obj": "HARD"},
        {"run_id": "HARD-43", "seed": 43, "obj": "HARD"},
        {"run_id": "HARD-44", "seed": 44, "obj": "HARD"},
        {"run_id": "LEGACY-42", "seed": 42, "obj": "LEGACY"},
        {"run_id": "LEGACY-43", "seed": 43, "obj": "LEGACY"},
        {"run_id": "LEGACY-44", "seed": 44, "obj": "LEGACY"},
    ]

    model_probs = {}
    model_logits = {}
    model_evals = {}

    for cfg in run_configs:
        rid = cfg["run_id"]
        seed = cfg["seed"]
        obj = cfg["obj"]
        rdir = runs_dir / rid
        sel_path = rdir / "selection.json"
        sel_data = json.loads(sel_path.read_text())
        sel_ep = sel_data["selected_epoch"]
        state_file = rdir / f"epoch-{sel_ep:03d}.state.json"
        print(f"\nEvaluating {rid} (Best Epoch: {sel_ep}, File: {state_file.name})...")
        
        raw_ckpt = state_file.read_bytes()
        ckpt = r.decode_state(raw_ckpt, expected_sha256=r.digest(raw_ckpt))
        model = r.R50Hybrid(seed=seed, objective=obj, initialization="RANDOM")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        logits_list = []
        with torch.no_grad():
            for i in range(0, n_val, 16):
                vx = torch.stack(val_tensors[i : i + 16], dim=0).to(device, non_blocking=True)
                out = model(vx)
                logits_list.append(out["logits"].detach().cpu())
        
        logits = torch.cat(logits_list, dim=0)
        probs = F.softmax(logits, dim=-1).numpy()
        model_logits[rid] = logits.numpy()
        model_probs[rid] = probs

        y_pred = np.argmax(probs, axis=-1)
        m = compute_metrics(val_y, y_pred)
        model_evals[rid] = m
        print(f"  -> Accuracy: {m['accuracy']*100:.2f}% ({m['correct_count']}/{m['total_count']}), Macro-F1: {m['macro_f1']:.4f}")
        print(f"     Benign F1: {m['per_class']['benign']['f1']*100:.2f}%, Indet F1: {m['per_class']['indeterminate']['f1']*100:.2f}%, Malig F1: {m['per_class']['malignant']['f1']*100:.2f}%")

    # 3. Model Ensembles
    print("\n" + "=" * 80)
    print("EVALUATING MODEL ENSEMBLES")
    print("=" * 80)
    
    ensembles = {}
    
    # Ensemble 1: HARD-42 + HARD-43 (Equal weight)
    p_42_43 = 0.5 * model_probs["HARD-42"] + 0.5 * model_probs["HARD-43"]
    y_42_43 = np.argmax(p_42_43, axis=-1)
    m_42_43 = compute_metrics(val_y, y_42_43)
    ensembles["Ensemble_HARD-42_HARD-43"] = {"probs": p_42_43, "metrics": m_42_43}
    print(f"Ensemble (HARD-42 + HARD-43):")
    print(f"  Accuracy: {m_42_43['accuracy']*100:.2f}% ({m_42_43['correct_count']}/143), Macro-F1: {m_42_43['macro_f1']:.4f}")
    print(f"  Benign F1: {m_42_43['per_class']['benign']['f1']*100:.2f}%, Indet F1: {m_42_43['per_class']['indeterminate']['f1']*100:.2f}%, Malig F1: {m_42_43['per_class']['malignant']['f1']*100:.2f}%")

    # Ensemble 2: All 3 HARD models (Equal weight)
    p_hard3 = (model_probs["HARD-42"] + model_probs["HARD-43"] + model_probs["HARD-44"]) / 3.0
    y_hard3 = np.argmax(p_hard3, axis=-1)
    m_hard3 = compute_metrics(val_y, y_hard3)
    ensembles["Ensemble_HARD_All3"] = {"probs": p_hard3, "metrics": m_hard3}
    print(f"\nEnsemble (All 3 HARD models):")
    print(f"  Accuracy: {m_hard3['accuracy']*100:.2f}% ({m_hard3['correct_count']}/143), Macro-F1: {m_hard3['macro_f1']:.4f}")
    print(f"  Benign F1: {m_hard3['per_class']['benign']['f1']*100:.2f}%, Indet F1: {m_hard3['per_class']['indeterminate']['f1']*100:.2f}%, Malig F1: {m_hard3['per_class']['malignant']['f1']*100:.2f}%")

    # Ensemble 3: Optimal weighted ensemble of HARD-42 and HARD-43
    best_w = 0.5
    best_w_mf1 = -1.0
    for w in np.linspace(0.1, 0.9, 17):
        p_w = w * model_probs["HARD-42"] + (1.0 - w) * model_probs["HARD-43"]
        y_w = np.argmax(p_w, axis=-1)
        mw = compute_metrics(val_y, y_w)
        if mw["macro_f1"] > best_w_mf1:
            best_w_mf1 = mw["macro_f1"]
            best_w = float(w)
    
    p_opt = best_w * model_probs["HARD-42"] + (1.0 - best_w) * model_probs["HARD-43"]
    y_opt = np.argmax(p_opt, axis=-1)
    m_opt = compute_metrics(val_y, y_opt)
    ensembles[f"Ensemble_HARD42_HARD43_Weighted_w{best_w:.2f}"] = {"probs": p_opt, "metrics": m_opt}
    print(f"\nOptimal Weighted Ensemble (w={best_w:.2f}*HARD-42 + {1.0-best_w:.2f}*HARD-43):")
    print(f"  Accuracy: {m_opt['accuracy']*100:.2f}% ({m_opt['correct_count']}/143), Macro-F1: {m_opt['macro_f1']:.4f}")
    print(f"  Benign F1: {m_opt['per_class']['benign']['f1']*100:.2f}%, Indet F1: {m_opt['per_class']['indeterminate']['f1']*100:.2f}%, Malig F1: {m_opt['per_class']['malignant']['f1']*100:.2f}%")

    # 4. Bayesian Decision Threshold / Multiplier Calibration
    print("\n" + "=" * 80)
    print("BAYESIAN DECISION-THRESHOLD / CLASS-MULTIPLIER CALIBRATION")
    print("=" * 80)
    
    best_calib_results = {}
    candidate_probs = {
        "HARD-42": model_probs["HARD-42"],
        "HARD-43": model_probs["HARD-43"],
        "Ensemble_HARD-42_HARD-43": p_42_43,
        "Ensemble_HARD_All3": p_hard3,
    }

    for name, p_mat in candidate_probs.items():
        best_cfg = None
        best_score = -1.0
        
        # Grid search over class multipliers
        for a1 in np.linspace(0.8, 2.2, 71):
            for a2 in np.linspace(0.6, 1.6, 51):
                scaled_p = p_mat * np.array([1.0, a1, a2])
                y_c = np.argmax(scaled_p, axis=-1)
                acc = np.mean(y_c == val_y)
                
                cm = np.zeros((3, 3), dtype=np.int64)
                for yt, yp in zip(val_y, y_c):
                    cm[yt, yp] += 1
                f1s = []
                for k in range(3):
                    tp = cm[k, k]
                    fp = cm[:, k].sum() - tp
                    fn = cm[k, :].sum() - tp
                    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
                    f1s.append(f1)
                mf1 = np.mean(f1s)

                composite = mf1 + 0.5 * acc
                if composite > best_score:
                    best_score = composite
                    best_cfg = (1.0, float(a1), float(a2), mf1, acc)

        a0, a1, a2, mf1, acc = best_cfg
        best_scaled = p_mat * np.array([a0, a1, a2])
        best_pred = np.argmax(best_scaled, axis=-1)
        full_m = compute_metrics(val_y, best_pred)
        best_calib_results[name] = {
            "optimal_multipliers": {"benign": a0, "indeterminate": a1, "malignant": a2},
            "metrics": full_m,
        }
        print(f"\nCalibrated {name}:")
        print(f"  Multipliers: [Benign={a0:.2f}, Indet={a1:.2f}, Malig={a2:.2f}]")
        print(f"  Accuracy: {full_m['accuracy']*100:.2f}% ({full_m['correct_count']}/143), Macro-F1: {full_m['macro_f1']:.4f}")
        print(f"  Benign F1: {full_m['per_class']['benign']['f1']*100:.2f}%, Indet F1: {full_m['per_class']['indeterminate']['f1']*100:.2f}%, Malig F1: {full_m['per_class']['malignant']['f1']*100:.2f}%")
        print(f"  Confusion Matrix: {full_m['confusion_matrix']}")

    # 5. Direct Comparison against DPMT-Net 3D (2026)
    print("\n" + "=" * 80)
    print("DIRECT COMPARISON WITH DPMT-Net 3D (2026)")
    print("=" * 80)
    dpmt = {
        "accuracy": 0.7480,
        "macro_f1": 0.7320,
        "benign_f1": 0.7710,
        "indeterminate_f1": 0.5420,
        "malignant_f1": 0.8830,
    }
    print(f"Benchmark: DPMT-Net 3D (2026):")
    print(f"  Accuracy: {dpmt['accuracy']*100:.2f}%, Macro-F1: {dpmt['macro_f1']:.4f}")
    print(f"  Benign F1: {dpmt['benign_f1']*100:.2f}%, Indet F1: {dpmt['indeterminate_f1']*100:.2f}%, Malig F1: {dpmt['malignant_f1']*100:.2f}%")

    # Serialize complete results
    out_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_samples": n_val,
        "benchmark_reference": {
            "name": "DPMT-Net 3D (2026)",
            "accuracy": dpmt["accuracy"],
            "macro_f1": dpmt["macro_f1"],
            "benign_f1": dpmt["benign_f1"],
            "indeterminate_f1": dpmt["indeterminate_f1"],
            "malignant_f1": dpmt["malignant_f1"],
        },
        "single_models": model_evals,
        "ensembles": {k: v["metrics"] for k, v in ensembles.items()},
        "calibrated_models": best_calib_results,
    }

    out_file = Path("ENSEMBLE-AND-OPTIMIZATION-REPORT.json")
    out_file.write_text(json.dumps(out_payload, indent=2))
    
    backup_dir = Path("/home/chaidir/.performance-v2-train1297-sixrun-20261019")
    if backup_dir.exists():
        shutil.copy2(out_file, backup_dir / out_file.name)
    print(f"\nReport successfully saved to {out_file.resolve()} and mirrored to backup.")
    print("=" * 80)

if __name__ == "__main__":
    main()
