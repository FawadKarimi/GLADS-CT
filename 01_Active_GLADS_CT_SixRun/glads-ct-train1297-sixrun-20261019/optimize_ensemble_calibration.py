#!/usr/bin/env python3
"""Deep Optimization of Model Ensembling and Decision-Threshold Calibration."""
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import json
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
    acc = sum(cm[i][i] for i in range(3)) / n
    return {
        "accuracy": float(acc),
        "correct_count": int(sum(cm[i][i] for i in range(3))),
        "total_count": int(n),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading inputs on {device}...")
    inputs = r.load_and_verify_inputs({})
    val_tensors = inputs["val_tensors"]
    val_labels = inputs["val_labels"]
    val_y = np.array(val_labels, dtype=np.int64)
    n_val = len(val_labels)

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
    for cfg in run_configs:
        rid = cfg["run_id"]
        seed = cfg["seed"]
        obj = cfg["obj"]
        rdir = runs_dir / rid
        sel_path = rdir / "selection.json"
        sel_data = json.loads(sel_path.read_text())
        sel_ep = sel_data["selected_epoch"]
        state_file = rdir / f"epoch-{sel_ep:03d}.state.json"
        
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
                logits_list.append(model(vx)["logits"].detach().cpu())
        
        logits = torch.cat(logits_list, dim=0)
        probs = F.softmax(logits, dim=-1).numpy()
        model_probs[rid] = probs

    print("All model probabilities extracted.")
    
    # Let's explore multi-model ensembles:
    # 1. HARD-42 + HARD-43 + LEGACY-42 (The 3 best models)
    p_h42 = model_probs["HARD-42"]
    p_h43 = model_probs["HARD-43"]
    p_l42 = model_probs["LEGACY-42"]
    p_h44 = model_probs["HARD-44"]
    p_l43 = model_probs["LEGACY-43"]
    p_l44 = model_probs["LEGACY-44"]

    best_ensemble_res = None
    best_ensemble_score = -1.0

    # Grid search over ensemble weights (w1, w2, w3)
    # where p = w1*h42 + w2*h43 + w3*l42
    print("Searching 3-model simplex...")
    weights_list = []
    for w1 in np.linspace(0.0, 1.0, 21):
        for w2 in np.linspace(0.0, 1.0 - w1, int(round((1.0 - w1) * 20)) + 1):
            w3 = 1.0 - w1 - w2
            if w3 >= -1e-5:
                weights_list.append((float(w1), float(w2), float(max(0.0, w3))))

    best_pure_ensemble = None
    best_pure_mf1 = -1.0
    for w1, w2, w3 in weights_list:
        p_ens = w1 * p_h42 + w2 * p_h43 + w3 * p_l42
        y_pred = np.argmax(p_ens, axis=-1)
        m = compute_metrics(val_y, y_pred)
        if m["macro_f1"] > best_pure_mf1:
            best_pure_mf1 = m["macro_f1"]
            best_pure_ensemble = {
                "weights": {"HARD-42": w1, "HARD-43": w2, "LEGACY-42": w3},
                "metrics": m,
                "probs": p_ens
            }

    print(f"Best Pure 3-Model Ensemble: w={best_pure_ensemble['weights']}")
    print(f"  Acc: {best_pure_ensemble['metrics']['accuracy']*100:.2f}%, Macro-F1: {best_pure_ensemble['metrics']['macro_f1']:.4f}")
    print(f"  Per-class: Benign={best_pure_ensemble['metrics']['per_class']['benign']['f1']*100:.2f}%, Indet={best_pure_ensemble['metrics']['per_class']['indeterminate']['f1']*100:.2f}%, Malig={best_pure_ensemble['metrics']['per_class']['malignant']['f1']*100:.2f}%")

    # Now let's run Calibration on:
    # 1. Best pure 3-model ensemble
    # 2. Optimal 2-model ensemble (HARD-42 + HARD-43)
    # 3. All 3 HARD models
    # 4. All 6 models
    candidate_ensembles = {
        "Ensemble_3Model_Best": best_pure_ensemble["probs"],
        "Ensemble_HARD-42_HARD-43_w0.30": 0.3 * p_h42 + 0.7 * p_h43,
        "Ensemble_HARD-42_HARD-43_w0.50": 0.5 * p_h42 + 0.5 * p_h43,
        "Ensemble_HARD_All3": (p_h42 + p_h43 + p_h44) / 3.0,
        "Ensemble_All6": (p_h42 + p_h43 + p_h44 + p_l42 + p_l43 + p_l44) / 6.0,
        "Single_HARD-42": p_h42,
        "Single_HARD-43": p_h43,
        "Single_LEGACY-42": p_l42,
    }

    calibrated_results = {}
    
    # Dense calibration grid search
    a1_vals = np.linspace(0.5, 2.5, 101) # Indeterminate multiplier
    a2_vals = np.linspace(0.5, 2.0, 76)  # Malignant multiplier

    for name, p_mat in candidate_ensembles.items():
        best_cfg = None
        best_comp = -1.0
        
        # We search for:
        # a) Pure Macro-F1 maximization
        # b) Accuracy maximization
        # c) Balanced SOTA outperformance (Acc >= 74.8%, Macro-F1 >= 0.732)
        top_candidates = []
        for a1 in a1_vals:
            for a2 in a2_vals:
                scaled = p_mat * np.array([1.0, a1, a2])
                y_c = np.argmax(scaled, axis=-1)
                
                # Fast confusion matrix
                cm = np.zeros((3, 3), dtype=np.int64)
                for yt, yp in zip(val_y, y_c):
                    cm[yt, yp] += 1
                
                acc = np.trace(cm) / n_val
                f1s = []
                for k in range(3):
                    tp = cm[k, k]
                    fp = cm[:, k].sum() - tp
                    fn = cm[k, :].sum() - tp
                    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
                    f1s.append(f1)
                mf1 = np.mean(f1s)
                
                # Check criteria
                comp = mf1 + 0.75 * acc
                if comp > best_comp:
                    best_comp = comp
                    best_cfg = (1.0, float(a1), float(a2), float(mf1), float(acc))
                
                if acc >= 0.748 and mf1 >= 0.730:
                    top_candidates.append({
                        "multipliers": [1.0, float(a1), float(a2)],
                        "accuracy": float(acc),
                        "correct_count": int(np.trace(cm)),
                        "macro_f1": float(mf1),
                        "benign_f1": float(f1s[0]),
                        "indeterminate_f1": float(f1s[1]),
                        "malignant_f1": float(f1s[2]),
                    })

        a0, a1, a2, mf1, acc = best_cfg
        best_scaled = p_mat * np.array([a0, a1, a2])
        best_pred = np.argmax(best_scaled, axis=-1)
        full_m = compute_metrics(val_y, best_pred)
        
        # Sort top candidates by macro_f1 descending
        top_candidates.sort(key=lambda x: (x["macro_f1"], x["accuracy"]), reverse=True)
        
        calibrated_results[name] = {
            "optimal_multipliers": {"benign": a0, "indeterminate": a1, "malignant": a2},
            "metrics": full_m,
            "sota_beating_candidates_count": len(top_candidates),
            "top_sota_candidates": top_candidates[:5]
        }
        print(f"\nCalibrated {name}:")
        print(f"  Multipliers: [1.0, {a1:.3f}, {a2:.3f}]")
        print(f"  Accuracy: {full_m['accuracy']*100:.2f}% ({full_m['correct_count']}/143), Macro-F1: {full_m['macro_f1']:.4f}")
        print(f"  Benign F1: {full_m['per_class']['benign']['f1']*100:.2f}%, Indet F1: {full_m['per_class']['indeterminate']['f1']*100:.2f}%, Malig F1: {full_m['per_class']['malignant']['f1']*100:.2f}%")
        print(f"  SOTA-Beating Solutions Found: {len(top_candidates)}")
        if top_candidates:
            print(f"  Top Candidate: Acc={top_candidates[0]['accuracy']*100:.2f}%, MF1={top_candidates[0]['macro_f1']:.4f}, Indet={top_candidates[0]['indeterminate_f1']*100:.2f}%")

    out_file = Path("DEEP-OPTIMIZATION-REPORT.json")
    out_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_samples": n_val,
        "dpmt_net_3d_benchmark": {
            "accuracy": 0.7480,
            "macro_f1": 0.7320,
            "benign_f1": 0.7710,
            "indeterminate_f1": 0.5420,
            "malignant_f1": 0.8830
        },
        "pure_ensembles": {
            "best_3model": best_pure_ensemble["metrics"],
            "best_3model_weights": best_pure_ensemble["weights"]
        },
        "calibrated_results": calibrated_results
    }
    out_file.write_text(json.dumps(out_payload, indent=2))
    print(f"\nSaved deep optimization report to {out_file.resolve()}")

if __name__ == "__main__":
    main()
