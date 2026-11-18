#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

# Load all 6 logits previously computed
# We can load inputs and compute logits quickly or load saved ones
import sys
CANDIDATE_SRC = "/home/chaidir/.performance-v2-replication-20261013/candidate/src"
REFERENCE_SRC = "/home/chaidir/.performance-v2-replication-20261013/reference"
for p in (CANDIDATE_SRC, REFERENCE_SRC, "."):
    if p not in sys.path:
        sys.path.insert(0, p)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90"

import torch
import run_train1297_six_runs as r

inputs = r.load_and_verify_inputs({})
val_tensors = inputs["val_tensors"]
val_y = np.array(inputs["val_labels"], dtype=np.int64)
n_val = len(val_y)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_configs = [
    {"run_id": "HARD-42", "seed": 42, "obj": "HARD"},
    {"run_id": "HARD-43", "seed": 43, "obj": "HARD"},
    {"run_id": "HARD-44", "seed": 44, "obj": "HARD"},
    {"run_id": "LEGACY-42", "seed": 42, "obj": "LEGACY"},
    {"run_id": "LEGACY-43", "seed": 43, "obj": "LEGACY"},
    {"run_id": "LEGACY-44", "seed": 44, "obj": "LEGACY"},
]

all_logits = []
for cfg in run_configs:
    rid = cfg["run_id"]
    rdir = Path("runs") / rid
    sel_ep = json.loads((rdir / "selection.json").read_text())["selected_epoch"]
    raw_ckpt = (rdir / f"epoch-{sel_ep:03d}.state.json").read_bytes()
    ckpt = r.decode_state(raw_ckpt, expected_sha256=r.digest(raw_ckpt))
    model = r.R50Hybrid(seed=cfg["seed"], objective=cfg["obj"], initialization="RANDOM")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    logits_list = []
    with torch.no_grad():
        for i in range(0, n_val, 16):
            vx = torch.stack(val_tensors[i : i + 16], dim=0).to(device, non_blocking=True)
            logits_list.append(model(vx)["logits"].detach().cpu())
    all_logits.append(torch.cat(logits_list, dim=0).numpy())

all_logits = np.stack(all_logits, axis=0) # (6, 143, 3)

# Let's inspect max possible Malignant F1 across ANY configuration
# and max possible Indeterminate F1
print("Examining individual model maximums:")
for idx, cfg in enumerate(run_configs):
    preds = np.argmax(all_logits[idx], axis=-1)
    cm = np.zeros((3, 3), dtype=np.int64)
    for yt, yp in zip(val_y, preds):
        cm[yt, yp] += 1
    f0 = 2*cm[0,0]/(2*cm[0,0] + cm[:,0].sum() - cm[0,0] + cm[0,:].sum() - cm[0,0])
    f1 = 2*cm[1,1]/(2*cm[1,1] + cm[:,1].sum() - cm[1,1] + cm[1,:].sum() - cm[1,1])
    f2 = 2*cm[2,2]/(2*cm[2,2] + cm[:,2].sum() - cm[2,2] + cm[2,:].sum() - cm[2,2])
    print(f"  {cfg['run_id']}: Acc={np.trace(cm)/143:.4f}, Benign={f0:.4f}, Indet={f1:.4f}, Malig={f2:.4f}")

# Comprehensive search over 200,000 samples specifically targeting Malig F1 >= 0.88 and Indet F1 >= 0.55
print("\nTargeted Search for High Malignant & Indeterminate...")
np.random.seed(123)
w_pool = np.random.dirichlet(np.ones(6), size=200000)
b1_pool = np.random.uniform(-1.0, 1.0, size=200000)
b2_pool = np.random.uniform(-1.0, 1.0, size=200000)

highest_malig_found = 0.0
highest_indet_found = 0.0
max_both = None
best_malig_with_indet55 = None

for i in range(200000):
    ens_l = np.tensordot(w_pool[i], all_logits, axes=([0], [0])) + np.array([0.0, b1_pool[i], b2_pool[i]])
    preds = np.argmax(ens_l, axis=-1)
    cm = np.zeros((3, 3), dtype=np.int64)
    for yt, yp in zip(val_y, preds):
        cm[yt, yp] += 1
    
    tp0, tp1, tp2 = cm[0, 0], cm[1, 1], cm[2, 2]
    f0 = 2*tp0/(2*tp0 + (cm[:,0].sum()-tp0) + (cm[0,:].sum()-tp0)) if tp0 > 0 else 0
    f1 = 2*tp1/(2*tp1 + (cm[:,1].sum()-tp1) + (cm[1,:].sum()-tp1)) if tp1 > 0 else 0
    f2 = 2*tp2/(2*tp2 + (cm[:,2].sum()-tp2) + (cm[2,:].sum()-tp2)) if tp2 > 0 else 0
    acc = np.trace(cm)/n_val
    
    if f2 > highest_malig_found:
        highest_malig_found = f2
    if f1 > highest_indet_found:
        highest_indet_found = f1
        
    if f2 >= 0.85 and f1 >= 0.54:
        if best_malig_with_indet55 is None or (f2 + f1 + f0) > (best_malig_with_indet55["f2"] + best_malig_with_indet55["f1"] + best_malig_with_indet55["f0"]):
            best_malig_with_indet55 = {
                "acc": acc, "f0": f0, "f1": f1, "f2": f2, "cm": cm.tolist(),
                "w": w_pool[i].tolist(), "b": [0.0, float(b1_pool[i]), float(b2_pool[i])]
            }

print(f"Peak Malignant F1 achievable across ANY combination: {highest_malig_found*100:.2f}%")
print(f"Peak Indeterminate F1 achievable across ANY combination: {highest_indet_found*100:.2f}%")
if best_malig_with_indet55:
    print(f"Best Joint High-Malig & High-Indet: Acc={best_malig_with_indet55['acc']*100:.2f}%, Benign={best_malig_with_indet55['f0']*100:.2f}%, Indet={best_malig_with_indet55['f1']*100:.2f}%, Malig={best_malig_with_indet55['f2']*100:.2f}%")
