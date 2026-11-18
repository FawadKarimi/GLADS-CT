#!/usr/bin/env python3
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90"

import json
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

np.random.seed(42)
N_SAMPLES = 250000
weights = np.random.dirichlet(np.ones(6), size=N_SAMPLES)
b1 = np.random.uniform(-1.5, 1.5, size=N_SAMPLES)
b2 = np.random.uniform(-1.5, 1.5, size=N_SAMPLES)

best_malig = None
max_malig_f1 = -1.0

best_indet = None
max_indet_f1 = -1.0

for i in range(N_SAMPLES):
    ens_l = np.tensordot(weights[i], all_logits, axes=([0], [0])) + np.array([0.0, b1[i], b2[i]])
    preds = np.argmax(ens_l, axis=-1)
    
    cm = np.zeros((3, 3), dtype=np.int64)
    for yt, yp in zip(val_y, preds):
        cm[yt, yp] += 1
    
    tp0, tp1, tp2 = cm[0, 0], cm[1, 1], cm[2, 2]
    f0 = 2*tp0/(2*tp0 + (cm[:,0].sum()-tp0) + (cm[0,:].sum()-tp0)) if tp0 > 0 else 0
    f1 = 2*tp1/(2*tp1 + (cm[:,1].sum()-tp1) + (cm[1,:].sum()-tp1)) if tp1 > 0 else 0
    f2 = 2*tp2/(2*tp2 + (cm[:,2].sum()-tp2) + (cm[2,:].sum()-tp2)) if tp2 > 0 else 0
    acc = np.trace(cm)/n_val
    mf1 = (f0 + f1 + f2)/3.0

    if f2 > max_malig_f1:
        max_malig_f1 = f2
        best_malig = {
            "f2": f2, "f1": f1, "f0": f0, "acc": acc, "mf1": mf1,
            "cm": cm.tolist()
        }

    if f1 > max_indet_f1:
        max_indet_f1 = f1
        best_indet = {
            "f2": f2, "f1": f1, "f0": f0, "acc": acc, "mf1": mf1,
            "cm": cm.tolist()
        }

res = {
    "absolute_max_malignant": best_malig,
    "absolute_max_indeterminate": best_indet
}
Path("CLASS_BOUNDARIES.json").write_text(json.dumps(res, indent=2))
print("MAX_CLASS_EVAL_COMPLETE")
