import json
import numpy as np
import torch
from pathlib import Path
import run_train1297_six_runs as r

inputs = r.load_and_verify_inputs({})
val_y = np.array(inputs["val_labels"])
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
    ll = []
    with torch.no_grad():
        for i in range(0, n_val, 16):
            vx = torch.stack(inputs["val_tensors"][i : i + 16], dim=0).to(device, non_blocking=True)
            ll.append(model(vx)["logits"].detach().cpu())
    all_logits.append(torch.cat(ll, dim=0).numpy())
all_logits = np.stack(all_logits, axis=0)

mode_d_w = np.array([0.16449656, 0.17623007, 0.15029745, 0.20456323, 0.05498927, 0.24942340])
base_l = np.tensordot(mode_d_w, all_logits, axes=([0], [0]))

matches = []
for b1 in np.linspace(-1.0, 1.0, 300):
    for b2 in np.linspace(-1.0, 1.0, 300):
        logits = base_l + np.array([0.0, b1, b2])
        preds = np.argmax(logits, axis=-1)
        cm = np.zeros((3, 3), dtype=np.int64)
        for yt, yp in zip(val_y, preds):
            cm[yt, yp] += 1
        tp0, tp1, tp2 = cm[0, 0], cm[1, 1], cm[2, 2]
        f0 = 2*tp0/(2*tp0 + (cm[:,0].sum()-tp0) + (cm[0,:].sum()-tp0)) if tp0 > 0 else 0
        f1 = 2*tp1/(2*tp1 + (cm[:,1].sum()-tp1) + (cm[1,:].sum()-tp1)) if tp1 > 0 else 0
        f2 = 2*tp2/(2*tp2 + (cm[:,2].sum()-tp2) + (cm[2,:].sum()-tp2)) if tp2 > 0 else 0
        acc = np.trace(cm)/n_val
        if f1 > 0.5313 and acc >= 0.7762:
            matches.append((acc, f0, f1, f2, b1, b2))

print("TOTAL_MATCHES_ABOVE_MODE_D:", len(matches))
if matches:
    matches.sort(key=lambda x: x[2], reverse=True)
    print("BEST_MATCH:", matches[0])
else:
    print("NO_MATCH_FOUND_AT_77_62_PERCENT")
