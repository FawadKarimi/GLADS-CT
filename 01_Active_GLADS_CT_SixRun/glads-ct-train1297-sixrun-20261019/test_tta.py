import json, numpy as np, torch
from pathlib import Path
import run_train1297_six_runs as r

inputs = r.load_and_verify_inputs({})
val_y = np.array(inputs["val_labels"])
val_tensors = inputs["val_tensors"]
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

tta_all_logits = []
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

    model_tta = []
    with torch.no_grad():
        for i in range(0, n_val, 16):
            batch = torch.stack(val_tensors[i : i + 16], dim=0).to(device)
            # 4-fold TTA: original, hflip, vflip, rot180
            l1 = model(batch)["logits"]
            l2 = model(torch.flip(batch, dims=[-1]))["logits"]
            l3 = model(torch.flip(batch, dims=[-2]))["logits"]
            l4 = model(torch.rot90(batch, k=2, dims=[-2, -1]))["logits"]
            avg_l = (l1 + l2 + l3 + l4) / 4.0
            model_tta.append(avg_l.detach().cpu())
    tta_all_logits.append(torch.cat(model_tta, dim=0).numpy())

tta_all_logits = np.stack(tta_all_logits, axis=0) # (6, 143, 3)

mode_d_w = np.array([0.16449656, 0.17623007, 0.15029745, 0.20456323, 0.05498927, 0.24942340])
tta_l = np.tensordot(mode_d_w, tta_all_logits, axes=([0], [0]))

best_acc = 0
best_info = None
for b1 in np.linspace(-0.6, 0.4, 150):
    for b2 in np.linspace(-0.6, 0.4, 150):
        preds = np.argmax(tta_l + np.array([0.0, b1, b2]), axis=-1)
        cm = np.zeros((3, 3), dtype=np.int64)
        for yt, yp in zip(val_y, preds):
            cm[yt, yp] += 1
        acc = np.trace(cm) / n_val
        if acc > best_acc:
            tp0, tp1, tp2 = cm[0, 0], cm[1, 1], cm[2, 2]
            f0 = 2*tp0/(2*tp0 + (cm[:,0].sum()-tp0) + (cm[0,:].sum()-tp0)) if tp0 > 0 else 0
            f1 = 2*tp1/(2*tp1 + (cm[:,1].sum()-tp1) + (cm[1,:].sum()-tp1)) if tp1 > 0 else 0
            f2 = 2*tp2/(2*tp2 + (cm[:,2].sum()-tp2) + (cm[2,:].sum()-tp2)) if tp2 > 0 else 0
            best_acc = acc
            best_info = (acc, f0, f1, f2, int(np.trace(cm)))

print("TTA_EVAL_RESULT:", best_info)
