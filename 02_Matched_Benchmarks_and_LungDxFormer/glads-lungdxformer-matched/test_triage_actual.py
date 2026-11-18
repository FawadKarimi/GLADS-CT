
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load high performance benchmark report
with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/high_performance_benchmark_report.json") as f:
    rep = json.load(f)

glads_tta = rep["models"]["GLADS_CT_TTA"]
probs = np.array(glads_tta["probs"])
targets = np.array(glads_tta["targets"])
preds = np.array(glads_tta["preds"])

max_p = np.max(probs, axis=1)

print("Total test samples:", len(targets))
print(f"Overall Accuracy (all {len(targets)}): {accuracy_score(targets, preds)*100:.2f}%")

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    mask = (max_p >= threshold)
    n_kept = np.sum(mask)
    cov = n_kept / len(targets)
    if n_kept > 0:
        t_acc = accuracy_score(targets[mask], preds[mask])
        t_f1 = f1_score(targets[mask], preds[mask], average="macro", zero_division=0)
        f1_c = f1_score(targets[mask], preds[mask], average=None, zero_division=0)
        print(f"Threshold P >= {threshold:.2f} | Kept: {n_kept}/{len(targets)} ({cov*100:.1f}%) | Acc: {t_acc*100:.2f}% | Macro-F1: {t_f1*100:.2f}% | F1(B/I/M): {f1_c[0]*100:.1f}/{f1_c[1]*100:.1f}/{f1_c[2]*100:.1f}")

