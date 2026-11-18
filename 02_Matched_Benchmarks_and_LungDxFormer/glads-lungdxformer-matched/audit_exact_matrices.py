
import json
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/high_performance_benchmark_report.json") as f:
    rep = json.load(f)

glads_tta = rep["models"]["GLADS_CT_TTA"]
probs = np.array(glads_tta["probs"])
targets = np.array(glads_tta["targets"])
preds = np.array(glads_tta["preds"])
max_p = np.max(probs, axis=1)

def print_audit(thresh_name, mask):
    t_sub = targets[mask]
    p_sub = preds[mask]
    cm = confusion_matrix(t_sub, p_sub, labels=[0, 1, 2])
    acc = np.sum(t_sub == p_sub) / len(t_sub)
    f1_c = f1_score(t_sub, p_sub, average=None, labels=[0, 1, 2], zero_division=0)
    macro_f1 = np.mean(f1_c)
    
    # Class supports
    support = [int(np.sum(t_sub == c)) for c in [0, 1, 2]]
    correct = int(np.sum(t_sub == p_sub))
    total = len(t_sub)
    
    print(f"=== {thresh_name} ===")
    print(f"Total: {total}/{len(targets)} ({total/len(targets)*100:.4f}%) | Correct: {correct}")
    print(f"Exact Accuracy: {correct}/{total} = {acc*100:.4f}%")
    print(f"Confusion Matrix (Rows=True, Cols=Pred):\n{cm}")
    print(f"Class Support (B, I, M): {support}")
    print(f"Class F1 (B, I, M): {[round(x*100, 4) for x in f1_c]}")
    print(f"Macro F1 (exact mean of class F1s): {macro_f1*100:.4f}%\n")

print_audit("FULL COVERAGE (NO REJECTION)", np.ones(len(targets), dtype=bool))
print_audit("THRESHOLD P >= 0.65", max_p >= 0.65)
print_audit("THRESHOLD P >= 0.85", max_p >= 0.85)

