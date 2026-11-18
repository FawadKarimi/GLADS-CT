
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Load test predictions and metadata
with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/high_performance_benchmark_report.json") as f:
    rep = json.load(f)

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")
test_df = df[df["split_slice"] == "test"].reset_index(drop=True)

probs = np.array(rep["models"]["GLADS_CT_TTA"]["probs"])
test_df["prob_0"] = probs[:, 0]
test_df["prob_1"] = probs[:, 1]
test_df["prob_2"] = probs[:, 2]

# Aggregate probabilities per nodule
nodule_df = test_df.groupby("nodule_id").agg({
    "prob_0": "mean",
    "prob_1": "mean",
    "prob_2": "mean",
    "label": lambda x: x.mode()[0]
}).reset_index()

nodule_probs = nodule_df[["prob_0", "prob_1", "prob_2"]].values
nodule_preds = nodule_probs.argmax(axis=1)
nodule_targets = nodule_df["label"].values

acc = accuracy_score(nodule_targets, nodule_preds)
macro_f1 = f1_score(nodule_targets, nodule_preds, average="macro")
f1_c = f1_score(nodule_targets, nodule_preds, average=None)
cm = confusion_matrix(nodule_targets, nodule_preds)

print(f"Total Unique Nodules in Test: {len(nodule_df)}")
print(f"Nodule-Aggregated Accuracy: {acc*100:.2f}%")
print(f"Nodule-Aggregated Macro-F1: {macro_f1*100:.2f}%")
print(f"Class F1 (B, I, M): {[round(x*100, 2) for x in f1_c]}")
print(f"Confusion Matrix:\n{cm}")

