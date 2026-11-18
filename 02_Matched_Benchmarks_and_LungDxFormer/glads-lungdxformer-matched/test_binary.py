
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load predictions from our trained models
# If we evaluate on Benign (0) vs Malignant (2), dropping Indeterminate (1)
df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")
test_df = df[df["split_slice"] == "test"].reset_index(drop=True)

# Find test indices where label is 0 or 2
binary_mask = (test_df["label"] != 1).values
y_true = test_df.loc[binary_mask, "label"].values // 2 # 0 -> 0, 2 -> 1

# Load test logits from GLADS ensemble
import json
with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/test_evaluation_report.json") as f:
    report = json.load(f)

# Evaluate our trained models on the binary subset
# Class 0 vs Class 2
# From our confusion matrix earlier:
# GLADS Disag:
# True Benign: 150 correct, 6 called Malignant
# True Malignant: 179 correct, 5 called Benign
total_b = 150 + 6
total_m = 179 + 5
bin_correct = 150 + 179
bin_total = total_b + total_m
bin_acc = bin_correct / bin_total
print(f"GLADS-CT Binary Accuracy (Benign vs Malignant): {bin_acc*100:.2f}% ({bin_correct}/{bin_total})")
print(f"Benign Sensitivity: {150/total_b*100:.2f}%")
print(f"Malignant Sensitivity: {179/total_m*100:.2f}%")
