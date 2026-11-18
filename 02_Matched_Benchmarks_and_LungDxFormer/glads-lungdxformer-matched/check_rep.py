
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")
test_df = df[df["split_slice"] == "test"].reset_index(drop=True)

# Load test predictions from evaluate_and_optimize_test.py
with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/test_evaluation_report.json") as f:
    report = json.load(f)

print("Keys in report:", report.keys())
if "calibrated_ensemble" in report:
    print("Ensemble Acc:", report["calibrated_ensemble"]["ensemble_raw"]["accuracy"])

