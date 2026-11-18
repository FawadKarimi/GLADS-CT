
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import json

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")
test_df = df[df["split_slice"] == "test"].reset_index(drop=True)

# Load test predictions from the 3-seed GLADS ensemble
# Let's inspect test_df with nodule grouping
print("Test Slices:", len(test_df))
print("Unique Nodules in Test:", test_df["nodule_id"].nunique())
print("Unique Patients in Test:", test_df["patient_id"].nunique())
