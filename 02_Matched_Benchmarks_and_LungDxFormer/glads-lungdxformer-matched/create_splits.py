
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
import hashlib

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/lidc_slice_manifest.csv")

# 1. Patient-Level Stratified Split (70:15:15)
# Using StratifiedGroupKFold to ensure no patient overlap and balanced nodule/slice class ratios
np.random.seed(42)

# Create patient-dominant label for stratification
patient_labels = df.groupby("patient_id")["label"].agg(lambda x: x.mode()[0])
patient_df = pd.DataFrame({"patient_id": patient_labels.index, "patient_label": patient_labels.values})

# Stratified split of patients: 70% train, 30% temp
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_p_idx, temp_p_idx = next(sss1.split(patient_df, patient_df["patient_label"]))

train_patients = set(patient_df.iloc[train_p_idx]["patient_id"])
temp_patient_df = patient_df.iloc[temp_p_idx]

# Split temp (30%) into 15% val and 15% test (50:50 of temp)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_p_idx, test_p_idx = next(sss2.split(temp_patient_df, temp_patient_df["patient_label"]))

val_patients = set(temp_patient_df.iloc[val_p_idx]["patient_id"])
test_patients = set(temp_patient_df.iloc[test_p_idx]["patient_id"])

# Verify zero patient overlap
assert len(train_patients.intersection(val_patients)) == 0, "Train-Val patient overlap!"
assert len(train_patients.intersection(test_patients)) == 0, "Train-Test patient overlap!"
assert len(val_patients.intersection(test_patients)) == 0, "Val-Test patient overlap!"

def assign_patient_split(p_id):
    if p_id in train_patients:
        return "train"
    elif p_id in val_patients:
        return "val"
    else:
        return "test"

df["split_patient"] = df["patient_id"].apply(assign_patient_split)

# 2. Slice-Level Stratified Split (70:15:15)
sss_slice1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_s_idx, temp_s_idx = next(sss_slice1.split(df, df["label"]))

temp_slice_df = df.iloc[temp_s_idx]
sss_slice2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_rel_idx, test_rel_idx = next(sss_slice2.split(temp_slice_df, temp_slice_df["label"]))

val_s_idx = temp_s_idx[val_rel_idx]
test_s_idx = temp_s_idx[test_rel_idx]

df["split_slice"] = "train"
df.iloc[val_s_idx, df.columns.get_loc("split_slice")] = "val"
df.iloc[test_s_idx, df.columns.get_loc("split_slice")] = "test"

# Output manifest
out_csv = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv"
df.to_csv(out_csv, index=False)

with open(out_csv, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()

print("=== SPLIT MANIFEST CREATED ===")
print("Saved to:", out_csv)
print("SHA-256:", h)

print("\n--- Patient-Level Split Summary (split_patient) ---")
print(df["split_patient"].value_counts())
print("Patients per split:", {
    "train": len(train_patients),
    "val": len(val_patients),
    "test": len(test_patients)
})
print("Slices by class in split_patient:")
print(pd.crosstab(df["split_patient"], df["class_name"], margins=True))

print("\n--- Slice-Level Split Summary (split_slice) ---")
print(df["split_slice"].value_counts())
print("Slices by class in split_slice:")
print(pd.crosstab(df["split_slice"], df["class_name"], margins=True))
