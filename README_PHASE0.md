# ConsenDxFormer — Phase 0: Environment, Source, and Patient-Split Lock

**Project title:** *ConsenDxFormer: A Disagreement-Aware CNN–Transformer Model with Explainable Attention for Pulmonary Nodule Malignancy Stratification*

This scaffold implements **Phase 0 only**. It does not train B0/B1 and does not implement the proposed model.

## What Phase 0 locks

1. **Baseline source identity**
   - Official GitHub repository: `Srinivasa459/LungDxFormer`
   - Locked Git commit: `0e5bcadcadfce5ea99df8484d23705d7bd4d778a`
   - Paper URL: Nature Scientific Reports article `s41598-026-50449-7`
   - Zenodo DOI: `10.5281/zenodo.19649897` (project-supplied identifier; no byte-level archive equivalence is claimed here)

2. **Environment/config identity**
   - YAML configuration with inheritance
   - SHA-256 hash of the resolved config and source-lock file
   - Python/package/GPU metadata
   - global random seed request (`2026` by default)

3. **Patient-disjoint split identity**
   - split is generated from **unique patient IDs**, not from CT series rows
   - every CT series belonging to a patient inherits that patient's split
   - malignancy scores, nodule eligibility, and boundary-zone status are intentionally not used during splitting
   - exact patient IDs are written to CSV and per-split text files
   - all manifest files receive SHA-256 hashes

## Why 1,010 patients and 1,018 CT series are not contradictory

The TCIA collection reports **1,010 unique patients** and **1,018 CT series**. Therefore `1,018` is not a valid patient count. A patient-level split must deduplicate by patient ID first. Some patients have more than one CT series. This scaffold:

- validates `unique(patient_id) == 1010` for the full locked inventory;
- validates `unique(SeriesInstanceUID) == 1018` after CT filtering;
- assigns splits only over the 1,010 unique patients;
- propagates each patient's split to all of that patient's CT series;
- fails if a series UID maps to multiple patients or a patient appears in multiple splits.

With a 70/15/15 split and the documented largest-remainder rule, 1,010 patients become **707 train / 152 validation / 151 test** for the default tie priority. These counts are not copied from the LungDxFormer paper; they are a deterministic consequence of splitting 1,010 unique patients.

## Expected input for the manifest generator

Use the TCIA DICOM Metadata Digest CSV or another complete LIDC-IDRI CT-series inventory containing at minimum:

- patient/subject ID;
- SeriesInstanceUID / Series UID;
- modality if the CSV also contains non-CT series.

The script auto-detects common column names. It does **not** infer patient identity from filenames.

## Commands

Install Phase 0 dependencies:

```bash
pip install -r requirements-phase0.txt
```

Record environment/config state:

```bash
python scripts/00_lock_environment.py --config configs/phase0.yaml
```

Clone/check out the locked baseline without editing it:

```bash
python scripts/01_prepare_locked_baseline.py --config configs/phase0.yaml --clone-if-missing
```

Generate the patient manifest after you have the TCIA metadata CSV:

```bash
python scripts/02_generate_patient_manifest.py \
  --config configs/phase0.yaml \
  --inventory-csv /absolute/path/to/LIDC-IDRI-metadata.csv
```

### Required successful manifest output

A final full-inventory run must print:

```text
PHASE0_MANIFEST=PASS
observed_unique_patients=1010
observed_unique_ct_series=1018
leakage_check_passed=True
```

It also writes:

- `artifacts/phase0/manifests/patient_split.csv`
- `artifacts/phase0/manifests/series_split.csv`
- `artifacts/phase0/manifests/train_patient_ids.txt`
- `artifacts/phase0/manifests/val_patient_ids.txt`
- `artifacts/phase0/manifests/test_patient_ids.txt`
- `artifacts/phase0/manifests/split_summary.json`
- `artifacts/phase0/manifests/manifest_checksums.json`

Do not edit these after generation. If the source inventory or split rule changes, generate a new manifest bundle and retain the old bundle for auditability.

## Important limitation

Phase 0 cannot know whether the eventual malignancy-filtered cohort contains exactly 4,095 nodules. That is a Phase 2 question. The patient split is intentionally fixed **before** that filtering. Phase 2 must report actual eligible-nodule and class counts by split without changing these patient assignments.

## Reproducibility caveat

Seed control and deterministic PyTorch settings reduce nondeterminism, but they do not guarantee bitwise-identical training across different GPUs, drivers, CUDA/cuDNN versions, PyTorch versions, or nondeterministic kernels. The environment lock records these factors instead of claiming perfect reproducibility.
