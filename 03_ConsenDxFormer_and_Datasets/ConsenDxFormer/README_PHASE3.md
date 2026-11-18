# Phase 3: reproducible LungDxFormer baseline experiments

Phase 3 provides guarded experiment infrastructure for the locked LungDxFormer
B0 and B1 baselines. It does not implement ConsenDxFormer and it does not add a
new training method. The infrastructure is ready for later real experiments,
but the real-data command is intentionally blocked until Phase 2 proves that
the complete raw LIDC-IDRI DICOM source and its derived cohort are ready.

**No real B0/B1 performance results were generated during Phase 3
infrastructure development.**

## Locked baselines

Both variants use the official LungDxFormer source at commit
`0e5bcadcadfce5ea99df8484d23705d7bd4d778a`. The source guard checks the exact
commit, official origin, clean worktree, tracked file content, and a complete
source-tree snapshot. The adapter imports the source read-only; it never edits
`external/LungDxFormer`.

- B0 loads the released Phase 1 profile in `configs/b0.yaml`: 96×96 inputs,
  batch size 16, dropout 0.2, six epochs, and patience eight.
- B1 loads the paper-aligned Phase 1 profile in `configs/b1.yaml`: 64×64
  inputs, batch size 32, dropout 0.5, 100 epochs, and patience ten. It still
  uses the exact released architecture and source.

The Phase 3 files reference those Phase 1 profiles rather than copying their
hyperparameters. Known paper/repository discrepancies remain visible and are
not guessed away, including input size, epoch, batch-size, dropout, patience,
path-aggregation behavior, and parameter-count differences.

## Hard real-data gate

There is no `--force`, skip, or readiness-bypass option. Before creating an
experiment directory or initializing a model, a real command verifies all of
the following:

1. The immutable Phase 0 manifest bundle still has its pinned checksums and
   1,010-patient/1,018-series, 707/152/151 patient split.
2. Phase 2 software/annotation evidence identifies exact official
   `pylidc==0.2.3` and records the scoped NumPy compatibility behavior.
3. A `strict_full`, final-dataset-eligible DICOM/source report covers all
   frozen patients and series, has readable DICOM evidence, has no listed
   reconciliation/completeness defect, and links to the frozen manifest.
4. Canonical extraction and resolution-specific rendering each have valid
   `COMPLETED.json`, checksum ledgers, provenance/config linkage, source
   linkage, and Phase 0 linkage. Canonical leakage evidence must pass, and all
   recorded images/masks and stable identities are revalidated.
5. The configured metadata is the exact deterministic Phase 2 cohort export,
   with the expected B0/B1 resolution and frozen patient split.

The current repository has software/annotation validation only. The complete
raw archive has not been tested, so `DICOM_ARCHIVE_TESTED=false` and
`SOURCE_READINESS_TESTED=false`. A real command therefore fails loudly at the
missing/incomplete Phase 2 evidence. It never falls back to the old processed
791-patient crop dataset.

Synthetic mode is a separate, explicitly marked
`SYNTHETIC_NON_RESEARCH_ONLY` path. It verifies a tiny committed metadata
fixture and the frozen manifest, but it keeps source, DICOM, extracted-cohort,
and real-research readiness false. `--synthetic-dry-run` is mandatory. This
path does not call the trainer and does not emit checkpoints, predictions,
metrics, or performance-shaped output.

## Configuration and commands

The real configs are `configs/phase3_b0.yaml` and
`configs/phase3_b1.yaml`. Each selects its baseline, metadata/cohort, frozen
split, output root, device, deterministic controls, validation checkpoint
policy, and prediction policy. Only seed and device have runner-level runtime
overrides.

Once the locked Phase 1 baseline checkout has been prepared, an
infrastructure-only dry-run can be exercised without internet, GPU, or LIDC
data:

```text
python scripts/30_run_phase3_baseline.py --config configs/phase3_synthetic_b0.yaml --seed 2026 --synthetic-dry-run

python scripts/31_run_phase3_matrix.py --b0-config configs/phase3_synthetic_b0.yaml --b1-config configs/phase3_synthetic_b1.yaml --seeds 2026 2027 --synthetic-dry-run
```

After the Phase 2 strict source check, canonical extraction, both resolution
renderings, and exact metadata exports have been generated and independently
audited, install the locked baseline runtime described in `README_PHASE1.md`,
keep both Git worktrees clean, and use:

```text
python scripts/30_run_phase3_baseline.py --config configs/phase3_b0.yaml --seed <predeclared-seed>
python scripts/30_run_phase3_baseline.py --config configs/phase3_b1.yaml --seed <same-predeclared-seed>

python scripts/31_run_phase3_matrix.py --b0-config configs/phase3_b0.yaml --b1-config configs/phase3_b1.yaml --seeds <explicit-seed-1> <explicit-seed-2> ...
```

Do not run those real commands until the source-readiness gate passes. Seeds
must be declared before examining test results; the orchestrator contains no
hidden default seed list.

## Identity, reproducibility, and artifacts

An experiment ID is content-addressed, not timestamp-addressed. Its SHA-256
preimage contains the B0/B1 variant, seed, locked baseline commit, project Git
commit, combined Phase 3/Phase 1 configuration hash, exact metadata checksum,
logical cohort checksum, ordered-sample checksum, and frozen split bundle
checksum. Real execution additionally requires a clean project worktree.
Existing experiment directories—including completed dry-runs—are never
overwritten. Each directory also snapshots the exact readiness-approved
metadata CSV bytes and binds them to the identity checksum.

Python, NumPy, PyTorch CPU, PyTorch CUDA, and DataLoader worker/generator seeds
are centralized. Deterministic algorithms, cuDNN settings, and the CUDA BLAS
workspace setting are recorded. Environment provenance records Python,
PyTorch, CUDA runtime, cuDNN, GPU names, NumPy, OS/platform, relevant package
versions, project commit, and baseline commit. These controls improve
repeatability; they do not promise bitwise equality across different GPUs,
drivers, framework versions, or operating systems.

Generated outputs are ignored by Git and use this layout:

```text
artifacts/phase3/
  experiments/
    <content-addressed-experiment-id>/
      config/
      provenance/
      checkpoints/
      history/
      predictions/
      metrics/
      status.json
      artifact_manifest.json
      COMPLETED.json               # real run only
      DRY_RUN_COMPLETED.json       # synthetic dry-run only
  run_index.<content-addressed-matrix-id>.json
```

Important JSON, CSV, status, manifest, and completion records use same-directory
temporary files plus atomic replacement. States progress through
`prepared/running/finalizing/completed`, or to `failed`; a synthetic run has a
distinct `synthetic_dry_run_completed` terminal state. A completion marker is
written only after required artifacts validate, Phase 2 readiness and input
fingerprints are revalidated, and both Git commits are proven unchanged.
Failure never creates a completion marker.

## Training, checkpoints, and evaluation

The real adapter constructs the exact released LungDxFormer architecture,
augmentation, cross-entropy loss, Adam optimizer, scheduler, and Phase 1 B0/B1
settings. It consumes only explicit `train`, `val`, and `test` assignments from
the readiness-approved metadata. Train is used for optimization. Validation
loss alone controls scheduler steps, early stopping, and best-checkpoint
selection; an earliest-epoch tie rule is validated independently. The test
dataset is not constructed or evaluated until the selected best checkpoint is
fixed and reloaded.

`best.pt` and `last.pt` are distinct. Both contain model, optimizer, and
scheduler state where present, epoch/best-epoch and best validation loss, seed,
variant, combined config hash, experiment ID, class order, source/data/split
checksums, and the effective baseline configuration. JSON sidecars bind each
checkpoint file checksum to its audit metadata.

Validation and test predictions are deterministic CSVs with this exact class
order: `benign, indeterminate, malignant`. Every row preserves:

```text
patient_id, series_instance_uid, nodule_id, split,
true_class, predicted_class,
prob_benign, prob_indeterminate, prob_malignant,
logit_benign, logit_indeterminate, logit_malignant,
seed, baseline_variant, experiment_id
```

Probabilities are recomputed and checked against a stable softmax of the raw
logits, and rows are sorted by split and stable three-part sample identity.
Final evaluation calls the Phase 1 metric interface explicitly with probability
columns. It reports accuracy, macro and per-class precision/recall/F1,
confusion matrix, and one-vs-rest multiclass AUROC. An unsupported AUROC is
stored as `null` with an explanatory warning, never fabricated.

## Multi-seed and fair-comparison behavior

The matrix runner validates B0/B1 fairness before launching any trainer. Every
paired seed must have the same logical cohort checksum, frozen split checksum,
ordered samples, and class order. TTA must be equivalently disabled. Expected
locked B0/B1 resolution and hyperparameter differences are intentionally not
fairness failures.

Each variant/seed receives an independent content-addressed directory. One
ordinary run failure is recorded without deleting or altering successful or
previously completed runs, and orchestration continues. The atomic run index
records each status and a postflight paired-comparison check. Reusing the exact
matrix identity or an existing experiment identity is refused.

## Scope and limitations

Phase 3 establishes code contracts and failure-safe infrastructure, not:

- validation of the unavailable complete LIDC-IDRI DICOM archive;
- validation of a final real extracted cohort or its sample counts;
- reproduction of the LungDxFormer paper's reported performance;
- evidence that either B0 or B1 performs at any particular level;
- calibration, uncertainty quantification, statistical comparison, or XAI;
- ConsenDxFormer architecture, disagreement losses, attention alignment, or
  cross-scale fusion.

The paper leaves HU window/crop details and other discrepancies unresolved;
Phase 2 provenance must keep those choices explicit. Phase 3 never forces a
4,095-sample cohort and never uses test labels for tuning or checkpoint
selection.
