# ConsenDxFormer — Phase 1: Locked B0/B1 Baseline Infrastructure

Phase 1 provides guarded, testable wrappers around the official LungDxFormer source. It does **not** construct the Phase 2 nodule cohort, run a real experiment, implement ConsenDxFormer, or make a performance claim.

Evidence is locked to the [Scientific Reports article](https://doi.org/10.1038/s41598-026-50449-7) and [official source commit](https://github.com/Srinivasa459/LungDxFormer/tree/0e5bcadcadfce5ea99df8484d23705d7bd4d778a).

## Baseline definitions

- **B0 (released baseline):** the architecture and default protocol in LungDxFormer commit `0e5bcadcadfce5ea99df8484d23705d7bd4d778a`. Only metadata and output paths are supplied externally.
- **B1 (paper-aligned baseline):** the same released source and architecture, with external configuration changes for paper-reported settings that the released API can express. It is not an exact paper reproduction because the paper, source, and available data do not fully agree.

Both modes remain three-class in the fixed order `benign`, `indeterminate`, `malignant`. The wrappers never copy or rewrite the model. Before and after an execution they require the exact source SHA, official origin, a clean Git worktree, and an unchanged full-tree snapshot.

## Paper/source/B0/B1 provenance

| Parameter or behavior | Paper | Pinned released code | B0 | B1 | Status / ambiguity |
|---|---|---|---|---|---|
| ROI input | 64×64 | 96×96 | 96×96, source | 64×64, paper | Aligned in B1 config. Phase 2 must prepare authoritative ROIs. |
| Batch size | 32 | 16 | 16, source | 32, paper | Aligned in B1 config. |
| Maximum epochs | 100 | 6 | 6, source | 100, paper | Aligned in B1 config. |
| Learning rate | 1e-4 | 1e-4 | 1e-4, both | 1e-4, both | No discrepancy. |
| Dropout | 0.5 after all dense layers | One `dropout: 0.2` value controls every transformer block and one classifier dropout located only between its two linear layers | 0.2, source | 0.5, paper | Placement does not match: B1 necessarily changes transformer dropout too and the released API cannot add paper-described dropout after every dense layer. |
| Early stopping | Validation loss, patience 10 | Validation loss, patience 8 | 8, source | 10, paper | Aligned in B1 config. |
| Optimizer / L2 | Adam; L2 on fully connected weights, coefficient unreported | Adam; weight decay 1e-5 on all parameters | Source | Source | Scope/coefficient cannot be matched without invention or source changes. |
| Scheduler | Reduce LR when validation saturates; details unreported | `ReduceLROnPlateau`, validation loss, factor 0.5, patience 3 | Source | Source | B1 retains released details because paper does not specify them. |
| Class weighting | Cross-entropy weighted inversely to class frequency | sklearn balanced weights passed to cross-entropy | Source | Source | Broadly aligned; numeric weights depend on future Phase 2 training metadata. |
| Augmentation | H/V flips 50%, rotation ±10°, “Random Scale” 0.9–1.1, contrast, Gaussian noise; elsewhere calls scaling “intensity scaling” and also claims stronger indeterminate augmentation | H flip 50%, no V flip, rotation ±15°, translation ±0.08, intensity scale/shift ±0.08, noise SD 0.01; same augmenter for every class | Source | Enables V flip and ±10° rotation; retains other source settings | Whether the paper's scale is spatial or intensity scaling is ambiguous; its contrast/noise magnitudes and class-targeted protocol are absent or underspecified, so none are reconstructed. |
| Fusion architecture | Claims path-aggregation/multi-scale fusion; detailed equation shows CNN + attention-feature concatenation and optional raw transformer information | Final CNN projection + attention-refined transformer concatenation and 1×1 projection; raw-transformer fusion disabled | Released architecture | Same released architecture | Claims/equations/source are inconsistent. No narrative architecture is reconstructed. |
| Numeric architecture | CNN/transformer channel widths, embedding width, attention heads/layers, MLP ratio, and classifier width are not reported precisely | CNN channels 32/64/128; embedding 128; 4 heads; 2 layers; MLP ratio 4; classifier width 128 | Source | Source | B1 retains all released architecture values; no missing paper values are inferred. |
| Transformer details | Describes post-attention residual/layer norm and ReLU FFN | Pre-normalization and GELU | Source | Source | Architecture discrepancy remains visible. |
| Attention dropout / AMP / evaluation flags | Not reported | Declared config fields are unused by training | Source | Source | Recorded, not presented as operational settings. |
| Preprocessing / labels | HU clipping, normalization, and four-reader mean; methods use `<2.5`, `==3.0`, and `>3.5` with boundary values excluded, while a later recap phrases the 2.5/3.5 boundaries differently | Training loads prepared images; helper scaffold maps one score as `<=2`, `==3`, otherwise malignant and does not build the reported cohort | External metadata only | External metadata only | Paper thresholds are internally inconsistent/underspecified and the helper is not equivalent. Phase 1 does not use either to fabricate Phase 2 data. |
| Evaluation design | Abstract says patient-wise cross-validation; methods give one 70/15/15 holdout | One holdout; random fallback if CSV lacks `split` | Frozen holdout | Frozen holdout | Adapter always supplies `split`; random fallback is never used. |
| Patient/series identity | Elsewhere says 1,010 volumes, but training section calls 1,018 scans “patients” and reports 713/153/152 | No authoritative cohort bundled | Frozen 1,010-patient split | Same frozen split | Phase 0 establishes 1,010 unique patients, 1,018 CT series, and 707/152/151 patients. |
| Seed | Unreported | 42 | 42, source | 42, retained | No paper seed is invented. Future multi-seed runs may explicitly override it. |

The literature-reported 97.35% accuracy is not a result of this repository.

## Frozen split and Phase 2 handoff

Every runner validates the committed manifest ledger against a separately pinned bundle hash, then validates 1,010 unique patients, 1,018 series, patient counts `707/152/151`, disjointness, and series-to-patient inheritance. Newline normalization is limited to CRLF→LF for cross-platform verification of the original LF hashes. The guard never regenerates or blesses a manifest.

Phase 2 may provide any nodule-level CSV with columns equivalent to `patient_id`, `image_path`, and `label`. The adapter canonicalizes the three labels and joins every row to the frozen patient mapping. Unknown patients fail by default; an explicit `drop` policy is available for controlled diagnostics. It emits one upstream-compatible CSV containing a mandatory `split` column plus per-split CSVs. It never row-splits or random-splits.

## Setup and tests

Install the lightweight infrastructure/test dependencies:

```bash
python -m pip install -r requirements-phase1.txt
pytest -q
python -m compileall -q src scripts
```

Actual LungDxFormer execution additionally needs the dependencies in its locked `requirements.txt`. Prepare its ignored checkout once:

```bash
python scripts/01_prepare_locked_baseline.py --config configs/phase0.yaml --clone-if-missing
```

## Dry runs

The checked-in metadata-only fixture uses real frozen patient IDs but nonexistent synthetic paths; dry-run validation does not load images or require a GPU:

```bash
python scripts/10_run_b0.py \
  --metadata tests/fixtures/phase1_smoke_metadata.csv \
  --output-dir artifacts/phase1/runs/b0-dry-run \
  --dry-run

python scripts/11_run_b1.py \
  --metadata tests/fixtures/phase1_smoke_metadata.csv \
  --output-dir artifacts/phase1/runs/b1-dry-run \
  --dry-run
```

Use `--image-root` for relative Phase 2 image paths. Runtime-safe overrides are repeatable, for example `--set seed=7` or `--set metadata.columns.label=target`. Architecture and protocol overrides are rejected so B0/B1 identities remain meaningful.

Omit `--dry-run` only after supplying prepared images and installing the locked baseline dependencies. A fresh output directory is required. Provenance records process state and return code; directory creation alone is never treated as successful training.

## Evaluation interface

Phase 3 may supply an existing prediction CSV with a sample/patient identifier, true label, predicted class, and optionally all three probability or logit columns. Phase 1 computes only classification metrics, a fixed-order confusion matrix, and one-vs-rest AUROC; unavailable AUROC is recorded as `null` with a warning.

```bash
python scripts/12_evaluate_baseline.py \
  --mode B0 \
  --predictions-csv /path/to/predictions.csv \
  --output-json /path/to/metrics.json
```

The command verifies the same locked baseline source and frozen manifest before reading predictions, and records their identities alongside the metrics. Recognized probability names include `prob_benign`, `prob_indeterminate`, and `prob_malignant`; use the explicit CLI column options for another schema. The evaluator never generates prediction rows and does not implement Phase 5 calibration, uncertainty, or statistical analyses.
