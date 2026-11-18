# GLADS-CT Consolidated Project Workspace Guide (`01_`–`04_`)

To keep the published `research/r50-2026-10-22/SHA256-LEDGER.json` verification passing (`README.md` is included in that byte-level ledger), root `README.md` is preserved byte-identical from `main`.

Alongside the root `research/r50-2026-10-22/` snapshot and baseline `src/`/`configs/`/`scripts/`, this repository now includes the curated consolidated workspace directories from our research server:

- **`01_Active_GLADS_CT_SixRun/`** – Six-run expanded cohort (`TRAIN1297`) scripts, Mode D+ calibration & optimization tools (`run_train1297_six_runs.py`, `calibrate_mode_d_plus.py`, `evaluate_ensemble_and_optimize.py`), JSON reports, and per-epoch metric records (`runs/*/epoch-*.metrics.json`).
- **`02_Matched_Benchmarks_and_LungDxFormer/`** – Matched GLADS-CT / LungDxFormer benchmark suite (`splits_paper_matched.csv`, `lidc_slice_manifest.csv`) and the `LungDxFormer` baseline codebase.
- **`03_ConsenDxFormer_and_Datasets/`** – Full `ConsenDxFormer` phase pipeline (`README_PHASE0..3.md`, configs, source, tests, notebooks, `canonical_nodule_ledger.csv`, and Phase 3–5 metric tables/summaries) plus supervisor presentation notebook.
- **`04_Performance_v2_Audits_and_Transfers/`** – Provenance, diagnostic scripts, and ImageNet/RadImageNet transfer audit records (`performance-v2-audit/`, `performance-v2-*-transfer-private-*/`).

## Excluded Large Assets (`.gitignore`)

Multi-GB CT volumes/rendered PNG slices (`rendered_64/`, `rendered_96/`, `lidc_real/`), trained checkpoints (`*.pt`, `*.pth`, `*.state.json`), raw handover archives (`05_Handover_Packages_and_Archives/`), server data symlinks (`06_Shared_Data_Links/`), and oversized raw logs/dumps (`perturbation_curves.csv`, `*.log`) are excluded via `.gitignore`.

Run `python tools/verify_publication_snapshot.py` to confirm all 226 released snapshot hashes remain `PASS`.
