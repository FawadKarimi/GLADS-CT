# GLADS-CT

Research code and retained aggregate evidence for radiologist-derived three-class lung-nodule assessment on LIDC-IDRI.
This repository is research software. It has not established clinical safety, calibration, diagnostic efficacy or superiority over other studies.

## Start here

- [R50 paper source and evidence snapshot](research/r50-2026-10-22/README.md)
- [Scientific protocol](research/r50-2026-10-22/protocols/SCIENTIFIC-PROTOCOL.json)
- [Results and their interpretation](research/r50-2026-10-22/RESULTS.md)
- [Source provenance and reproducibility limits](research/r50-2026-10-22/REPRODUCIBILITY.md)
- [Software citation](CITATION.cff), [license](LICENSE), [third-party terms](THIRD_PARTY_NOTICES.md)

The original root `src/`, `configs/`, `scripts/`, `pyproject.toml` and `README_PHASE*.md` remain historical Phase 0/1 baseline infrastructure. They are not the executed R50 paper implementation. The historical `phase4-consendxformer` branch uses a custom encoder. Do not substitute either for the R50 snapshot.

## What the experiments establish

HARD is class-weighted categorical cross-entropy and has **no reader-distribution loss**. LEGACY mixes hard labels with reader distributions using observed disagreement entropy. Single-reader examples have zero observed entropy; this does not prove diagnostic certainty.

The original 807-nodule and expanded 1,297-nodule experiments are separate cohorts. Every expanded run had lower validation Macro-F1 than its matched original run. On the expanded cohort, HARD averaged 68.99% validation Macro-F1 and LEGACY 66.27% across seeds 42/43/44. These are validation results, not independent test estimates.

Configuration D's 77.62% accuracy and 73.99% Macro-F1 were obtained after fitting ensemble weights and class offsets to the same 143 validation labels. They are exploratory development results. The earlier 183-nodule TEST cohort was historically evaluated. No result here supports a clinical operating point or a directly comparable literature ranking.

The implementation contains a Transformer, spatial attention and gated fusion; their isolated benefit has not been established by matched R50 component ablations. The extra reader head was not capacity controlled.

## Safe verification

From a clone, run:

```sh
python tools/verify_publication_snapshot.py
```

This standard-library-only command checks released bytes and saved confusion-matrix arithmetic. It does not import PyTorch, construct a model, read research images, download weights or run training/inference. Run from an unmodified clone and cite the exact Git commit used.

This is a source-and-aggregate-evidence release, not a one-command reproduction package. Patient-level images, membership/schedule records, qualified native execution records and weight files are not redistributed. See the reproducibility document before interpreting or executing historical scripts. Root package installation does not install or qualify the R50 study environment.

## Citation and use

Use GitHub's **Cite this repository** entry or `CITATION.cff`, plus an immutable commit permalink. No paper DOI or accepted publication is claimed. MIT applies to original project code only. LIDC-IDRI, ImageNet/RadImageNet weights, dependencies and third-party code retain their own terms. No weight redistribution or institutional ethics clearance is granted by this release.
