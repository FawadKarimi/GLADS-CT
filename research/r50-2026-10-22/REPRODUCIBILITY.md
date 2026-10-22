# Reproducibility and scope

## What is byte verified

All 167 original replication bindings matched the retained candidate locally. This release publishes its 132 source and 21 test files unchanged. The original lock has SHA-256 `d0deb9fb880b8a71e0364e26c5e87d8962b15d57c21bbe3cdcfff9f7c56fc51f`; the remaining 14 configuration records are private/site-specific and not included. Hash agreement establishes byte identity, not publisher authentication, scientific validity or fresh native compatibility.

The recovered expanded runner matches a second retained server copy (SHA-256 `08664330728a5c95e227b5c9e23d62bb68bc331dd387beeab8b3ddf93c7cf902`). Its recovered dependency snapshot differs in `durable_budget.py`, `durable_lifecycle.py`, `full_supervisor.py` and `storage_bound.py`. The scientific model/objective/input/optimizer/runner modules match the replication versions. No complete expanded execution-time source seal has been established; reconstruction is not proof of historical execution.

Source comments saying "preparation" or "blocked" are retained historical context. The constructor is RANDOM-only by design; the separately verified import path copies approved backbone tensors. Checkpoint resume and fresh pretrained import are separate operations. Do not edit stale comments and then describe the edited bytes as historical evidence.

## Data and input bindings

LIDC-IDRI was partitioned by patient before the original study. Original TRAIN has 807 nodules from 437 patients; expanded TRAIN has 1,297 nodules from 529 TRAIN patients. Validation remains 143 nodules/84 patients and historical TEST 183 nodules/99 patients. Expanded TRAIN has 4,273 eligible axial views, but one view per nodule per epoch is sampled under matched frozen schedules, not all views per batch or volumetric inference.

Patient-level membership, labels, geometry crosswalks, per-file image hashes, materialized crop manifests and seed-specific schedules remain controlled artifacts and are not included in this public source release. Thus this snapshot alone cannot reproduce the exact data inputs. Obtain the original de-identified collection and an appropriately approved manifest release before making an exact reproduction claim. Do not use historical Phase 0 split files at the repository root as the paper's authenticated split.

Relevant retained expanded bindings from the runner are: materialized view manifest `073c7ac48ee2f0fd96754f4c8647d677e83b6d3d1e0e69c42f590980293a2cad`; crop manifest `d77509a311083c8d7345266be6cce90dc3c399d76eb7e202aa9ef0e333ab673e`; frozen validation records `766ccd07c75a0ff2b0833867203cc57b146f9c3ac9ae52f07500d605298ad8a9`. Seed-specific schedule digests remain literal constants in the recovered runner. Publishing their hashes does not publish or freshly authenticate the original data bytes.

## Environment and weights

The recorded research runtime is Linux/Python 3.10.17, PyTorch 2.3.0+cu121 and torchvision 0.18.0+cu121 on a shared RTX 2080 Ti. Root Phase 0/1 requirements are not that runtime's qualified lock. No environment was installed or qualified for this release. The private runtime admission records and hardware/resource permissions cannot be transferred as permissions to a new user or host.

Official ImageNet and RadImageNet artifacts were acquired and transferred through restricted CPU/non-pickle paths. The published import code preserves tensor origin/value checks, independent RNG streams and the 318-entry backbone mapping. RadImageNet supplies all 53 BatchNorm counters; none are synthesized/reset. The distinct legacy ImageNet adapter must not be applied to RadImageNet. No raw or exported pretrained weights or trained checkpoints are redistributed. Weight-specific terms and full-digest publisher authentication remain limited as described in the third-party notice.

## Evaluation and remaining limits

The earlier TEST set was historically exposed. Expanded and ensemble results are validation results; repeated validation development and ensemble search introduce selection optimism. Three matched seeds provide descriptive variability, not reliable population significance. Expansion simultaneously changes membership, annotation support, class weights, updates and views. Common ImageNet normalization differs from RadImageNet's recommended recipe; pretraining overlap with LIDC-IDRI is unknown. Gate/Transformer benefit, capacity-independent reader benefit, calibration and clinical efficacy are unestablished.

Source/configuration hashes, selected checkpoint hashes, stopping epochs and per-seed values are retained where available in `results/`. These are aggregate records, not replacements for missing original per-case logits or complete epoch histories. Checkpoint bodies, immutable native launch receipts and some expanded execution-time controls remain outside this public release. The safe verifier validates released bytes and saved arithmetic only, not model-to-prediction authenticity or all reported CE values.

## Safe onboarding

Clone, record `git rev-parse HEAD`, and run `python tools/verify_publication_snapshot.py`. It imports only the Python standard library and writes no research data. Inspect sources without importing the package. Any future training/inference requires separately qualified runtime, lawful artifacts, authenticated data/schedules, resources and fresh approvals; nothing in this release authorizes execution.
