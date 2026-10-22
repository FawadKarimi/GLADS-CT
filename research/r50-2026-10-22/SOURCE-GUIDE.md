# R50 implementation guide

We organize the implementation by experimental stage so readers can follow the exact model, input and objective definitions used in our study.

## Original-cohort source

We preserve 132 source files and 21 invented-data test files under `historical-replication/`, byte matched to our original 167-entry source lock. The remaining 14 configuration records are site-specific and remain outside this public release. The unchanged [original lock](provenance/replication-source-lock.original.json) and [provenance record](provenance/SOURCE-PROVENANCE.json) document this scope.

Suggested reading order:

1. [Model and feature fusion](historical-replication/src/consendxformer/performance_v2/model.py)
2. [Input decoding and augmentation](historical-replication/src/consendxformer/performance_v2/input_adapter.py)
3. [Objective definitions](historical-replication/src/consendxformer/performance_v2/objectives.py)
4. [Optimizer and schedule](historical-replication/src/consendxformer/performance_v2/optimization.py)
5. [Training lifecycle](historical-replication/src/consendxformer/performance_v2/training_runner.py)
6. [Validation selection](historical-replication/src/consendxformer/performance_v2/selection.py)
7. [RadImageNet initialization](historical-replication/src/consendxformer/performance_v2/radimagenet_initialization.py)

We retain the test sources for inspection; the standard-library verifier at the repository root checks public hashes and aggregate arithmetic without executing these model tests.

## Expanded-cohort source

We provide the recovered [TRAIN1297 runner](recovered-expanded/run_train1297_six_runs.py), [ensemble exploration script](recovered-expanded/ultra_optimize_logits.py) and associated dependency snapshot separately. The model, objective, input, optimizer and training-runner modules match their historical counterparts. Four guard/storage modules differ, as enumerated in the provenance record; the complete expanded execution-time source seal is not established by this release.

We preserve historical source bytes, absolute paths and inactive defaults rather than silently rewriting them as portable training examples. Read these scripts alongside the [scientific protocol](protocols/SCIENTIFIC-PROTOCOL.json) and [reproducibility guide](REPRODUCIBILITY.md) before considering a separately prepared execution environment.

## Results and source identity

Our `results/` directory separates original initialization, original replication, expanded replication and exploratory validation ensembles. We include per-seed values, class metrics and checkpoint identities where retained. The [public ledger](SHA256-LEDGER.json) checks the current source/evidence/documentation snapshot; the original research source lock remains unchanged.
