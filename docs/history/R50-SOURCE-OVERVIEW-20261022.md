# R50 source and aggregate evidence snapshot

This directory separates an authenticated historical source candidate from a later recovered expansion runner. It preserves supplied source bytes; public packaging did not retrain or evaluate models.

- `historical-replication/src/`: all 132 source files bound by the retained 167-file replication lock, copied byte for byte.
- `historical-replication/tests/`: 21 retained invented-data test sources; not rerun for this release. Some construct or execute models and are not part of the safe verifier.
- `provenance/replication-source-lock.original.json`: unchanged 167-entry historical lock. Its 14 private/site-specific configuration records are deliberately not redistributed. The public release is not represented as the complete private admission closure.
- `recovered-expanded/`: recovered TRAIN1297 runner, post-hoc ensemble search and dependency snapshot. Four guard/storage modules differ from the replication candidate. The full expanded execution-time source seal remains unverified. Do not relabel this as the original sealed replication closure.
- `protocols/`: descriptive study contract, not an executable authorization or replacement for native specifications.
- `results/`: copied aggregate CSV records and explicitly identified aggregate extracts from retained records. No new predictions or research metrics were generated for this release.

Start with [reproducibility](REPRODUCIBILITY.md), [results](RESULTS.md) and the [source provenance](provenance/SOURCE-PROVENANCE.json). The scripts retain historical absolute paths, environment checks and inactive defaults. Do not execute them as portable examples, replay expired authorizations, or bypass admission controls.
