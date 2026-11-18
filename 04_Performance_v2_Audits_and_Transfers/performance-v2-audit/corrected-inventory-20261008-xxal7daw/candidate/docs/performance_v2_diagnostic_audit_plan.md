# Performance-v2 diagnostic audit v1 — standalone read-only source extraction

The owner authorized this extraction, not GPU pilots. Stage A is preserved and
Stage B paused. The script is
[53_extract_performance_v2_diagnostics.py](../scripts/53_extract_performance_v2_diagnostics.py).
It uses Python standard library only, does not import the project or its completion
validators, and has no network/GPU/model/torch/YAML-evaluation path. No connection
to the university server is available in this desktop task; real extraction is
**not executed**. Synthetic tests use invented records only.

## Source contracts and limitations

The server repo is `/experiment-storage/chaidir/fawad-dsai/ConsenDxFormer`.
The script inventories one directory level under only:

- `artifacts/phase3/experiments`
- `artifacts/phase4/experiments`
- `artifacts/phase4/ablations/runs`

At most500 children per root and32MiB per allowlisted source file are permitted.
No recursive discovery, latest-directory selection or filename-only run selection.
Actual directory, receipt, manifest and content-addressed identity must agree.
Expected primary slots are B1/full/four OFF variants x seeds42/43/44. B0/other
seeds remain separately identified historical context. Synthetic, failed,
incomplete, tuning and incompatible candidates are not substituted for primary
sources. Multiple eligible candidates yield an explicit partial report with all
identities, hashes and configuration snapshots; do not select the best score.

Before/after Git reads use `git --no-optional-locks`, with fsmonitor disabled.
Actual branch/HEAD/dirty state are reported, not automatically repaired. An older
documentation commit need not be updated. There are no pull/reset/checkout calls.

The allowlist is exactly the brief's shared/Phase3/Phase4 paths: completion,
artifact manifest, history, run receipt, applicable identity/config/input-lock,
input metadata/supervision and `metrics/val.json`. Only those payload bytes are
read/hashed. Canonical manifest digest is SHA-256 of UTF-8 JSON sorted keys,
compact separators, `ensure_ascii=false`, no newline; completion binds the raw
manifest-file hash. Unknown schema, unsafe path, missing/mismatched binding,
nonfinite/ambiguous metrics and class ordering fail that source's diagnosis.
All path ancestors reject symlink/reparse traversal. Manifest entries outside
the allowlist are **not traversed**, even to verify their hashes. Thus recorded
checkpoint hashes remain references; `full_source_bundle_verified=false`.

Phase3 identity digest adds its schema_version=1 preimage. Phase4 full/OFF digest
uses the identity minus digest/experiment_id. All must bind the recorded run
identity; OFF also binds canonical effective configuration. Full Phase4 recomputes
the combined base/effective/input-lock/tuning-protocol digest from saved JSON;
Phase4 binds
canonical input-lock identity. B1 metadata binds raw CSV bytes. Frozen split pin
`0c983f0ebbc6dd70623b570c8a657aa14ccc550ac590b577648b3163bc8a01e3`, upstream
`0e5bcadcadfce5ea99df8484d23705d7bd4d778a`, and Phase4 cohort/resolution rules
are checked as applicable. Logical supervision/source fingerprints are archived
references, not recomputed by interpreting test rows. A full original-loader
validation would violate this task's no-tensor/test-read scope and is not claimed.

No source path found inside JSON is followed. In particular, never open anything
under checkpoints, predictions (including val), test metrics, image/mask or binary
array files. Metadata containers can contain all splits: their **whole bytes** are
hashed, then CSV rows streamed and split filtered **before** label/reader/identity
interpretation. Test rows are discarded, not copied or summarized. The original
mixed CSV is not exported. Say mixed containers were accessed, not “no test bytes.”

## Schema adapters and permitted diagnostics

- Histories use `{epoch,train,validation}`. B1's `f1_macro` and ordered per-class
  arrays map to normalized names; Phase4's `macro_f1` and named `per_class` map
  separately. Export original keys and adapter versions. Classes always remain
  benign/indeterminate/malignant; unknown/missing metrics block the affected source.
- Full Phase4 `metrics/val.json` has `evaluation_losses`; OFF has separate
  `prediction_metrics` / `within_variant_evaluation_losses`. Do not label nested
  total loss as plain diagnostic CE. Saved validation JSON is retained without
  generating metrics from predictions. B1 validation exports may lack loss; this
  is not repaired from predictions.
- B1 receipt `selection.best_epoch` and full receipt `best_epoch` are compared
  to first exact minimum historical validation loss. OFF has no verified receipt
  epoch: export null and `history_derived_criterion_only`. Also report earliest
  maximum stored validation macro-F1 epoch, its stored metrics, actual history
  length, initial/final/minimum loss and final validation-minus-training gap.
  This is counterfactual history inspection, not evaluation or replacement of an
  alternative checkpoint. Saved loss consistency uses explicit atol1e-8/rtol1e-6;
  auxiliary composition uses the historical atol1e-6/rtol1e-5 tolerance.
- Epoch-long output contains all stored required losses/class metrics plus weighted
  ordinal/attention contributions where bound lambda/component semantics permit.
  Training is online/augmented; validation deterministic. Gaps are not definitive
  overfitting, and total-loss scales across objectives are not commensurate.
- Train/validation identity records are retained with source CSV line numbers,
  exact patient/nodule/series/split/label joins, counts, reader counts, disjointness
  and expected train-derived balanced weights. No deduplication. Compare807/143
  nodules and437/84 patients; mismatch is a reported blocker, never filtered away.
  Verify stored weights when present. Missing B1 weights/hyperparameters remain
  missing; opaque YAML snapshots are included for later desktop parsing.
- TRAIN q/u only: finite [0,1], q sum atol1e-6/rtol0, no normalization. Export q_y,
  1-q_y, u*(1-q_y), hard label below maximum by >1e-8, maximal ties within1e-8,
  and unique agreement. Uniform targets remain ties. Summarize overall/by class
  and bins [0,.25), [.25,.5), [.5,.75), [.75,1]; empty means null with status empty.
  Proxies are target discordance, not gradients. No analogous test or validation
  conflict aggregate is computed. Missing supervision is explicit, not inferred.
- LR traces, task gradients and fusion-gate traces are `NOT_RECORDED`; configured
  scheduler/dropout is not a reconstructed time trace. Original JSON/YAML config
  snapshots support later review of maximum epochs, dropout and optimizer policy.

## Output, immutability and failure semantics

Use the external workspace
`/experiment-storage/chaidir/fawad-dsai/performance-v2-audit/`. The script creates
`audit-<UTC microsecond timestamp>-<input digest>/` exclusively and refuses path
overlap with the repo, symlink parents and existing destination. It also writes
an exclusive sibling ZIP. Source rechecks and unchanged Git state precede the
marker; changed inputs are fatal, with no successful marker. Failed/partial
attempts must be preserved. Read-before/read-after is not an atomic filesystem
snapshot or proof that historical execution declarations are true.

Outputs: `audit.json`, `run_inventory.csv`, `epoch_history_long.csv`,
`checkpoint_selection_diagnostics.csv`, `train_val_metadata_summary.json`,
`train_target_conflict_summary.csv`, `source_manifest.json`,
`missing_or_unverified.json`, `limitations.md`, `README.md`,
`artifact_manifest.json`, `AUDIT_COMPLETED.json`. Additional traceability exports
are `train_val_source_rows.csv` (filtered only) and `saved_validation_metrics.json`.
Config snapshots are in source_manifest; the mixed source CSV is never copied.

The audit records script hash/CLI/environment, original binding versus freshly
checked subset bytes, Git state, model_forwards=0, training_performed=false,
checkpoint_tensors_loaded=false, test_performance_computed=false,
development_diagnostics_computed=true and full_source_bundle_verified=false.
`audit_completed` means diagnostic export completed, not research certification.
Exit0 means completed, exit2 means partial/blocked, exit1 fatal. B1 opaque-YAML
interpretation remains a desktop-review item, so a first transfer can legitimately
be partial. Do not suppress its issue list to obtain a success label.

## Synthetic validation and handoff

Focused [tests](../tests/test_performance_v2_diagnostics.py) cover invented
B1/full/OFF records, nested loss schemas, first ties, per-class mapping, malformed
q/u, uniform ties and bin endpoints, duplicate nodules/patient overlap, mixed test
rows, corrupt manifests/identities, forbidden reads before open, input mutation,
reparse-path rejection, source immutability, deterministic table bytes, no-overwrite
and an18-source invented inventory. Git checks use a real temporary repository.
The Windows reparse check uses an explicitly labelled stat stub, not an assertion
that an actual server symlink was tested. No green repository-wide suite is claimed.

The external review folder supplies the standalone byte-identical script, its
SHA-256, upload instructions and `server-command.sh`. Upload to the external
workspace, not the repo. Run the supplied single command in a Jupyter terminal;
it uses `set -euo pipefail`, the explicit env-310-fawad Python, `-B`, hidden CUDA
and a script hash check before extraction. No SSH, `rg`, notebooks or installation
are required. Partial exits deliberately do not print a success message; return
their report/ZIP and issue list too. A fatal traceback/error without ZIP should
be returned rather than repaired by changing source evidence.

Next action is owner upload/run of the already authorized extraction and return
of its ZIP/report. Then review Gate B blockers before training implementation or
release. No new scientific approval for this read-only handoff is requested.
