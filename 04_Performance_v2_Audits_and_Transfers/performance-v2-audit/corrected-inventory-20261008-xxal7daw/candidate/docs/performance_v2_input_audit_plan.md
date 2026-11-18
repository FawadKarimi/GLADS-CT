# Performance-v2 frozen TRAIN/VALIDATION input audit — proposed execution only

Recorded 2026-09-27. The standalone
[script](../scripts/55_audit_performance_v2_inputs.py) is prepared for review and
synthetic CPU testing only. No research inputs were decoded on desktop and no
university command was executed. Stage A evidence remains immutable, Stage B
paused, and `pilot_execution_authorized=false`. The accepted pilot matrix is
unchanged. The corresponding [official-weight plan](performance_v2_pretraining_acquisition_plan.md)
does not acquire or load weights.

## Exact source resolution

The immutable diagnostic archive is pinned to SHA-256
`e5663b68523b345a3b270108e6958e1f49498e3bec14ef6959e9f4d53394e36b`.
The script verifies its manifest/completion linkage and payload hashes. Within
that archive the authority for this read-only audit is full-model seed42
`phase4-final-s42-c1e650d77585ed9c`; all 18 exported sources were already shown
to contain the same train/validation membership. This choice is by fixed identity,
not performance, directory recency or a newly selected cohort.

It rehashes the live run manifest, completion, run identity, input lock and saved
supervision against the archive's source records. It checks live identity equality
and the recorded input-lock digest. Exactly 807 train nodules/437 patients and
143 validation nodules/84 patients must resolve; class counts remain 348/224/235
and 65/38/40. Every selected identity, label and reader count must agree with its
archived source row. Missing/extra/duplicate or inconsistent records fail closed.

The Phase 0 manifest checksum ledger and all six manifest text files use the
existing CRLF-to-LF-only hash convention; raw source bytes are separately captured
and must stay unchanged during execution. Patient and series mappings must match
each selected row, with no train/validation patient overlap.

Canonical/rendered checksum-ledger hashes and bundle hashes come from the frozen
run's `phase3_input_fingerprint`. Their completion bindings and canonical file-map
digests must agree. Only canonical/rendered CSVs and selected image members are
read, not the whole Phase 2 bundles. Nodule/patient/series/split/SHA identities,
resolution64, primary eligibility, label, valid source status and reader count
must agree. No eligibility or exclusion is recomputed from model results.

Expected paths are exclusively
`data/processed/consendxformer_phase2/rendered_64/resolution_64/<nodule_id>.npy`.
The saved absolute supervision path must match that exact ledger-derived path.
Symlink/reparse ancestors, escapes, aliases with TEST paths and duplicate image
paths fail. All 950 selected image hashes must match the rendered ledger before
the first decode. No image discovery/globbing or arbitrary CSV-path traversal is
used. The original server root is required; this script does not rewrite paths
for relocated data.

Mixed CSV and checksum bytes contain TEST metadata references. They are hashed
and inspected for bindings and path-alias rejection; no TEST image, mask,
prediction, checkpoint or DICOM is opened. This is a scoped audit, not a fresh
verification of every member of the original bundles.

## Decode and implementation identity

Production rendering stores NPY images. This audit supports that exact branch;
an unexpected other image format fails rather than silently changing decoder.
It uses `numpy.load(..., allow_pickle=False)` on the already hashed bytes,
requires an actual numeric 64x64 finite grayscale array in [0,255], casts to
float32, calls `cv2.resize(...,(64,64),INTER_LINEAR)` and divides by255 only when
the resized maximum exceeds1. Output must be finite float32 in [0,1]. The strict
input shape check prevents an incorrect-size image from being silently repaired
by the historically documented resize call.

The source text of frozen `phase4_dataset._read_image`, normalized to LF, must
retain SHA-256
`f47a12c5b999bbd657eb5f0ccfa1814655e3ba4627c9e3afcb11615552c0b595`.
The project module is parsed as text, never imported. Reports capture package
and runtime versions, NumPy/OpenCV entry-module bytes, primary decoder extension
binary hashes and the OpenCV build-information digest. These do not claim a
complete transitive native-library attestation. Decoder versions must match
owner-reported NumPy2.2.6/OpenCV5.0.0/wheel5.0.0.93. Torch/torchvision metadata
must remain2.3.0+cu121/0.18.0+cu121; neither is imported. The accepted synthetic
transform report is referenced by its exact hash, not rerun. The v2 resize to224
and augmentation are not applied to research images in this audit.

## Output and failure semantics

The output must be a new directory outside the repository. Before/after Git
state must remain identical, clean, on `phase4-consendxformer`; an older
documentation HEAD is reported without automatic synchronization. The report
records per-image raw hash, identity, shape, decoded dtype/range, finite status,
division-by255 flag, scaled range and scaled-array hash. Per-split totals include
division counts and dtype counts. No image pixels are exported.

Outputs are `report.json`, `input_ledger.json`, `source_hashes.json`,
`artifact_manifest.json` and `COMPLETED.json`. JSON uses sorted keys, compact
separators, UTF-8, no NaN plus one LF. The cohort-ledger digest uses canonical
JSON without a final LF. The artifact manifest hashes exact output bytes and
binds its files map; completion binds the raw manifest hash and grants no pilot
authorization. Source bytes and implementation binaries are rechecked before
completion. Read-before/read-after checks are not an atomic filesystem snapshot.

Missing/corrupt/hash-mismatched/nonfinite/inconsistent inputs, version drift or
warnings stop the audit. No repair, rerender, relabel, silent exclusion, alternate
format or package fallback is implemented. Caught failures preserve the report,
source hashes and decoded subset ledger when available, without a completion
marker. An uncatchable termination leaves its running report, not a passed one.

## Review and later command

The external review package supplies a byte-identical script, SHA-256, synthetic
validation report and a hash-checking `input-audit-server-command.sh`. The proposed
command, only after review/authorization, is:

```bash
bash /experiment-storage/chaidir/fawad-dsai/performance-v2-audit/input-audit-server-command.sh
```

The wrapper uses the existing immutable diagnostic ZIP, explicit env-310-fawad
Python, hidden CUDA and disabled bytecode, and creates a unique external output
directory. It must not be run as part of this task. If the expected external ZIP
location differs, confirm its path and bytes; do not silently choose another ZIP.
Successful completion would resolve only these selected input/decoder checks.
Weight provenance, overlap, topology/preprocessing and CUDA/pilot release remain
separate gates; RadImageNet remains blocked.
