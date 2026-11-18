# Performance-v2 lifecycle interfaces and training-mode synthetic review

Recorded 2026-10-07. This adds interfaces and tests to the reviewed hybrid at
`performance-v2`, HEAD `f8278c34d6145170345dce0a518276adf0bb4144`.
The hybrid, objectives, optimizer, selection implementation, v2 protocol,
historical configurations, input/security records and frozen-B1 policy remain
byte-identical. This document adds current implementation scope without
rewriting their historical statements. All 280 files present at the start of
this step are preserved; eight files are added, unstaged and unpublished.

Stage A accepted closure remains intact. Stage B and GPU pilots remain paused.
`pilot_execution_authorized=false`. A hash-bound plan/decision is an evidence
interface, not execution authorization. No B1 execution is requested or added.

## Implemented boundaries

| Interface | What it checks or prepares | What it does not establish |
|---|---|---|
| `bindings.FrozenInputs` | Raw lock SHA-256 before parsing; exact record fields, unique nodule identities, patient/series/split consistency, fixed labels/class order, finite q/u, source/supervision hashes, exact batch supervision; source bytes may be passed to an explicit digest checker | It has no file paths, decoder, research reader or TEST adapter. Trusted expected hashes and original supervision/image-byte authentication must come from a later approved source adapter |
| `metrics.DiagnosticAccumulator` | Identity-bound complete TRAIN/VALIDATION population; rows=true confusion matrix, first-class argmax ties, all-three-class Macro-F1 including absent classes, zero-denominator zero, per-class precision/recall/F1, secondary Accuracy=Micro-F1, unweighted diagnostic CE | No historical prediction reading, model evaluation launcher, uncertainty analysis or scientific superiority claim |
| `streams.TrainingStreams` | Arm-independent private shuffle and augmentation generators, exact batch16/drop_last=false/workers0 plan, flip/angle/translation trace, complete generator states | Plans do not transform images. The previously reviewed transform implementation and unresolved oblique-edge limit remain separate |
| `checkpoint` | Strict non-pickle typed JSON/tensor codec, bounded finite CPU states, model layout, optimizer groups/moments/update count, exact schedule, complete RNG and selection/patience reconstruction | No pretrained importer, `torch.load`, external state lookup, GPU transfer or automatic loading into a model |
| `lifecycle.ArtifactStore` | Exclusive new directory, flat names, bounded exclusive temporary writes, fsync, no-overwrite hard-link publication, fresh output rehashes | Private output-root ownership/ACL/mount safety and Linux crash durability remain deployment prerequisites, not proven by Windows tests |
| `lifecycle.RunLifecycle` | Exact plan and partial-batch accounting, sequential validation epochs, checkpoint/history retention, validation-selected best/last identities, completion manifest and final receipt, caught failure/cancellation guard | It never calls a model, optimizer or data reader. Caller observations must be authentic in any future production adapter |
| `lifecycle.BudgetLedger` | Every start/resume consumes a slot; failure has no refund; limit/plan/decision identities; trusted-hash restoration of the attempt ledger | No autonomous retries, cap increases, approval system, durable cross-process lock or GPU-hour estimate |

The new input-interface lock is **not** the original archived cohort ledger and
must not reuse its digest as the new lock digest. It carries that archived digest
as a separate field. Production scope additionally checks the preserved
807/143 nodule, 437/84 patient and [348,224,235]/[65,38,40] class counts, but matching
counts alone never authenticates membership. Independent review of the exact
new lock bytes and their mapping to original evidence is required. Synthetic
locks have distinct scope and are rejected as production cohort declarations.

Metrics accumulate confusion counts and sample CE, never equally weighted batch
means. CE uses float32 log-softmax and float64/Python summation divided by the
actual number of nodules. Train class weights and reader losses never enter
checkpoint CE. A partial split, duplicate ID, foreign split or label mismatch
cannot produce a selection record. Only a validation accumulator can create
`ValidationEpoch`; the preserved selector and independent raw patience rules
remain authoritative. No TEST interface was added.

## Checkpoints, resume and commit semantics

An execution specification contains the existing 13 scientific/source/input
bindings plus actual model tensor layout, optimizer ledger, input-interface
digest, fixed metric/selection definitions, schedule and loader policy. The
requested specification must match exactly on resume, including objective,
topology/initialization, transforms, runtime, RNG policy and every input binding.
The future adapter must build this specification from the actual effective
configuration and actual model/optimizer, rather than trusting a label alone.

Each saved epoch includes all model tensors (including BN buffers), AdamW
moments/step counts, scheduler position, CPU Torch/Python/NumPy RNG, private
shuffle/augmentation state and original stream seeds, validation history,
patience, selection, completed epoch/update counts and caller-supplied cumulative
elapsed time. Global RNG formats are validated with private generators before
restoration mutates any global state. The state transport rejects nonfinite,
unknown, oversized, malformed or noncanonical payloads and verifies SHA-256
before JSON parsing. It uses no pickle or generic object deserialization.

The current CPU interface requires all four groups and exact recipe-A rates,
decay, betas and eps. It rejects missing/unknown moments, alternate optimizer
flags and changed update counts. This strict interface may require a reviewed
adapter if a later approved runtime serializes a different schema; no silent
conversion or dependency substitution is supplied.

Resume is intentionally limited to an **interrupted completed-validation-epoch
boundary**. It requires a trusted terminal digest and all prior checkpoint
digests/bytes; histories must agree. All candidates are retained because the
global tolerance rule can select an earlier checkpoint. Reconstructing the
selection rules from history must reproduce every saved field. A restoration
callback targets a disposable candidate; complete live recapture must hash
identically before an admission object is returned. A changed/lost model,
optimizer, scheduler or RNG component rejects admission. No budget reservation
is made by the review function. A separately requested resume consumes another
attempt and creates a new exclusive directory, preserving original partial
evidence and retained checkpoint bytes.

Mid-epoch interruption, stale/partial checkpoint rollback, failed runs, completed
runs and hash/contract drift are not automatically resumable. Hard kills cannot
be caught: an outstanding reservation and no committed receipt remain partial
evidence. Mid-epoch exact recovery and crash reconciliation are future reviewed
work; never replay already consumed batches silently. Completion does not arise
from a caller-supplied status flag: the existing patience/max-epoch rule must
have stopped, every validation epoch/state must exist and outputs must freshly
rehash before the manifest and final receipt are published.

The atomic link of `completion.json` is the commit point. Cancellation or write
failure before it leaves failed/interrupted evidence and no completion receipt.
An interruption immediately after a successfully committed receipt is recognized
as committed rather than relabelled failed. A bounded temporary duplicate may
remain after such an interruption; it is retained as evidence. Receipt consumers
must reconcile directory/ledger state and authenticate the completion digest.
There is no automatic discovery/reuse daemon. Output-root substitution by an
adversarial same-user process and power-loss durability are not certified here.

Use `session.guard()` around future adapter operations: an uncaught exception
marks failure, cancellation marks interruption, and normal exit without
finalization also marks interruption. A full/unwritable disk may prevent a new
terminal JSON; the in-memory failure and already written bounded `.pending`
and completed artifacts remain, and no complete receipt is attempted. This
limitation is exercised and reported, not treated as successful retention.

Synthetic lifecycles emit `status=synthetic_complete`. Both the original reuse
validator and the new production completion validator reject these receipts.
Failed/interrupted executions have terminal records, never complete receipts.
The new reuse validator additionally requires production scope, the full
execution specification, manifest identity and reconstructed stopped selection.
Caller-supplied "verified hashes" must actually be freshly measured and the
trusted receipt digest must come from an independent ledger. Hash equality
alone is not approval or evidence that research execution occurred.

## Synthetic validation and retained evidence

**80 tests passed; zero failures, errors, skips or warnings.** This comprises
43 preserved component/contract tests and 37 added tests. Subcases cover every
resume binding, multiple rejection paths and all five objective variants.

- Every BN layer updates running state/counter in train mode and remains
  byte-identical during eval. All BN affine parameters stay trainable.
- Combined HARD, LEGACY, SHARED-010, SEPARATE-010 and explicitly enabled synthetic
  CAPACITY-010 losses connect all intended parameters; gradients are finite and
  nonzero in every intended backbone/shared/head scope. No optimizer steps occur.
- SEPARATE reader loss has no direct diagnostic-head gradient, and its combined
  diagnostic-head gradient equals HARD alone. Reader gradients still reach
  shared features and can affect future diagnostic predictions indirectly.
  SHARED-010 intentionally applies its reader term to the diagnostic output.
- Extra-head construction preserves all common initial tensors, process CPU RNG,
  live shuffle order, batches and augmentation traces across multiple plans.
- Synthetic singleton and seven-item tails pass in train mode. Plans of 17/33
  invented nodules retain their final item. Identity/metric aggregation remains
  correct under uneven batch partitioning.
- Invented model tensors and mock optimizer counters exercise 20 synthetic
  **lifecycle record epochs**, not 20 model-training epochs. There are no
  optimizer updates or production data loops in those tests. The codec roundtrip
  uses newly created synthetic state only, never an existing checkpoint.
- Tests retain corruption, pre-commit cancellation, failed checkpoint writes,
  mid-epoch/boundary interruption, guarded early exit/numeric failure and
  post-commit interruption evidence outside the checkout. Synthetic completion
  artifacts are also retained to show rejection by production reuse.

The external harness denies network and test subprocess operations, research
and existing checkpoint suffix reads, weight downloading/loading, compilation,
and real AdamW steps. New synthetic `.state.json` files are the only state
artifacts exercised. The initial real Windows host metadata observation happens
before subprocess denial; it does not impersonate a different runtime. Exact
source/configuration hashes are captured before/after tests in TEST-RESULTS.json.
The initial 75-test result/evidence remains retained alongside the final result.

## Runtime and exact remaining blockers

Observed local validation: Windows AMD64, CPython **3.14.6**, torch **2.13.0+cpu**,
torchvision **0.28.0+cpu**, NumPy **2.5.2**, two CPU threads. The interpreter is
`C:/Python314/python.exe`, SHA-256
`03168c01b7b7491423350e82c26fee71f35b43694d1319d3c668bda6903a0c38`.
Standard-library unittest is used because pytest is absent. No package changes
were made. The bundled Python3.12.14 has no torch/torchvision/pytest; the listed
Windows Store Python3.9 returns WinError5 Access denied. No full-repository
pytest result is claimed.

**Offline Linux validation is not presently available/verified from this task.**
`wsl.exe --list --quiet` returns exit4294967295 with
`Wsl/EnumerateDistros/Service/E_ACCESSDENIED`. Docker and Podman are absent from
PATH. No Linux dependency inventory could be verified, and Windows wheel success
does not establish Linux compatibility. No distribution/container was started,
no installation or alternate dependency set was substituted, and no workflow
was published or dispatched. Portable Python3.9 grammar checking is static only.

Before any production integration/execution:

1. Review these interfaces, exact source hashes and the codec/receipt trust
   boundaries. Authenticate the effective settings from actual model/optimizer
   objects and establish trusted plan/decision/receipt ledgers. The current
   budget implementation has one process owner; durable multi-process admission,
   power-loss recovery and original-output receipt reconciliation remain open.
2. Supply an approved source/supervision adapter, byte authentication and new
   interface-lock mapping to frozen evidence. Integrate the accepted transforms
   without changing membership, labels or RNG semantics. This is not a request
   to reopen Stage A or repeat an accepted historical audit.
3. Validate CPU native Linux codec, hard-link/no-overwrite/fsync, filesystem
   ownership/ACL/mount behavior and resource bounds using the separately approved
   runtime. Complete transitive/native-image/security and full Docker acceptance.
4. Integrate an explicitly authorized runner only after review: actual batch
   operations, timings (including failed work), objective component logging,
   optimizer state capture, checkpoint restoration and ledger admission must be
   verified end to end. This preparation has no trainer, scheduler launcher,
   remote job, production loader or GPU path. Maximum100 epoch-state retention
   has a per-file512MiB cap; actual storage/time quotas require measurement and
   independent approval. No GPU-hour estimate is inferred.
5. Real pretrained acquisition/loading remains blocked pending existing terms,
   provenance, all-split overlap and strict semantic mapping gates; RadImageNet
   remains blocked. CPU synthetic states cannot validate pretrained weights.
6. CUDA RNG, device restoration, determinism, memory feasibility and GPU runtime
   acceptance remain unimplemented/unverified. A CPU checkpoint cannot silently
   resume on GPU. Mid-epoch recovery requires a separately reviewed extension.

The 9–10 core-run plan, conditional three-run capacity extension and zero
automatic B1 retraining policy remain unchanged. No training, research
evaluation, original checkpoint reading, package acquisition, publication,
workflow dispatch, GPU or server job occurred in this step.
