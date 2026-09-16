# Phase 5 insertion/deletion subset: implementation for review

Scope is `insertion_deletion_only`, implementing section 3 and applicable source
contracts of the unchanged `phase5_supplemental_evaluation_proposal.md` (LF
SHA-256 `4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620`).
This milestone authorizes implementation and synthetic CPU tests only. It does
not authorize research execution, server work or publication. The separate
versioned specification is `configs/phase5_insertion_deletion_v1.json`; existing
protocols, architecture, selected checkpoints and completed bundles are unchanged.

All proposed results remain exploratory post-hoc, fixed after test exposure,
not preregistered or confirmatory. Macro-F1 remains primary. This subset computes
no pointing, accuracy intervals, boundary, uncertainty, patient-weighted
correctness, calibration or hypothesis tests. All 15 formal hypothesis IDs remain
`unresolved_not_computed`, with no p-values, placeholders or significance labels.
The full supplement and Stage A remain incomplete.

## Separate execution gate and protected sources

`scripts/49_run_phase5_insertion_deletion.py` is a separate opt-in entry point.
It rejects missing `--authorize-execution` before source loading/model preparation.
Future execution requires a clean committed `phase4-consendxformer` checkout,
the exact new protocol and unchanged reviewed proposal, normal dependencies and
the actual pinned external B1 source. This document does not grant that future
authorization; code review and the later timing review are separate gates.

Explicit future options (not executed here):

- `--full-source DIR` three times, exactly one complete source at each seed.
- `--b1-source DIR` three times, exactly one B1 source at each seed.
- `--attention-source DIR` for `phase4-evaluation-22cb656eabd040c2` and
  `--attention-manifest-sha256 HASH` for its explicitly approved manifest.
- `--authorize-execution`, after separate authorization.
- `--device cpu` (default) or explicitly authorized `cuda`. Deterministic CUDA
  support requires the existing runtime prerequisites such as
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` before interpreter startup; unsupported
  deterministic operations fail, without tolerance changes or fallbacks.
- Optional `--output-root DIR`, contained in the new namespace below.

There is no discovery, alternate checkpoint, source allowlist extension, reduced
cohort, ablation model or research retry option. The existing distinct complete/B1
loaders validate all six saved sources and exact frozen identities. They may
check saved metrics as part of their unchanged source-validity contracts; no
earlier evaluation routine or result bundle is rerun. The matched
archive's completion, checksums, array metadata, interpretation permission and
all six source bindings must agree, including original saved prediction replay.
The existing prior is checked against the same 807 training identities, disjoint
from test patients; it is never refitted. All 183 nodules from 99 patients remain.

Only persisted validation-selected best checkpoints are used, through the existing
complete-model loader and actual pinned B1 constructor/checkpoint loader. Before
any forward, existing input-lock/readiness guards are revalidated and unaugmented
live CT/mask/identity batches are compared exactly with the archived inputs.
Masks are validation evidence only; neither the curve engine nor any predictor
accepts lesion masks. Source/input hashes and in-memory frozen model state are
checked again before completion. Test stubs do not weaken these production guards.

## Preserved numerical and scheduling contract

Images are unchanged float32 `[183,1,64,64]` values in `[0,1]`; maps are the saved
native `[183,1,8,8]` DSA and the one `[8,8]` training-only prior. Both directions
use the float32 zero baseline in normalized input coordinates, with no blur,
baseline fitting, saliency generation, map interpolation or target-mask changes.

Each native cell controls its entire nonoverlapping 8-by-8 input tile. Every
ordering includes all 64 cells. DSA/prior orders sort raw values descending,
then flat index ascending for exact ties. The non-saliency order sorts SHA-256
digest bytes of the exact proposal JSON, then flat index. The stored digests
are indexed by original flat cell index; permutations give the removal/addition
order. Controls are identical across models/seeds for a nodule. Rankings stay
fixed from unperturbed saved evidence, regardless of any perturbed attention.

All 1,098 unperturbed batch-one replays across both models and seeds 42/43/44
must satisfy the existing per-class `1e-5 + 1e-4*abs(saved)` logit/probability
tolerance and saved-class argmax agreement. A late failure prohibits every
baseline/interior forward. The global endpoint cache is sealed to the complete
source/checkpoint/input identity, units, orders and raw output hashes and read
back before the barrier is granted. The target remains the saved predicted class,
including originally wrong predictions, not the true class or a changing argmax.

Every fresh probability output must be nonempty float32 `[N,3]`, finite, in
`[0,1]`. A private float64 copy is summed with `atol=1e-6, rtol=0`. The raw values
are not clipped, normalized or passed through the stricter shared metric guard.
Logits must be finite float32 as emitted by the frozen no-autocast runtime.
Raw probabilities, row sums and signed/absolute deviations are exported. Float64
promotion for CSV precision and trapezoidal AUC arithmetic changes no raw values.

There are 65 positions at `k/64` per curve. No curve is truncated, made monotone
or normalized by its endpoints. Cache only each unit's original and baseline
outputs, shared by its six curves. Every interior position is evaluated, even
for zero images or identical orderings; no cross-unit/order deduplication occurs.

- 1,098 original replays, 1,098 baseline calls and 415,044 interior calls.
- Exactly 417,240 batch-one forwards, 139,080 per trained seed, with no retries.
- 6,588 curves and 428,220 exported positions, including cached endpoints.
- 36 model/seed/order/direction summaries and 24 paired within-model summaries.

The implementation schedules model, seed, then archive row for the global replay;
after the barrier, unit, ordering, direction, then interior step. All six frozen
models are retained for this schedule. Preparation/loading makes no extra forward.
Actual attempted/accepted counts and failed attempts are recorded, not inferred
solely from the planned budget.

Per-seed summaries report arithmetic nodule means, median and linear quartiles
of AUC across all 183 nodules, and the 65 mean-curve positions. Paired differences
are DSA-minus-prior and DSA-minus-hash-order within each model/seed/direction.
Constant/baseline-identical images, flat/equal-order and nonmonotonic curves are
retained and flagged. There is no cross-model AUC hypothesis or patient/seed
independence claim.

## Exports, verification and failure semantics

New outputs use
`artifacts/phase5/insertion_deletion/phase5-perturbation-<identity hash>/`.
Identity distinguishes the proposal's base and publication commits from the
clean implementation/execution checkout commit (the latter two are equal for
this runner). It binds protocol/proposal, exact sources and
snapshot hashes, attention manifest, runtime/device and frozen-loading provenance.
Reject existing complete or partial paths, source/data overlaps, escaped or
symlink paths and completed ancestors. This namespace is the only new ignore rule.

The bundle includes:

- `protocol.json`, `provenance/reviewed_proposal.md`, `evaluation.json`,
  `summary.md` and `provenance/provenance.json`.
- `tables/units.csv`: exact identity/order, fixed target, original image, DSA,
  prior, baseline, source-record and best-checkpoint hashes; constant/zero flags.
  Curve rows reference this table's unique `unit_id` for their immutable source
  and map provenance rather than copying or overwriting original maps/CT tensors.
- `orders.npz`: all permutations and all unsigned hash-control digest bytes.
- `replay_cache.npz`, `replay_barrier.json`, `replay_checks.json`:
  globally validated original outputs, source/cache binding and all discrepancies.
- `baseline_cache.npz`, `curves.npz`: exact float32 logits/probabilities,
  scheduled call indices and hashes of actual forward inputs. Only endpoints
  reuse call indices; interior calls are individually accounted for.
- `tables/perturbation_curves.csv`: identities, fixed target, order, direction,
  step/fraction/changed tile, raw three-class logits/probabilities, current
  predicted class, fixed-target response, mass diagnostics and endpoint-cache
  references. Step zero uses changed-tile index `-1` (no change yet).
- `tables/perturbation_auc_per_nodule.csv`, `perturbation_per_seed.csv`,
  `perturbation_paired_per_nodule.csv`, `perturbation_paired_descriptive.csv`,
  `perturbation_mean_curves.csv`, and the unresolved `hypothesis_register.csv`.
- `call_ledger.json`, an artifact manifest and final `COMPLETED.json`.

NPZs use the existing deterministic ZIP/array hash machinery and forbid object
arrays/pickle. CSV numbers round-trip from promoted raw values. Before completion,
inference-free read-back checks every array hash, all steps, fixed targets,
endpoint reuse, call indices, cohorts, controls and exported tables. AUC/summary
replay uses float64 `atol=1e-12, rtol=0`; raw curve values are exact. Completion
binds every payload and certifies only `insertion_deletion_only`. It cannot close
the formal hypothesis register, the full supplement or Stage A.

Failures, including catchable Python interruptions (`KeyboardInterrupt`,
`SystemExit` and `GeneratorExit`), retain the new partial directory and attempt
to persist `FAILED.json` with actual attempted/accepted counts, current
stage/unit/step context and replay diagnostics, then re-raise the original
exception. An interrupted forward is not accepted as a validated output; no
further forwards or automatic retries/resume occur. Only the directory created
by this attempt is eligible for failure recording and completion-marker removal;
existing attempts and previous research successes/failures are never edited.

Failure recording and removal of this attempt's completion marker are independent
best-effort actions, including when final validation is interrupted. Their errors
must not replace the original exception. A recorded failed attempt is rejected
by completion validation even if the filesystem prevents marker removal. No
overwriting or reuse of partial/failed output directories is allowed. SIGKILL,
power loss, repeated interruption during cleanup or an unwritable filesystem
cannot guarantee persisted failure evidence or marker removal; a leftover marker
under those conditions is not proof of successful execution and requires review.

## Synthetic validation and later timing procedure

New tests use invented images/identities and tiny deterministic image-only
predictors, not research model checkpoints. They exercise hand AUCs, mapping,
exact ties, independently generated hash-order golden vectors, changing predicted
classes, raw float32 edge cases, all-zero/nonmonotonic inputs, fixed rankings,
global late-replay failures, source corruption/mutation, deterministic exports,
complete call/cache accounting and subset-only completion. A full synthetic
183-nodule schedule exercises the 417,240 calls; it is not a research result or
a timing measurement. Historical source-loader fixtures are explicitly stubbed;
the existing source-contract tests remain part of the full CPU suite.

Research runtime and feasibility are unmeasured. The following is a proposed
synthetic timing procedure for later review and separate authorization, not a
benchmark run or permission to execute during this milestone:

1. On the intended device, record hardware, package/source hashes, shapes,
   thread settings, memory and deterministic/no-autocast policy. Use the actual
   frozen architectures/configurations but synthetic weights/inputs only, with
   no research checkpoint, prediction, label or mask access. Identify clearly
   that synthetic weights do not validate research replay or accuracy.
2. For each architecture and seed 42/43/44, initialize synthetic weights once
   with that seed before timing. Use batch-one float32 `[1,1,64,64]` synthetic
   zero, linear-ramp and `Generator(PCG64(20260917))` uniform random images in
   `[0,1]`, plus the same tile-construction path. Cycle zero/ramp/random in that
   order; use identical input sequences across the six synthetic models. Fix
   this sequence before timing. Propose 32 warm-up
   calls and three measured blocks of 128 calls per model/seed: 2,496 calls
   including warm-up across six synthetic models, separate from the research
   budget. Report all blocks, not a fastest selected block. Review these timing
   choices before using them; they do not modify the evaluation protocol.
3. Synchronize CUDA around timed blocks where applicable; report warm-up and
   measured-call counts separately, block seconds, time per call and peak
   memory. Measure input construction, model execution and export/read-back
   overhead separately rather than treating model-only time as end-to-end time.
4. Extrapolate per-model/seed measurements to its fixed 69,540 scheduled calls,
   then sum all six; report the observed block range and representativeness
   limits. Loading, replay validation, disk I/O, six simultaneously resident
   models and synthetic-versus-research differences limit that extrapolation.
   No wall-clock feasibility or performance claim exists until this is reviewed
   and measured. Any later batching/caching optimization requires separate
   numerical validation and review; never alter the fixed scientific schedule.

Zero baseline is an input-space intervention, not a physically exact air image.
Black tiles can be out of distribution. High insertion/low deletion AUC describes
response of a fixed predicted class only, not clinical causality, causal
faithfulness, attention-loss contribution or diagnostic superiority. Complete
and B1 can target different classes. Do not choose a replacement model, baseline,
ordering, threshold or seed from perturbation results.
