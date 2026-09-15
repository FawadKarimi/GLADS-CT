# Guarded execution of the four existing component ablations

This milestone implements execution and CPU fixture tests only. No university
training, real-data evaluation, GPU job, new tuning search or artifact replacement
is performed by implementation, import, or planning.

The four specifications in `phase4_ablations.py` and the historical
`ablation_protocol.json` remain unchanged, including
`training_execution_authorized=False`. A separate versioned execution protocol
and runtime authorization govern the new runner. Component evaluations take place
**after** complete-model test results and the supplementary attention comparison
were seen. This is not a claim of preregistration. Neither set of results may be
used to redesign components, select weights/thresholds, or optimize these ablations.

## Frozen controls and exact semantics

The complete-model training reference stays
`e1b174490245c616e355ef7c0c6f7c3c239d3acd`. New identities record the actual ablation
code commit separately, rather than claiming that the new runs were trained at
the historical commit. The canonical configuration and B1 anchor must reproduce
the already-frozen combined configuration hash. No YAML, loss-weight, epoch-budget,
augmentation or model overrides are accepted by the CLI.

- `disagreement_weighting`: switch only `use_disagreement_weighting` off. The
  diagnostic term becomes weighted hard cross-entropy; independently enabled
  ordinal and attention auxiliary contributions remain.
- `ordinal_auxiliary`: switch only `use_ordinal_head` off under its existing
  **loss-ablation** semantics. The ordinal head remains in the architecture;
  only its auxiliary contribution is removed. Lambda `0.5` remains provenance.
- `attention_alignment`: switch only `use_attention_alignment` off. The DSA
  attention mechanism remains enabled and emits its native attention maps. Only
  the alignment-supervision loss contribution is removed; lambda `0.05` remains
  provenance. Disabling DSA is not an equivalent experiment.
- `hierarchical_fusion`: switch the ablation and model hierarchical-fusion
  controls off together. Execution uses the existing `deep_fused` path without
  the already-defined C2/C1 hierarchical stages; no new architecture is introduced.

Exactly one declared component changes per variant. All-off is neither an
authorized variant nor a substitute for B1. The twelve planned runs are four
variants times seeds 42, 43 and 44, not a hyperparameter grid.

Optimizer, scheduler, epoch budget, batch size, preprocessing, paired augmentation,
class weighting, clipping, determinism/warn-only policy and checkpoint selection
are inherited unchanged. Class weights use training hard labels only. Every run
initializes fresh seeded weights; complete-model checkpoints are reference
artifacts, never fine-tuning initializers. The primary 64x64 cohort remains 1133
nodules, train/validation/test 807/143/183, with 99 test patients.

## Read-only plan

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python scripts/45_run_phase4_ablations.py plan
```

This prints all twelve exact specifications, with the three
`attention_alignment`-OFF seeds first. It needs no university files and does not
construct datasets/models/loaders, create run directories, discover GPUs or launch
processes. A plan is not execution authorization or proof of input readiness.

## Future authorized first group: explicit GPU assignments

The following commands are **instructions for a later authorized university run**,
not actions performed by this implementation milestone. First obtain explicit GPU
allocations from the user/operator. Set `ABLATION_GPU42`, `ABLATION_GPU43` and
`ABLATION_GPU44` to three distinct full physical GPU UUIDs, each in the format
`GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`. A read-only UUID listing, if useful,
is `nvidia-smi --query-gpu=index,uuid --format=csv`; it is not proof a device is free.

The commands deliberately fail if an assignment variable is missing. Numeric
indices, implicit GPU 0, comma-separated device lists and duplicate UUIDs are
rejected. Each child sees only its assigned physical GPU through
`CUDA_VISIBLE_DEVICES`; its logical CUDA device does not imply physical GPU 0.

Preferred batch interface, after allocation and explicit authorization:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/45_run_phase4_ablations.py launch \
  --job "attention_alignment:42:${ABLATION_GPU42:?Assign the GPU UUID for seed 42}" \
  --job "attention_alignment:43:${ABLATION_GPU43:?Assign the GPU UUID for seed 43}" \
  --job "attention_alignment:44:${ABLATION_GPU44:?Assign the GPU UUID for seed 44}" \
  --full-source-root artifacts/phase4/experiments \
  --authorize-training
```

Equivalent explicit single-job commands (run in independently assigned sessions
if concurrent execution is desired):

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/45_run_phase4_ablations.py run \
  --variant attention_alignment --seed 42 \
  --full-source artifacts/phase4/experiments/phase4-final-s42-c1e650d77585ed9c \
  --gpu "${ABLATION_GPU42:?Assign the GPU UUID for seed 42}" --authorize-training

PYTHONDONTWRITEBYTECODE=1 python scripts/45_run_phase4_ablations.py run \
  --variant attention_alignment --seed 43 \
  --full-source artifacts/phase4/experiments/phase4-final-s43-946220804796df93 \
  --gpu "${ABLATION_GPU43:?Assign the GPU UUID for seed 43}" --authorize-training

PYTHONDONTWRITEBYTECODE=1 python scripts/45_run_phase4_ablations.py run \
  --variant attention_alignment --seed 44 \
  --full-source artifacts/phase4/experiments/phase4-final-s44-8c74c3dd1e666bb5 \
  --gpu "${ABLATION_GPU44:?Assign the GPU UUID for seed 44}" --authorize-training
```

The launcher preflights the **whole explicitly requested group** before spawning
anything. It starts one independent process per assigned GPU with separate
combined stdout/stderr logs, per-job exit JSON and a group exit-code summary under
`artifacts/phase4/ablations/launches/group-<hash>`. It propagates any nonzero child
exit as a nonzero group exit and waits for other normally running children.
Spawn failure or interruption terminates only children started by that launcher.
It does not discover free GPUs, schedule unspecified jobs, retry failures, or reuse
completed identities. Bytecode writing is disabled before external-source imports.

## Trust boundaries and artifacts

The dedicated runner validates the tracked configuration, frozen Phase 0 manifests,
Phase 2/3/4 readiness and input lock before constructing any dataset, loader or
model. Its paired complete-model source must pass the existing strict historical
loader and match the exact seed and input fingerprint. It seals reference files
and all image/mask inputs, verifies the external baseline checkout, and rechecks
readiness, source files and clean branch/commit state after execution.

Each new run has a separate content-addressed identity:

`artifacts/phase4/ablations/runs/phase4-ablation-<variant>-s<seed>-<hash>`

The identity binds run kind, variant, seed, toggles, frozen lambdas, effective
configuration, input hashes, the ablation code commit, historical full-model
commit and the verified same-seed complete-model reference/checkpoint hash.
Effective configurations are validated copies; original files are never edited.
The namespace cannot be redirected through symlinks or nested inside completed
artifacts. Existing final, tuning, evaluation and ablation destinations are protected.

Within each variant/run, best checkpoint selection uses minimum validation total
loss under that variant's remaining authorized terms, with the existing strict
improvement/early-stopping behavior. Differently scaled total validation losses
must not be compared across variants. No test dataset/loader is constructed until
best selection and persisted best/last validation have completed and best weights
have been reloaded. Final inference checks every ordered sample identity, hard
label and input path.

Persisted loss reports must contain the authorized composition, with disabled
auxiliary terms exactly zero. The fixed accounting check uses `atol=1e-6` and
`rtol=1e-5` solely for float32 accumulation rounding; it changes no training loss,
selection rule, attention threshold or evaluation metric.

The separate ablation checkpoint schema stores model, optimizer, scheduler,
epoch/best epoch, losses, class order and full ablation identity. Both persisted
files are loaded and reconciled with history after training; returned records
come from those snapshots, not mutable `state_dict()` references. The runner reloads
them again for sidecars and final artifact validation. Loading best weights cannot
turn the retained last snapshot into the best snapshot.

Artifacts include configuration/input/protocol snapshots; before/after readiness;
Phase 0 manifest provenance; RNG, environment, device and source-state evidence;
history; complete best/last checkpoints and metadata; exact validation/test CSVs;
and classification/calibration metrics. Predictions have an explicit ablation
`run_kind`/`variant` schema and the fixed benign/indeterminate/malignant class order.
The schema deliberately is **not** the historical `phase4-final` schema.

Completion is written only after persisted payload validation, readiness/source/Git
rechecks and checksum-manifest verification. Failures retain partial artifacts with
no valid completion. No failed or completed identity is automatically overwritten,
resumed or rerun. Inspect failures before requesting any new action; this milestone
does not offer a cleanup/retry or completed-run reuse option. Existing completed
sources may be read for analysis only after the separate loader validates them.

## Future completed-source comparison

`load_ablation_source()` is the distinct validated loader. Historical final/tuning
runners and source loaders are unchanged and intentionally reject these artifacts;
no identity rewriting or production guard monkeypatching is used. New ablations
are not retrospectively attributed to the historical complete-model commit.

After all three sources of a variant are genuinely completed and validated, an
explicit CPU analysis command is:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python scripts/46_evaluate_phase4_ablations.py \
  --ablation-source "${ABLATION_SOURCE42:?Use the validated seed 42 output directory}" \
  --ablation-source "${ABLATION_SOURCE43:?Use the validated seed 43 output directory}" \
  --ablation-source "${ABLATION_SOURCE44:?Use the validated seed 44 output directory}" \
  --supplementary-source artifacts/phase4/evaluation/phase4-evaluation-22cb656eabd040c2 \
  --authorize-analysis
```

Analysis requires exactly one variant and all three seeds. It reuses strict
complete-model source verification, exact paired validation/test identities and
the existing classification/calibration functions. Existing patient-paired
classification CIs use seed 20260914, 10,000 replicates and 95% percentile intervals,
reported per trained seed as ablation minus full model. Calibration is reported
with the existing point-estimate definitions; no new calibration CI is invented.

Best ablation weights are loaded in frozen eval/no-grad mode. Live unaugmented CT
and masks must match the immutable supplementary arrays exactly. Forward calls
receive images only. Native DSA shapes must match full-model maps and the existing
prior grid; no map adaptation is allowed. Saved classification replay uses the
already-recorded fixed tolerance (`atol=1e-5`, `rtol=1e-4`). Discrepancies are
reported and comparison interpretation disabled, never repaired by changing saved
predictions, checkpoints or tolerances.

The existing five alignment metrics, inclusive 0.90 quantile ties and 0.5
area-resized soft-mask threshold are unchanged. The already-completed supplementary
bundle must be replay-consistent, source-bound and checksum-valid. Its prior is
loaded as-is, with exact 807-training-nodule identities and array provenance; it
is never rebuilt, selected or tuned. New safe NPZ evidence includes native maps,
stable identifiers, CT/target arrays and an unchanged copy of that prior, with
dtype/shape/array/file hashes. Per-nodule tables retain the flagged case
`LIDC-IDRI-0052__nodule-49c6132bfb77f3f1`; an empty coarse consensus foreground
remains zero Dice/IoU with an explicit reason, not a repaired output.

Attention differences use the existing paired patient-cluster bootstrap for
ablation-minus-full and ablation-minus-fixed-prior, separately per trained seed.
Missing correlations remain explicit, including empty-defined replicate counts.
No 549-observation independence assumption, CI-endpoint averaging, favorable-seed
selection, cross-variant total-loss comparison, causal claim or population-wide
significance claim is made. Outputs occupy a new ablation-analysis namespace and
cannot replace earlier evaluations.

## Verification limits

Tests use tiny CPU tensors/models and explicitly synthetic source fixtures. Real
readiness boundaries are isolated where necessary; separate tests exercise strict
configuration, identity, persistence, source, launcher and evidence contracts.
Launcher tests replace processes with fakes: no GPU job or university training is
launched. Original augmentation/metric implementations are reused, not modified.

Run locally with:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python -m pytest -q
python -m compileall -q src scripts tests
git diff --check
```

The implementation cannot establish university-source availability, runtime/GPU
compatibility, convergence, replay agreement or any ablation result without the
separately authorized real runs. No ablation results are claimed by this milestone.
