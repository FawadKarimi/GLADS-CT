# Phase 5 synthetic timing: implementation, not execution

This separate `scope=synthetic_timing_only` implementation supports later timing
review. No actual architecture timing benchmark or research execution has been
performed by this implementation task. Tiny synthetic test predictors, fake
clocks and device stubs are explicitly labeled and are not runtime evidence.
Synthetic constructor tests initialize the real architectures without forwarding
them. The frozen proposal and insertion/deletion scientific protocol are unchanged.
The protocol's historical `synthetic_timing_evidence: null` remains historical;
new measurements live in a separate versioned evidence bundle, not that field.

## Fixed benchmark and access boundary

`configs/phase5_synthetic_timing_v1.json` is checked against the exact versioned
specification. `scripts/50_run_phase5_synthetic_timing.py` requires explicit
`--authorize-synthetic-timing` and a safe, operator-chosen `--attempt-id`. There is
no automatic retry, resume, device selection or fallback. Outputs are create-only
under `artifacts/phase5/synthetic_timing/phase5-timing-<identity hash>/` (an optional
output root must remain within that protected namespace). Existing complete,
partial and failed attempts are never overwritten. Failed attempts must not be
repackaged as completion; repeating a benchmark requires a new explicit decision.

The actual complete model uses `phase4_execution._model` and the B1 model uses
`phase4_attention_runtime.construct_b1` inside the verified pinned external source
context. Both use the normal `load_phase4_config` configuration path. The runner
reads code/configuration/proposal and pinned external source only: no research
checkpoints, predictions, images, masks, labels, input CSVs or dataset loader.
It constructs once per seed 42/43/44 with synthetic initialized weights. All six
models remain resident before the first warm-up until all measurements and the
bounded synthetic export finish. Initial/final state hashes must agree; parameter
counts, state bytes and structure hashes are recorded. Construction has no probe
forwards. All calls reuse the real research image-only adapter, frozen eval mode,
no gradients, no autocast and strict deterministic algorithms (not warn-only),
with cuDNN benchmarking disabled and deterministic mode enabled.

The fixed unit order is complete 42, 43, 44, then B1 42, 43, 44. Each unit has
32 warm-up calls and three measured blocks of 128 calls: 416 calls/unit,
192 warm-up + 2,304 measured = **2,496 calls**, all batch-one float32
`[1,1,64,64]`. No call is deduplicated, probed or retried. Every block is exported.
Before model preparation/any forward, `plan.json` records and hashes all 416
positions reused identically by the six units, with the native row-major tile
order and exact tensor hashes; `synthetic_inputs.npz` records the base arrays.

For zero-based call index `j=0..415` within each unit:

- Source image is `[zero, ramp, random][j % 3]`, float32 `[1,64,64]`.
- Zero is all zero; ramp is `np.linspace(0,1,4096,dtype=np.float32)` reshaped
  to `[1,64,64]`; random is
  `Generator(PCG64(20260917)).random((1,64,64),dtype=np.float32)`.
- Direction is `[insertion,deletion][(j // 3) % 2]`; step is `(j // 6) % 65`.
- The unchanged `phase5_perturbation.tile_input` constructs whole 8-by-8 input
  tiles using `np.arange(64)` order and the same zero baseline. This synthetic
  order is not a replacement for any research DSA/prior/hash-control ranking.
- `j<32` is warm-up; later block index is `(j-32)//128`. Input construction and
  array hashing make no forwards. Raw logits/probabilities remain float32;
  the existing private float64 row-sum check is `atol=1e-6, rtol=0`, without
  clipping/normalization. Invalid outputs terminate the attempt immediately.

## Device isolation and compatibility

Both timing and research CLI entry points import a standard-library-only
bootstrap before importing PyTorch. CPU mode explicitly hides CUDA. CUDA mode
requires a **full physical GPU UUID**, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set
before startup, and an existing `CUDA_VISIBLE_DEVICES` that is either unset or
exactly that UUID (a numeric index is rejected). A fresh read-only `nvidia-smi`
inventory verifies the UUID and records physical index, PCI bus, device name,
driver, memory and active compute processes. The operational idle gate requires
no active compute process on that UUID and at most 256 MiB initially used. This
is an operational safety ceiling, not a scientific threshold, an availability
reservation or protection from later contention. Failure does not switch GPUs.

The process sets `CUDA_VISIBLE_DEVICES` to the verified UUID and
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, then verifies exactly one available logical device
`cuda:0` with matching name and total memory (NVIDIA MiB rounding allowance only).
The full UUID and physical/logical mapping are recorded. PyTorch 2.3 does not
expose UUID in device properties, so verification uses NVIDIA's full-UUID
visibility mapping, not an unsupported PyTorch property. See
[NVIDIA CUDA environment-variable semantics](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-environment-variables)
and the [PyTorch 2.3 device-property binding](https://github.com/pytorch/pytorch/blob/v2.3.0/torch/csrc/cuda/Module.cpp).
CPU tests use explicit CUDA/device stubs; they do not validate real GPU behavior.

Both entry points also capture the inherited raw `CUDA_LAUNCH_BLOCKING` value
before PyTorch/CUDA initialization, without writing or normalizing it. Unset is
explicit JSON `null`; empty text, `"0"`, `"1"` and any other explicit text remain
distinct identities. Live runtime observation checks the environment against that
capture, and offline validation/copy/readback require the field in both bootstrap
and runtime evidence and check their binding. Missing evidence is not a default.
Timing/research mismatches fail before research source/checkpoint loading.
This is compatibility recording, not a new environment-tuning policy. Boundary
synchronization does not erase differences in internal launch behavior; see
[NVIDIA's CUDA_LAUNCH_BLOCKING definition](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-launch-blocking).

Runtime identity includes Python/NumPy/pandas/PyTorch/CUDA/cuDNN versions, PyTorch
build hash, host/OS/processor, CPU affinity, thread/inter-op counts, relevant CUDA
and thread environment variables, TF32/matmul policy, deterministic execution
policy, and physical/logical device identity. Architecture identity binds exact
effective configuration hashes, constructor paths, LF-normalized code/configuration
hashes and the locked external repository/tree. Raw source-file snapshots are
also recorded and checked for mutation. A later research run must match these
architecture/configuration/source and runtime/device identities exactly; a change
requires review, not a compatibility bypass or implicit alternative benchmark.
Initial free memory, timestamps and startup physical index are diagnostics, not
exact-match compatibility fields; UUID/PCI bus/name/driver/memory are bindings.

## Measurement boundaries, exports and estimates

The real clock is `time.perf_counter_ns`. For each call, time construction
separately; time pre-call hash validation separately; synchronize CUDA before
starting the adapter timer, then call the adapter and synchronize before ending
it; time external output validation/hashing separately. Adapter latency includes
its input guards/private host copy, host-to-device transfer, actual forward,
device-to-host transfers, raw output copies and post-synchronization. It is not
kernel-only latency. Pre-synchronization, Python bookkeeping and diagnostics
outside these intervals belong only to the loop envelope.

Three mutually disjoint components are recorded per call and summed for each
of 18 measured blocks: construction, adapter, and external validation/hashing.
Warm-ups have individual records but do not enter extrapolation. The preparation
wall envelope includes preflight, deterministic plan generation/hash/export and
six-model initialization/hash; loop wall time includes all calls and bookkeeping.
These envelopes are **not added** to their nested components. A separately timed,
bounded NPZ export of exactly 2,496 raw logit/probability rows includes serialization,
write, hashing and read-back. It is not a proxy for research curve CSV export.
Final source/model checks and completion writing are outside this bounded I/O
measurement. No total runtime estimate is manufactured by adding nested timings.

Call records now retain integer `*_nanoseconds` for all three disjoint components,
and the loop retains `loop_wall_nanoseconds`, using the same clock reads as before.
The two validation intervals are added as integers before conversion. Their seconds
fields must equal the canonical integer value divided by `1e9`. For **all 2,496
calls including warm-up**, validation requires the loop integer duration to contain
the exact integer sum. Even a one-nanosecond deficit fails. It separately checks
`math.fsum` of all 7,488 component-second fields against the loop seconds, allowing
only `4 * math.ulp(contained_seconds)` of binary64 rounding. This covers component
conversion, stable-sum rounding (including a possible last-bit double rounding)
and envelope conversion; it is not an absolute or percentage timing allowance.
The exact integer guard still applies when float seconds cannot resolve 1 ns.
Equality is not required: untimed bookkeeping, pre-synchronization, memory sampling
and clock overhead legitimately lengthen the loop. Warm-up stays excluded from
extrapolation; preparation and bounded export remain separate and are not added
to the loop's nested measurements.

These are required corrections to the unpublished timing-evidence v1 contract,
not changes to the fixed benchmark schedule or frozen scientific protocol. Older
candidate timing records missing these fields are not accepted as new evidence.
Historical research v1 read-only validation remains unchanged and requires no
new timing fields; no completed historical result bundle is rewritten.

Memory snapshots occur with all six models resident before calls, at every
measured block end, and after bounded export. GPU allocated/reserved/current/peak
and free/total bytes are recorded; GPU peaks reset only after six-model loading.
Host RSS is process-lifetime peak (and Windows current working set where available),
not an independently reset phase peak. OOM/deterministic-operation failures stop
the attempt; no reduced residency, changed settings or CPU fallback is allowed.

Let `t[m,s,b]` be summed adapter seconds in each 128-call block. For every model
and seed, estimate `69540 * mean_b(t[m,s,b] / 128)`, then sum all six estimates
for 417,240 research calls. Also report each unit's three blocks and block min/max
extrapolations and sums of these minima/maxima. These ranges are **not confidence
intervals**. The estimate excludes research loading/startup, external construction,
validation and export/I/O. Synthetic weights/inputs, scheduling/cache locality,
different output distributions, GPU/CPU contention, warm-up and storage effects
limit transferability even with identical configurations and six-model residency.
Synthetic measurements do not establish actual research runtime or feasibility.

Every bundle contains `specification.json`, `plan.json`, `identity.json`,
`bootstrap.json`, `models.json`, `measurements.json`, `synthetic_inputs.npz`,
`synthetic_outputs.npz`, `source_snapshot.json`, `summary.md`, a checksummed
`artifact_manifest.json` and `COMPLETED.json`. Completion validates every payload,
sequence/input/output hash, actual attempted/accepted counts, all blocks and
extrapolation, model residency, memory/export diagnostics and bootstrap binding.
The manifest SHA-256 must be explicitly pinned by the later reviewer/operator.
Call failures/interruptions retain partial outputs and actual counts; independent
best-effort failure logging/marker removal preserve the original exception and
touch only this attempt's directory. Hard kills/unwritable filesystems retain the
same limitations as the research runner; a marker alone is not audit approval.

## Research integration without scientific changes

The research runner now requires `--timing-bundle PATH` and
`--timing-manifest-sha256 SHA256`. Completion, integrity, call counts and exact
architecture/runtime/device compatibility are checked **before loading any
research source/data/checkpoint**. Hashes bind the evidence into run identity and
the source snapshot. A create-only embedded copy under `provenance/timing/` permits
offline completion validation without reopening live research sources. New
research artifact/completion schemas are v2; historical v1 validation is retained
read-only, not a new-execution path.

Research provenance separates measured synthetic timing, adapter-only extrapolation
and any later observed `execute_curves` wall time (including replay barrier I/O,
excluding loading, final export/validation: not end-to-end runtime). The frozen
historical null is recorded as such. No research forward is added: 1,098 global
replays, endpoint caching, fixed targets/rankings/raw probabilities and the
417,240-call budget are unchanged. Benchmark completion never authorizes research
execution or completes insertion/deletion. All 15 hypotheses remain unresolved;
the full supplement and Stage A remain incomplete.

## Proposed future invocation — NOT executed or authorized here

Supplied server observations, not locally reverified: Python 3.10.17;
PyTorch 2.3.0+cu121; four RTX 2080 Ti GPUs, 11,264 MiB each; CPU affinity 64;
PyTorch CPU threads 32. GPU 0 had 10,094 MiB occupied and GPUs 1–3 had 8 MiB each.
GPU 1 is only a provisional target. After code review and separate timing-execution
authorization, an operator must obtain and recheck the intended physical GPU's
full UUID and current availability. Do not copy a numeric index into the UUID.
For example, after replacing the placeholder and choosing an unused attempt ID:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=GPU-<freshly-verified-full-UUID> \
PYTHONDONTWRITEBYTECODE=1 \
python -B scripts/50_run_phase5_synthetic_timing.py \
  --authorize-synthetic-timing --attempt-id timing-review-01 \
  --device cuda --gpu-uuid GPU-<freshly-verified-full-UUID> --torch-threads 32
```

The placeholder command is intentionally not executable as written. Use the
normal pinned dependencies/external source and clean reviewed checkout. Review
all 18 blocks, memory, environment, failures and evidence hashes afterward.
Only a separate later authorization may enable the research runner with its own
`--authorize-execution`, approved timing path/hash, all unchanged research-source
options and the same UUID/runtime/thread settings. This document grants neither
timing execution nor university-server/research access.
