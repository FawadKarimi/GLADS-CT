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
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. Both Torch and the CUDA driver must enumerate
exactly one logical device. `phase5_cuda_identity` queries **ordinal 0**, not the
NVIDIA management index, using `cuDeviceGet`, `cuDeviceGetUuid_v2` and
`cuDeviceGetPCIBusId`. The directly queried 16 UUID bytes must identify the
selected physical UUID. PCI domain/bus/device/function are parsed as numeric
components and compared, so `00000000:3B:00.0` equals `0000:3b:00.0` without
discarding either raw string. Full four- or eight-digit domains, two-digit bus
and device, and a function in 0..7 are required; truncated/malformed addresses
fail. The exact Torch/management name check remains auxiliary. Torch 2.3's
missing UUID/PCI properties are not required, and neither name, memory, index
nor the environment alone can substitute for a failed direct query.

The small standard-library ctypes binding loads `libcuda.so.1` lazily through
the Linux system loader only in the live CUDA path. It never searches toolkit
stub paths, downloads libraries or tries an alternative driver/device. CUDA
execution through this helper is Linux-only; Windows CPU execution remains
supported. The declarations use Linux's C calling convention (`CDLL`), C `int`
for `CUresult`/`CUdevice`, unsigned int for `cuInit` flags, a 16-byte `CUuuid`
structure, and a writable 32-byte PCI buffer (at least the API's 13-byte minimum,
also accommodating an eight-digit domain). Every required symbol and return code
is checked, including `cuInit(0)`; a toolkit stub's non-success initialization
cannot pass. An unterminated/non-ASCII buffer, zero UUID, wrong device count or
identity mismatch fails before model construction/research loading. There are no
contexts created by this helper, model probes, tensor allocations or kernel
launches. See NVIDIA's [CUDA 12.1 device declarations and UUID semantics](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-driver-api/group__CUDA__DEVICE.html),
[PCI buffer contract](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-driver-api/group__CUDA__MEM.html)
and [initialization contract](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-driver-api/group__CUDA__INITIALIZE.html).
The [NVIDIA header's CUDAAPI definitions](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/cuda_8h_source.html)
also distinguish Windows stdcall from the Linux C convention; no Windows CUDA
calling-convention compatibility is inferred from these CPU tests.

Timing specification container v1 now explicitly requires `evidence_revision: 2`
and mapping marker `cuda_driver_uuid_pci_v2`. Runtime provenance includes
`direct_driver_identity` (API, schema, queried ordinal/count, raw UUID bytes,
canonical UUID, raw PCI and numeric components), Torch count and auxiliary name.
Missing fields or the old environment-only method cannot certify this revision.
Exact specification checks reject earlier timing evidence, without migrating or
rewriting it. Historical research-v1 read-only validation remains untouched.
The new fields participate in existing runtime/source hashes, run identity,
research compatibility checks and embedded copy/readback. Offline validation
checks recorded bindings without querying or loading any GPU driver. CPU tests
exercise native signatures and buffers with explicitly labeled ctypes callbacks;
they do not validate an actual CUDA installation or server compatibility.

Memory equality across APIs is **not device identity**. Runtime diagnostics
retain `nvidia_smi_total_mib`, its exact `nvidia_smi_total_bytes` conversion,
`torch_device_properties_total_bytes`, and signed `torch_minus_nvidia_smi_bytes`.
Positive integer totals and integer diagnostics are required; offline validation
recomputes the conversion and delta. No tolerance, clipping or fitted offset
connects these readings. They remain part of strict timing/research runtime
compatibility, and existing allocator/free/peak consistency guards are unchanged.
NVIDIA documents [framebuffer memory reporting/reservations](https://docs.nvidia.com/deploy/nvidia-smi/index.html#fb-memory-usage),
but that does not identify the cause of any particular cross-API difference.

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

T1/T2 were required corrections to the then-unpublished timing-evidence v1 contract,
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

### Supplied failed attempt and zero-forward mapping check

The supplied server record says `timing-review-01` failed at the old memory-based
live identity guard, before model construction, forwards or output-directory
creation. It produced no measurements or timing bundle. Its terminal evidence
must be retained as-is; do not manufacture a failed/completed bundle for it or
reuse its attempt ID. The supplied diagnostic reported management total 11,264
MiB (11,811,160,064 bytes) and Torch total 11,546,394,624 bytes, a signed difference
of -264,765,440 bytes (-252.5 MiB), with matching names and one logical device.
This is not a measured leak, model allocation, insufficient-free-memory result
or proven driver-overhead amount. These are supplied observations, not locally
verified GPU measurements or production acceptance constants.

After review/publication, a separately authorized operator should first run this
**unexecuted zero-forward mapping check** from the university checkout, replacing
the placeholder with a freshly verified full UUID. Bootstrap performs the fresh
availability check. Do not set or normalize `CUDA_LAUNCH_BLOCKING`; its inherited
raw value is captured. This command writes no bundle and constructs no models:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=GPU-<freshly-verified-full-UUID> \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python -B - 'GPU-<freshly-verified-full-UUID>' <<'PY'
import json, sys
from consendxformer.phase5_device import bootstrap_device, validate_device_evidence
bootstrap = bootstrap_device("cuda", sys.argv[1], torch_threads=32)
from consendxformer.phase5_timing_runtime import runtime_environment
observed = runtime_environment("cuda")
validate_device_evidence(bootstrap, observed)
print(json.dumps({"bootstrap": bootstrap, "runtime": observed,
                  "model_forwards": 0, "timing_executed": False}, indent=2))
PY
```

A successful mapping check is not authorization or proof of benchmark completion.
Only afterward may a new timing attempt be considered under separate approval.

### Later timing example (separate authorization)

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
  --authorize-synthetic-timing --attempt-id REPLACE_WITH_NEW_APPROVED_ATTEMPT_ID \
  --device cuda --gpu-uuid GPU-<freshly-verified-full-UUID> --torch-threads 32
```

The placeholder command is intentionally not executable as written. Use the
normal pinned dependencies/external source and clean reviewed checkout. Review
all 18 blocks, memory, environment, failures and evidence hashes afterward.
Only a separate later authorization may enable the research runner with its own
`--authorize-execution`, approved timing path/hash, all unchanged research-source
options and the same UUID/runtime/thread settings. This document grants neither
timing execution nor university-server/research access.
