# Performance-v2 implementation preparation — 2026-10-07

Implemented additive review components under
`src/consendxformer/performance_v2/` against the reviewed
[v2 protocol](performance_v2_protocol.md) and
[exact topology](performance_v2_model_topology_v2.md). These components supersede
the earlier "not implemented" description only within the scope below. The
historical plan/configuration, provenance, input/security records and Stage A
remain byte-identical. No training/data/server/workflow launcher is added.
`pilot_execution_authorized=false`; Stage B and GPU pilots remain paused.

## Implemented scope

| Component | Behavior |
|---|---|
| `model.py` | Random CPU ResNet50 V1.5 construction with `weights=None`; layer2/3/4 features, fixed 49×128 token path, two four-head pre-norm blocks, spatial attention, deep/gated fusion, pooled128 and linear three-class diagnostic/optional reader heads |
| `objectives.py` | HARD, exact LEGACY batch target-mass reduction, SHARED-010, SEPARATE-010; conditional CAPACITY-010 with unweighted hard reader CE; invalid q/u/labels/weights rejected without repair |
| `rng.py` | Existing SHA-256 stream derivation; private shuffle/augmentation generators; isolated CPU backbone, shared-fusion and two head construction scopes; restores global CPU RNG and never seeds CUDA |
| `optimization.py` | Four exhaustive identity-disjoint AdamW groups, MHA projection and positional-table handling, rejection of unknown/aliased/frozen parameters, exact recipe-A epoch schedule; no optimizer step |
| `selection.py` | Typed validation-only epoch records; global-maximum checkpoint tolerance then unweighted diagnostic CE/earliest epoch; separate raw patience; complete declared eligible screens and two complete three-seed repeat families |
| `reuse.py` | All 13 exact config/input bindings, fixed model/protocol/recipe/seed/objective identity, trusted completion digest, completion/manifest linkage and supplied freshly verified output digest map; mismatch/incomplete evidence rejects reuse |

Pretrained initialization requests are explicitly rejected. No loader, state-dict
import, download or permissive missing-key path exists. RANDOM, HARD/LEGACY/SHARED/
SEPARATE objective arms share the exact common construction; optional head
creation cannot perturb it. A test changes invented backbone values in memory to
check independence of new shared/head initialization; that is not ImageNet or
RadImageNet loading/compatibility evidence. CAPACITY construction/loss requires
an explicit component flag, which is not pilot execution authority.

`diagnostic_probabilities` requires eval mode and bypasses the reader head.
Ordinary `forward(images)` returns diagnostic/reader outputs plus intermediate
features for review; it accepts no annotation/mask inputs. There is no ordinal
head, attention-alignment objective or hidden dropout. Construction validates
key R50 constructor semantics; full pretrained key/dtype/layout/security mapping
remains a separate gate. Arbitrary ResNet50 tensors cannot be loaded into the
unchanged custom Stage A CNN through these components.

Reuse is a pure validator, not evidence acquisition: its caller must obtain the
completion digest from a trusted ledger and actually rehash original outputs
before supplying `verified_output_hashes`. Selection likewise validates declared
validation records, not their experimental provenance. A later input/receipt
adapter must prove the split/source bindings. Neither function grants execution.
Initialization scopes serialize their own use of torch's global CPU generator;
unrelated threads must not concurrently consume that generator during model
construction. Cross-version seeded-byte equivalence is not claimed.

## Actual validation

**43 focused synthetic tests passed**, zero failures/errors/skips and no warnings.
Tests use invented CPU tensors/labels/receipts only; gradient checks perform
autograd without optimizer updates. Coverage includes:

- full hybrid feature/output dimensions, probability/spatial-softmax mass and
  diagnostic-only inference bypass;
- independent arithmetic for weighted hard, batch-mass LEGACY and unweighted
  reader/capacity losses; invalid supervision and unsupported arms;
- reader gradients reaching the shared encoder while bypassing the diagnostic
  head, plus separate hard/reader gradient routes;
- bitwise common initialization across objective arms, distinct seeds, private
  stream independence and global CPU RNG restoration;
- exhaustive optimizer coverage, bias/norm/MHA/position policies, alias/unknown/
  frozen-parameter rejection and schedule endpoints without a step;
- each execution/input reuse digest mismatch, partial/bad receipts, output
  mismatch, unclassified fields and incompatible model/seed/objective/recipe;
- non-transitive tolerance traps, CE/earliest ties, patience boundaries, complete
  eligible screening, all-three-seed means and rejection of survivor selection.

Exact local runtime: Windows AMD64, CPython **3.14.6**, Torch **2.13.0+cpu**,
torchvision **0.28.0+cpu**, NumPy **2.5.2**; two CPU threads, CUDA hidden and
CPU-only torch build. `pytest` is absent; standard-library `unittest` ran the
focused tests. No dependencies were installed or changed. The bundled CPython
3.12.14 runtime has no torch/torchvision/pytest. The listed Windows Store Python
3.9 executable could not start (Access is denied). No approved Linux/GPU testing
environment was exercised, and no full repository-suite pass is claimed.

An initial import probe with immediate subprocess denial stopped at Python3.14's
Windows `platform.machine()` → `cmd /c ver` host-information probe. The final
harness completes that real read-only host observation first, then denies
network and subprocess calls during imports/tests. It also denies research-array/
checkpoint file access, weight loaders/download helpers, compilation and AdamW
steps. No platform metadata is mocked. This harness restriction was adjusted
explicitly; no package substitution or environment repair occurred.

The external review package retains exact runtime/source hashes, test output,
the initial 41-test result and the final 43-test result. The two added selection
tests guard against omitting an eligible screen to select only survivors. The
standard reproduction command with already available compatible dependencies is
`python -B -m unittest discover -s tests -p 'test_performance_v2_*implementation.py'`
plus the contracts test file. Prefer the included guarded review harness for
this specific local evidence scope. Do not install dependencies or enable a
pilot merely to reproduce the record.

## Remaining implementation and execution blockers

1. Independent review of these components and exact topology/initialization/
   optimizer mapping. The passing tests are local synthetic evidence, not owner
   acceptance, pretrained compatibility or scientific improvement.
2. No reviewed research data/transform adapter, production trainer, checkpoint
   writer, atomic completion pipeline or phased budget-enforcing launcher exists.
   Keep frozen inputs, labels, transforms, metric definitions and exact reuse;
   a future implementation must bind them rather than assume caller truth.
3. Approved supported runtime, complete transitive/native-image inventory,
   security/serialization/containment review and full Docker/native acceptance.
   The local Python3.14 CPU result does not replace the proposed Linux runtime
   or its hash/ABI/glibc/CUDA/determinism and resource gates.
4. ImageNet terms/provenance/all-split overlap and strict semantic loading/state
   verification. RadImageNet remains blocked on its own release, terms, bytes,
   interface, mapping and overlap evidence. No real weight bytes were accessed.
5. Separately scoped future integration/synthetic validation, frozen-reference
   binding audit and explicit owner execution authorization. No automatic B1
   retraining, extra pilot run, retry or budget extension is implemented.

The 9–10-run core and optional three-run capacity budget are unchanged proposals,
not executed runs or GPU-hour estimates. Historical test exposure, exploratory
versus independent-confirmatory boundaries and Stage A accepted closure remain
unchanged. Dependency/metadata collection remains separately authorized work.
