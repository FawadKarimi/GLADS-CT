# Exploratory post-hoc attention comparison and evidence export

This separate, opt-in evaluator was proposed **after the original attention
results were seen**. It is not preregistered. It does not train, tune, select a
model, or alter the existing evaluation definitions. Spatial attention alignment
does not establish causal explanation, superiority over B1, or the contribution
of the attention loss. The location prior is a deterministic training-only
aggregate, not a learned classifier or an ablation.

The versioned specification is `configs/phase4_attention_evidence_v1.json`.
The evaluator rejects changes to that specification. Final source model commit
`e1b174490245c616e355ef7c0c6f7c3c239d3acd` remains distinct from the new evaluator
commit. The final lambdas remain ordinal `0.5` and attention `0.05`.

## University-server usage (not run in the coding environment)

Run from a clean checkout of the committed `phase4-consendxformer` branch with
the existing university source bundles and frozen data paths available. The
actual pinned `external/LungDxFormer` checkout and installed runtime dependencies
must be present. Do not substitute another baseline implementation. Checkpoints
are loaded through the existing repository checkpoint APIs: use only trusted
research bundles, not arbitrary uploaded PyTorch pickle files.

```bash
python scripts/44_run_phase4_attention_comparison.py \
  --phase4-source artifacts/phase4/experiments/phase4-final-s42-c1e650d77585ed9c \
  --phase4-source artifacts/phase4/experiments/phase4-final-s43-946220804796df93 \
  --phase4-source artifacts/phase4/experiments/phase4-final-s44-8c74c3dd1e666bb5 \
  --b1-source artifacts/phase3/experiments/phase3-b1-s42-67d865cd9ff05a4c \
  --b1-source artifacts/phase3/experiments/phase3-b1-s43-7947651595ca43ab \
  --b1-source artifacts/phase3/experiments/phase3-b1-s44-be7454203e5b8aa0 \
  --reference-attention-bundle artifacts/phase4/evaluation/phase4-evaluation-7894ae140c091dba \
  --device cpu
```

CPU is the default. `--device cuda` is an explicit server-only option; no
autocast is used and deterministic PyTorch algorithms are required. CUDA may
require `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before process startup. Unsupported
deterministic operations fail instead of silently falling back to nondeterminism.
Do not relax the numerical tolerances in response to real-data discrepancies.

The CLI prints `OUTPUT_DIR` and `COMPLETION_MARKER`. Outputs go into a new
`artifacts/phase4/evaluation/phase4-evaluation-<identity hash>` directory. Identity
includes the supplementary protocol, evaluator commit, all six source descriptors
and file hashes, prior attention-bundle manifest hash, and recorded runtime
versions/device settings. An existing destination is never overwritten, including
an incomplete destination. An output cannot be nested inside a completed bundle.
An optional `--output-root` can place the new bundle elsewhere; protected source,
data and external trees remain forbidden. Preserve any failed partial bundle for
inspection rather than deleting or rewriting earlier evidence.

## Binding and matched inference

Exactly one verified final ConsenDxFormer and one verified B1 source are required
for **each** seed 42, 43 and 44. Existing completed-source, checkpoint, manifest,
configuration, frozen-cohort and saved-prediction validators are reused. Duplicate
seeds, different cohorts or source hashes are rejected. The previous attention
bundle must be completed, checksum-consistent, use the unchanged metric definition,
and bind to these same three ConsenDxFormer best checkpoints and test identities.

B1 uses its actual pinned external class, recorded effective Phase 1 configuration
plus the recorded Phase 3 reproducibility settings, and persisted `best.pt`
`model_state`. Best epoch, validation loss, configuration and source/data/split
provenance are checked against the saved history/identity. ConsenDxFormer uses the
existing final-best loader. No `last.pt`, approximation, disabled-ablation model,
or new weights are substituted. Imported baseline files are checked for their
actual pinned-source locations and the source tree must remain unchanged.

Both models receive clones of the **same** unaugmented image batch in eval mode,
with frozen parameters and `torch.no_grad()`. Masks never enter either forward
call. The readiness-validated loader binds all 183 test nodules and 99 patients.
Images, masks and identities must also agree exactly across all three seeds.
Actual `[N,1,H,W]` native attention shapes are recorded and must match between
models and across seeds. No attention-grid adaptation is allowed.

Saved logits and probabilities are matched by nodule identity, not incoming row
order, and compared with reconstructed float32 outputs using
`abs(actual - saved) <= 1e-5 + 1e-4 * abs(saved)` for every class.
All differences outside tolerance are exported, with identities and both values.
Saved predictions are never changed.

## Fixed train-only location prior

The prior uses exactly the 807 frozen **training** nodules. Readiness is revalidated
at prior construction, including frozen manifests, supervision and the Phase 3
input fingerprint; the exact training identities/mask paths must agree with the
persisted source. Test/validation rows, duplicates and incorrect counts are
rejected by the prior computation boundary.

For each training soft mask, use the existing loader and metric area-resize
convention to reach the common native grid, then normalize that mask to unit
spatial mass. Compute the arithmetic mean of those 807 distributions in canonical
identity order and normalize numerical drift once. This is not normalization of
an average of unnormalized masks. No parameter or alternative prior is selected
using test masks or test alignment. The identical resulting array is used for
every test nodule and trained seed, and is saved only once.

Provenance includes every training identity, each mask path and source-file SHA256,
readiness evidence, the computation definition, and the prior array's dtype,
shape and hash. Input files are sealed before reading and checked again before
completion. Test masks are used only as fixed evaluation targets.

## Evidence and numerical replay

The five existing metrics are reused directly: KL(consensus || attention), spatial
correlation, distribution overlap, Dice and IoU. Existing normalization/degeneracy
rules remain unchanged. The attention foreground is `attention >= quantile(0.90)`
**including ties**, and consensus foreground is `area_resized_soft_mask >= 0.5`.
There is no top-k replacement.

New outputs include:

- `test_inputs.npz`: stable Unicode patient/nodule/series/split/class identifiers,
  actual input CT crops, input soft masks, area-resized native-grid masks and
  normalized target distributions. This shared row order applies to every model
  archive and the repeated prior.
- `seed_42_model_arrays.npz`, `seed_43_model_arrays.npz`,
  `seed_44_model_arrays.npz`: raw native B1/CDX attention maps and reconstructed
  logits/probabilities; class order is benign, indeterminate, malignant.
- `location_prior.npz`: the one fixed native-grid prior array.
- `tables/alignment_per_nodule.csv`, `alignment_per_seed.csv`,
  `alignment_aggregate.csv`, `prior_per_nodule.csv`: all five metrics, with
  mean-of-seed-means and sample SD descriptive summaries for the two models.
- `tables/paired_differences_per_nodule.csv` and
  `paired_patient_confidence_intervals.csv`: all paired comparisons for every
  trained seed; the fixed prior is not counted as three independently fitted models.
- `tables/flagged_case.csv`: all six model/seed records plus the fixed prior for
  `LIDC-IDRI-0052__nodule-49c6132bfb77f3f1`.
- `overlay_panel.png`, `evaluation.json`, `summary.md`, full provenance, a copied
  supplementary protocol, checksum manifest, and a completion marker written last.

Per-nodule evidence records the actual native shape, numerical quantile threshold,
attention and consensus foreground counts, intersection, union, constant-map and
undefined-correlation flags. `zero_overlap_reason` distinguishes an empty target
foreground after coarse resizing/thresholding from two nonempty disjoint sets.
The flagged case is reported as observed, never repaired to remove zero scores.

NPZ archives forbid object arrays and are read back with `allow_pickle=False`.
Each array has recorded shape, NumPy dtype string and SHA256 over the ASCII
canonical-JSON header hash followed by contiguous array bytes. Each file also has
a SHA256; sorted manifest entries have a bundle hash. ZIP entry order/timestamps
are fixed. Exported model maps are read back and their metrics recomputed at
`atol=1e-12, rtol=1e-12`. Earlier ConsenDxFormer per-nodule and per-seed aggregate
scalars are compared at the separately fixed `atol=1e-8, rtol=1e-6`, including
explicit null/defined-count agreement. These are compatibility tolerances, not
criteria for changing any metric definition.

A finished diagnostic export can have `status=complete_with_replay_discrepancies`.
In that case `replay_consistent` and `comparison_interpretation_permitted` are
false; inspect every classification/scalar discrepancy before interpreting model
comparisons. The previous bundle remains untouched. Completion means export
finished, not that replay necessarily agreed or a scientific claim is established.
Source/input mutation or failed trust checks cannot produce a valid completion.

## Deterministic panel and descriptive uncertainty

The panel always includes the flagged nodule, plus low, lower-median and high
mean ConsenDxFormer Dice cases over all three seeds, sorted ascending by value
then nodule ID. Duplicate selections share a panel row with all selection reasons.
Display seed is fixed at 42, never selected for favorable results. Columns show
actual CT, area-resized consensus, B1, ConsenDxFormer and the fixed prior.
Nearest-neighbor grid upsampling and 50% heatmap alpha are display-only; attention
maps/prior are divided by their own maxima for visibility, not for metrics. CT is
clipped to [0,1] and consensus displayed on [0,1]. Raw values remain in the NPZs.
No synthetic or generated heatmaps are used as research evidence.

Paired differences are ConsenDxFormer minus B1, ConsenDxFormer minus prior and
B1 minus prior. Positive KL differences indicate worse alignment, whereas positive
differences in the other four metrics indicate greater alignment. These signs
alone do not establish superiority or an attention-loss contribution.

Each trained seed has separate 10,000-replicate patient-cluster bootstrap intervals
using PCG64 seed `20260914` and linear 2.5/97.5 percentile endpoints. Each sampled
patient occurrence includes all of that patient's nodules; both sides use the
same patient draw. The estimand is the nodule-mean paired difference, not an
equal-weight patient mean. The full patient order and draw hash are recorded.
Correlation differences use pairwise-defined nodules only while preserving the
full patient draws. Replicates with no defined pair are omitted with explicit
counts; if no pair is defined, the estimate/interval is null. No 549-row pooling,
CI-endpoint averaging, causal or population-wide significance claim is made.

## Local verification and limitations

Unit/integration tests use small synthetic CPU fixtures, never university data.
They exercise source/cohort rejection, actual pinned model/checkpoint API
compatibility with untrained weights, image-only no-grad inference, deterministic
training-only prior computation, safe map round trips, threshold ties/empty coarse
foregrounds, paired patient resampling, discrepancy exports and no-overwrite/input
immutability. Synthetic fixture panels are test outputs, not research evidence.

The coding-environment validation does not establish that the university bundle
will replay within tolerance, nor what B1/prior comparisons or the flagged case
will show. Those observations require the separately authorized server run. No
real inference, GPU job, training, ablation or existing-artifact rewrite is part
of this implementation milestone.
