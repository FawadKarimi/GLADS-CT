# Phase 4 evaluation and ablation infrastructure

This milestone implements prespecified analyses, not research results. No real
final-model retraining, real evaluation or ablation training was performed as
part of implementation. Final sources must come from commit
`e1b174490245c616e355ef7c0c6f7c3c239d3acd`, seeds 42/43/44, with loss weights
exactly 0.5 (ordinal) and 0.05 (attention). The evaluator must run from a clean
`phase4-consendxformer` descendant checkout with its canonical configuration
tracked at HEAD. There is no trust-bypass CLI flag.

## Entry points

Run from the repository root on the server holding the completed artifacts.
The following are command templates, not commands executed for this milestone.
Replace each uppercase source placeholder with an existing completed experiment
directory; pass exactly one source for each of seeds 42, 43 and 44.

```text
python scripts/42_run_phase4_evaluation.py --phase4-source CDX42 --phase4-source CDX43 --phase4-source CDX44 --b1-source B142 --b1-source B143 --b1-source B144
python scripts/43_run_phase4_posthoc.py --phase4-source CDX42 --phase4-source CDX43 --phase4-source CDX44 --device cpu
```

Script 42 reads saved predictions only. Script 43 performs unaugmented,
no-gradient inference using each persisted FINAL `best.pt`, never `last.pt`.
It requires the exact frozen rendered data, readiness evidence and manifests
to be available at the configured locations. Add `--include-boundary` only
when explicitly requesting the separate boundary-stress inference. CUDA is
opt-in; deterministic algorithms are required and unsupported operations fail
rather than silently weakening determinism. Cross-device bitwise equality is
not claimed. Source checkpoints should be trusted repository-produced files;
PyTorch checkpoint deserialization is not a sandbox for untrusted pickle data.

Both commands accept `--output-root` (default `artifacts/phase4/evaluation`).
Outputs include JSON results, CSV tables, Markdown summary, protocol/provenance,
a checksum manifest and a `COMPLETED.json` written last. IDs bind evaluator code,
protocol and source snapshots. Existing output IDs and paths inside protected
source directories are rejected. Failed partial output directories have no
completion marker; inspect and explicitly relocate them before retrying the
same ID. Existing final experiment artifacts are never overwritten.

## Fixed analysis definitions

Class order is always benign, indeterminate, malignant. Calibration is reported
per model and seed, and as three-seed mean/sample SD (`ddof=1`). NLL is the mean
negative natural log of the true-class probability, floored at `1e-15`. Brier
is the mean sample sum of squared errors across all three classes, without
division by the number of classes. ECE is top-label multiclass ECE: sum over
bins of bin frequency times the absolute accuracy/mean-confidence gap. There
are exactly 15 equal-width bins on [0,1], left closed/right open except the
final bin includes 1. No temperature scaling or calibration fitting is used.

Predictive entropy uses natural logs and `0 log 0 = 0`. Spearman association
with frozen reader-disagreement entropy uses average ranks for ties; constant
inputs produce an explicit undefined/null result. Pooled descriptive analysis
first averages model entropy over seeds for each nodule, avoiding treatment of
three predictions of the same nodule as independent observations. The high/low
disagreement threshold is the median of the frozen **training** cohort, not a
threshold selected from test performance. Equality belongs to the high group;
training cohort/threshold provenance is included.

Risk is retained-sample 0/1 argmax classification error. Coverage is `k/N`.
The pipeline reports both increasing-entropy and decreasing-maximum-confidence
rankings, with deterministic patient/series/nodule identity tie-breaking. AURC
is the right-endpoint discrete integral: the mean risk at `k/N`, `k=1..N`.
There is no test-driven threshold fitting.

Attention analysis measures spatial correspondence, **not causal explanation**.
DSA attention must be finite, nonnegative and already unit spatial mass within
`1e-6`. Soft masks are area-resized to the native attention grid. Continuous
metrics are target-to-attention KL, flattened spatial Pearson correlation and
unit-distribution overlap. Dice/IoU compare soft mask >=0.5 with attention >=
its per-map 0.90 quantile (linear interpolation). KL uses `1e-8` stabilization.
Constant-map correlation is undefined and excluded with a defined-count report.
Metrics and exact patient/nodule identities are retained per seed, with
mean/sample SD of seed means; these thresholds are not tuned on test outcomes.

## Boundary cohort and limitations

The authoritative Phase 2 integer-score rule is reused exactly: lower band
`[2.5, 3.0)` and upper band `(3.0, 3.5]`. The approximate 2.6–2.9 / 3.1–3.4
description must not replace these endpoints. Mean exactly 3.0 is not boundary.
The pipeline validates integer score sums and reader counts, canonical/rendered
membership, completion/checksum ledgers, frozen patient splits, input lock and
absence from the primary nodule cohort. It rechecks source files around inference.

Boundary outputs retain original split, continuous reader mean, score sum,
reader count, stratum and three model probabilities. No primary true class is
invented; no primary macro-F1 is computed. Counts and mean class probabilities
are summarized by seed/stratum, separately from the primary headline. Patients
can overlap primary training patients even though boundary nodules were excluded;
this stress cohort is not an independent held-out clinical validation cohort.
Inference availability also requires all declared boundary rendered inputs.

## Patient bootstrap and ablations

The fixed bootstrap uses NumPy PCG64, seed 20260914, 10,000 replicates and 95%
percentile intervals with linear quantiles. Each replicate samples patients
with replacement and includes every nodule for each sampled patient occurrence.
Paired B1/ConsenDxFormer comparisons are reported as **ConsenDxFormer minus
B1**, using the same patient draw after exact patient/nodule/series/split/true-
class matching. Metrics are macro-F1, benign F1, indeterminate F1/recall and
malignant F1. Zero denominators yield zero; macro-F1 always averages all three
fixed classes, including absent classes. Intervals are per trained seed and
conditional on these models and this cohort; they are not parametric tests
over three seeds, nor claims about training-population uncertainty.

`phase4_ablation_specifications()` returns exactly four single-toggle OFF
specifications, and `phase4_ablation_protocol_record()` supplies the complete
frozen control/provenance record. These are non-executing specifications:

- Disagreement OFF uses baseline-style weighted hard cross-entropy for the
  diagnostic term; independently enabled ordinal/attention terms remain.
- Ordinal OFF removes the ordinal auxiliary contribution.
- Attention OFF removes the attention-alignment auxiliary contribution.
- Hierarchical fusion OFF uses the model's existing non-hierarchical path.

Seeds, cohort, splits, preprocessing, applicable selected loss weights and
validation checkpoint selection stay fixed. B1 remains the external fair
baseline; "all toggles off = B1" is neither supported nor claimed. This
milestone supplies no ablation-training launcher.
