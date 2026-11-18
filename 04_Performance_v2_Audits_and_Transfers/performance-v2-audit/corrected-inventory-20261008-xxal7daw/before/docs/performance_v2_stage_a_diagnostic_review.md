# Performance-v2 Stage A diagnostic review

Recorded 2026-09-27. This is a read-only review of saved TRAIN/VALIDATION
records, not a new Stage A analysis or a pilot release. Stage A remains immutable
within its accepted bounded sign-off. Stage B remains paused and
`pilot_execution_authorized=false`.

## Evidence and verification boundary

The reviewed input is
`audit-20260927T100333767013Z-539541171cb6.zip`, SHA-256
`e5663b68523b345a3b270108e6958e1f49498e3bec14ef6959e9f4d53394e36b`.
Its `artifact_manifest.json` SHA-256 is
`59d80e704803de57d38a5f9228cfdb20aa9ff763f1f9ef8e8a09ccb3e1c28781`;
the completion marker binds that exact manifest. All 12 manifest-listed payload
hashes were recomputed, and the archive contains exactly those payloads plus the
manifest and completion marker. The recorded extractor SHA-256 is
`1b3ad98fbe8acd572f1427586c31f755d6457b49640ce8a95eebb8b74e09b01d`,
matching the reviewed standalone script.

The server record reports Python 3.10.17, clean unchanged
`phase4-consendxformer` HEAD
`7908b8d69a8064240ac7733d683c28489b0077a2`, zero model forwards, no
training, no test-performance computation and no source changes. That historical
server checkout is not claimed to be synchronized to the later Stage A
documentation HEAD `f8278c34d6145170345dce0a518276adf0bb4144`.

Desktop verification covered archive structure, payload bytes, exported tables
and archived configuration strings. The 219 allowlisted source snapshots are
server-recorded fresh-byte checks; their original server files were not rehashed
on desktop. No checkpoint tensor, complete original result bundle, image or live
patient record was opened. Mixed metadata containers had been byte-hashed on the
server, but test rows were discarded before interpretation. The audit remains a
historical `status=partial` artifact and must not be rewritten.

## Resolved archive issues and source consistency

The only three reported issues were one per B1 seed for desktop parsing of two
YAML snapshots. All six UTF-8 snapshots safely parse, their reconstructed raw
bytes match the server-recorded SHA-256 values, and seeds 42/43/44 plus the
nominal B1 settings are consistent. This later read-back resolves those three
issues at the archived-configuration level only; no server rerun or completion
marker change is needed.

All 18 primary slots are present exactly once: B1, full ConsenDxFormer and the
four single-component OFF variants at seeds 42/43/44. The separate inventory
contains 24 tuning, four synthetic and three historical-context entries; none was
promoted into the primary set. No primary source is blocked.

| Check | Independently confirmed from exports |
|---|---:|
| Source rows | 17,100 = 18 x 950, TRAIN/VALIDATION only |
| TRAIN cohort | 807 nodules / 437 patients; 348 / 224 / 235 by class |
| VALIDATION cohort | 143 nodules / 84 patients; 65 / 38 / 40 by class |
| Identity agreement | Exact patient/nodule/series/split identity across all 18 |
| Within-source integrity | No duplicate nodule; no TRAIN/VALIDATION patient overlap |
| History rows / epochs / epoch-split groups | 19,312 / 442 / 884 |
| Per-class support checks | 2,652/2,652 agree with source counts |
| Checkpoint diagnostic rows | 18/18 recomputed from histories |
| Saved validation read-back | 18/18 match the minimum-loss history epoch |

The extractor did not originally enforce every history-support comparison as a
source gate. The successful read-back closes that concern for this immutable
archive, not for future extractor runs. A future extractor revision would need to
add the check or retain the limitation explicitly.

## Observed selection behavior

The following are arithmetic means across three saved validation histories, not
test results or confirmatory estimates.

| Variant | Executed epochs (42/43/44) | Macro-F1 at minimum-loss epoch | Maximum stored macro-F1 | Difference (pp) |
|---|---|---:|---:|---:|
| B1 | 34 / 23 / 28 | 0.618312 | 0.662401 | 4.409 |
| Full | 33 / 19 / 26 | 0.644107 | 0.677042 | 3.293 |
| Attention alignment OFF | 17 / 19 / 33 | 0.593238 | 0.657868 | 6.463 |
| Disagreement weighting OFF | 18 / 28 / 29 | 0.622365 | 0.660101 | 3.774 |
| Hierarchical fusion OFF | 33 / 25 / 29 | 0.659640 | 0.671096 | 1.146 |
| Ordinal auxiliary OFF | 17 / 16 / 15 | 0.627430 | 0.681628 | 5.420 |

Minimum-loss and maximum-macro-F1 epochs differ in 16/18 histories. Every
history ends exactly 10 epochs after its minimum-loss epoch, consistent with the
archived stopping patience; the configured 100 epochs was a cap, and actual runs
lasted 15--34 epochs. Full-model minimum-loss epochs are 23/9/16 and maximum-F1
epochs 28/19/16.

This supports prospectively aligning v2 stopping and selection to its declared
primary validation macro-F1. It does not show that longer training will improve
generalization. The maximum-F1 value is optimistically selected from the same
validation history, and corresponding tensors were neither shown to exist nor
loaded. Cross-entropy and macro-F1 can prefer different epochs; auxiliary losses
are not the sole explanation because B1 shows the same phenomenon. Per-epoch
learning rates, task gradients and fusion-gate traces were not recorded.

## Observed training-target discordance

The 15 Phase 4 copies contain the same 807 training targets. They are one cohort,
not 12,105 independent nodules. Independent read-back reproduces all proxy rows
and summaries within `1e-12`.

| Hard label | N | Strict hard-label/readers-mode disagreement | Tied maximum | Unique maximum | `q_y=0` | `u` approximately 1 |
|---|---:|---:|---:|---:|---:|---:|
| Benign | 348 | 9 | 53 | 286 | 0 | 0 |
| Indeterminate | 224 | 25 | 24 | 175 | 21 | 24 |
| Malignant | 235 | 7 | 28 | 200 | 0 | 0 |
| Total | 807 | 41 | 105 | 661 | 21 | 24 |

Among the 21 indeterminate cases with `q_indeterminate=0`, 17 have
`(0.5,0,0.5)`, three `(2/3,0,1/3)`, and one `(1/3,0,2/3)`. The 24 uniform
reader targets are ties, not strict conflicts. Mean `q_y` is 0.82783 / 0.68973 /
0.77057 and mean stored `u*(1-q_y)` is 0.10241 / 0.26697 / 0.13209 for benign /
indeterminate / malignant.

These are observed supervision-objective differences, not corrupted labels or
measured gradient conflict. They motivate the already accepted HARD, LEGACY,
same-head and separate-head comparisons. They do not establish that head
separation will help, and the shared encoder can still receive competing
gradients. No case may be deleted, relabeled or downsampled in response.

## Remaining limits and disposition

- Archived B1 receipts do not record numerical training class weights. The
  source-count calculation `[0.7729885057471264, 1.2008928571428572,
  1.1446808510638298]` agrees with recorded Phase 4 values but is not a newly
  verified historical B1 tensor.
- The snapshots do not establish the historical optimizer class or actual dropout
  wiring. Later owner-provided installed-version metadata is recorded in the
  [Phase 1 proposal](performance_v2_phase1_implementation_proposal.md). It resolves
  metadata availability only; runtime compatibility and implementation pins
  remain unverified and do not retroactively bind historical execution versions.
- Checkpoint bytes, complete bundles, installed model definitions, decoded image
  ranges and all-split overlap were not independently verified here.
- Pretraining bytes, SHA-256, license/terms, architecture mapping and numerical
  preprocessing compatibility remain unresolved. RadImageNet remains blocked.
- This validation-only development evidence establishes neither causal mechanism,
  test superiority, clinical validity nor external benchmark comparability.

The archive is sufficient to proceed to a reviewed implementation proposal. It
does not release Phase 1, replace any Stage A checkpoint, rewrite any result, or
authorize Stage B.

## Gate B correction read-back — 2026-09-27

The initial desktop verifier compared derived source-copy fields without fully
recomputing every target quantity or exported summary. The strengthened external
`verify_audit.py` now independently derives q_y from label and q, 1-q_y,
u*(1-q_y), categories at tolerance 1e-8 and bins [0,.25), [.25,.5), [.5,.75),
[.75,1]. It checks finite raw q/u, q sum at atol1e-6/rtol0 without normalization,
and compares every derived number/mean at atol1e-12/rtol0. It checks all 300
summary rows' counts, status, category counts and means, including empty means.
Label/identity consistency spans all 18 TRAIN/VALIDATION copies; raw TRAIN q/u
consistency spans the 15 Phase 4 copies. The unique denominator is 807 nodules;
12,105 repeated training rows are verification records, not additional samples.
Synthetic corruption tests and the recheck are recorded in the external review
validation report. The unchanged ZIP retains its historical partial marker.
