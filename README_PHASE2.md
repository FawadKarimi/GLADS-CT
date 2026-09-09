# Phase 2: authoritative LIDC-IDRI data and annotation pipeline

Phase 2 converts the original LIDC-IDRI CT source and radiologist annotations into deterministic, nodule-level 2D records. It implements data preparation only: it contains no model, loss, training, tuning, evaluation, or performance result.

## Data authority and scope

The research source is the original [TCIA LIDC-IDRI collection](https://www.cancerimagingarchive.net/collection/lidc-idri/) together with its radiologist annotations. `data/processed/lidc_real` is never read or used as a fallback. That older directory has only aggregated scores and cannot preserve the individual diagnostic ratings or segmentations required by the project hypothesis.

The real-data adapter uses only the official optional `pylidc==0.2.3` release. `pylidc` exposes the annotation database derived from the LIDC XML annotations and locates the corresponding DICOM series. Phase 2 verifies that every resolved DICOM directory is beneath the explicitly supplied raw root and checks a DICOM header per series. The checker does not claim to hash the source collection or independently prove raw XML byte identity; provenance labels the verification level `identity_and_header_inventory`.

The frozen Phase 0 manifests remain authoritative: 1,010 patients, 1,018 CT series, and train/validation/test patient counts of 707/152/151. Every nodule inherits its split from the frozen `patient_id + SeriesInstanceUID` reconciliation. There is no sample-level or random splitting, and multiple series belonging to one patient cannot cross splits.

## Provider and physical-nodule identity

Core logic consumes provider-neutral `ScanRecord`, `AnnotationRecord`, `NoduleClusterRecord`, `ConsensusVolumeRecord`, and `DicomSliceRecord` values. Automated tests use `InMemoryLIDCProvider`; real runs use a lazy `PylidcProvider`. Database primary keys and reading-session positions are not research identifiers.

Physical annotations are grouped with [`Scan.cluster_annotations()`](https://pylidc.github.io/scan.html), called with every relevant value explicit:

```yaml
metric: min
tol: null
factor: 0.9
min_tol: 0.1
```

The provider also records and enforces `pylidc==0.2.3`; it does not substitute `pylidc-updated` or another fork.

### Official pylidc compatibility on modern NumPy

Official pylidc 0.2.3 uses the removed `np.int` and `np.bool` aliases inside the clustering/consensus call graph. Phase 2 does not edit or vendor pylidc and does not mutate the process-wide NumPy module. Immediately before a real official clustering or consensus call, the provider acquires a process-wide reentrant lock and temporarily replaces only the `np` bindings inside the official `pylidc.Contour` and `pylidc.Annotation` modules with a delegating proxy. The proxy supplies `int → Python int` and `bool → Python bool` only when each name is absent from the installed NumPy namespace. All other attributes delegate to the original NumPy module, and the exact original module bindings are restored in `finally`.

`np.float` is intentionally not supplied: it occurs in unrelated pylidc visualization code, not in the Phase 2 clustering or consensus paths. Compatibility setup is verified transactionally during provider initialization. Setup or restoration failure raises `ProviderCompatibilityError`; it cannot be downgraded to a malformed annotation or missing mask. Provenance records whether the adapter was required, the exact aliases supplied, and confirms that neither algorithms, NumPy, nor pylidc source files were modified.

Official pylidc also imports the legacy `pkg_resources` API without declaring its provider as a runtime dependency. The LIDC extra and `requirements-phase2-pylidc.txt` therefore pin `setuptools>=68,<81`; setuptools 81 and later no longer provide that API. This is an environment dependency only and does not modify pylidc source or behavior.

Each annotation signature hashes the patient, study and series identifiers; slice index, SOP UID and z position; inclusion flag; canonicalized contour points; and malignancy rating. Polygon start position and direction are canonicalized. A physical nodule SHA-256 then hashes the sorted annotation hashes, making identity independent of provider iteration and annotation order. The human identifier is `<patient_id>__nodule-<first-16-hash-chars>`. Malformed annotations receive deterministic diagnostic signatures and remain represented with exclusion reasons.

Annotation order within one nodule is the lexicographic order of those annotation hashes. IDs such as `local-01-...` describe only this local deterministic order. They are not global radiologist identities.

## Diagnostic targets

Individual malignancy values follow the documented [`pylidc.Annotation.malignancy`](https://pylidc.github.io/annotation.html) integer scale 1–5. Missing values are counted; non-integral or out-of-range values are rejected and recorded. Valid ratings are retained as stable JSON rather than averaged away.

For `R` valid local ratings, Phase 2 stores:

- each score and its mapped class (`1/2 → benign`, `3 → indeterminate`, `4/5 → malignant`);
- exact score sum and unrounded mean;
- `q_benign`, `q_indeterminate`, and `q_malignant`, the mapped-class proportions;
- normalized disagreement `-Σ q_c log(q_c) / log(3)`;
- cumulative ordinal data targets `t_k = mean(I(score > k))` for `k=1..4`.

These ordinal and disagreement quantities are data fields only. No corresponding network head or loss is implemented.

The paper-aligned hard-label view uses integer arithmetic:

- `2 * score_sum < 5 * R`: benign;
- `score_sum == 3 * R`: indeterminate;
- `2 * score_sum > 7 * R`: malignant;
- otherwise: `boundary_zone`.

Thus exact means 2.5 and 3.5 are boundary values, while exact 3.0 is indeterminate. Boundary samples remain in the canonical ledger and are merely ineligible for the paper-aligned primary export.

No four-reader assumption is made. The ledger separately records annotation count, valid-rating count, missing-rating count, multi-reader status, exact-four-valid-reader status, and a strict-four-reader candidate flag. The final minimum-reader rule is intentionally not frozen in Phase 2.

## Spatial targets and representative 2D slice

The provider invokes [`pylidc.utils.consensus`](https://pylidc.github.io/utils.html) with explicit `clevel`, `pad`, and `ret_masks=True`. Its common-frame individual 3D masks are reordered to match the stable local annotation order. Phase 2 computes the float32 soft mask as their arithmetic mean and keeps a diagnostic `>= 0.5` binary mask; the binary view never replaces the soft values.

The representative axial slice is selected from the 3D soft mask by maximum summed area. `numpy.argmax` supplies the required lowest common-frame index on an exact tie. The ledger records the common/full index relationship, SOPInstanceUID, z position, and rule version. Only that slice's aligned individual masks and soft mask are stored in the canonical deterministic `.npz`; the model input remains 2D.

## Canonical extraction versus ROI rendering

Canonical extraction preserves identities, ratings, targets, source geometry, selected-slice masks, source status, and exclusion reasons without committing to an image crop or HU window. Rendering is a separate operation over that same canonical ledger.

The renderer:

1. revalidates the frozen identity and canonical mask checksum;
2. loads the recorded DICOM slice and verifies patient/study/series/SOP identity;
3. applies `HU = stored_pixel * RescaleSlope + RescaleIntercept`;
4. clips to explicitly supplied HU endpoints and maps to `[0,1]`;
5. derives a square, label- and split-independent crop from the union of individual masks, with configured context, minimum side, and constant HU padding;
6. applies identical crop geometry to image and masks;
7. uses bilinear image resizing and nearest-neighbor individual-mask resizing;
8. recomputes the rendered soft mask from the aligned resized individual masks;
9. writes lossless float32 `.npy` images and deterministic mask `.npz` files at 64×64 or 96×96.

The paper reports HU processing but not reproducible window endpoints or a complete crop extent. Consequently no HU endpoints are labeled paper-reported, both endpoints are mandatory at the render CLI, and all HU/crop values are provenance-tracked. The 64×64 output is paper-aligned in resolution; it is not claimed as an exact reconstruction. The same canonical nodules and frozen splits feed both 64×64 and 96×96 renderings.

## Source modes and generated records

`strict_full` requires complete frozen patient/series coverage and rejects duplicate series, unknown or mismatched identities, malformed patient IDs, unreadable series, missing series/patients, or unavailable annotations. `partial_development` reports the same defects but always records `final_dataset_eligible=false`. It cannot silently become the final research dataset.

Inventory writes source counts, annotated-series and annotation counts, physical-cluster counts, reader-count distributions, score-fraction and hard-label distributions, boundary/four-reader/multi-reader counts, malformed/missing counts, and rejection reasons. A count of 4,095 is never enforced or treated as a pass criterion.

Canonical extraction writes:

- `canonical_nodule_ledger.csv`;
- deterministic `masks/<nodule_sha256>.npz` artifacts;
- `leakage_check.json`;
- `provenance.json`, `run_configuration.json`, `checksums.json`, and `COMPLETED.json`.

Generated datasets live under ignored `data/processed/consendxformer_phase2/` (or another explicit ignored runtime path). A completed directory cannot be overwritten. `--resume` accepts only an incomplete extraction with the identical configuration hash, verifies existing ledger/mask/provenance content, and preserves the original provenance timestamp.

Provenance includes the repository Git state, frozen manifest bundle hash, source mode/root and observed counts, dependency versions, all clustering/consensus/slice/crop/HU settings, output resolution, and effective configuration hash. Checksums cover small records and deterministic inventory digests for generated artifacts. They are never described as raw-source byte hashes.

## Real-source setup and commands

Use a dedicated compatible environment for real data:

```text
python -m pip install -r requirements-phase2-pylidc.txt
```

Configure the official pylidc `.pylidcrc` DICOM path as documented by pylidc. Pass that same LIDC root explicitly as `--raw-root`; the adapter rejects resolved series outside it. Start with the readiness checker and inventory. Do not launch full extraction until their reports have been audited.

```text
python scripts/20_check_lidc_source.py --raw-root <LIDC-IDRI-root> --mode strict_full --output-json artifacts/phase2/runtime/source_readiness.json

python scripts/21_inventory_lidc.py --raw-root <LIDC-IDRI-root> --mode strict_full --output-dir artifacts/phase2/runtime/inventory

python scripts/22_extract_lidc_nodules.py --raw-root <LIDC-IDRI-root> --mode strict_full --output-dir data/processed/consendxformer_phase2

python scripts/23_render_lidc_rois.py --raw-root <LIDC-IDRI-root> --mode strict_full --canonical-ledger data/processed/consendxformer_phase2/canonical_nodule_ledger.csv --output-dir data/processed/consendxformer_phase2_render64 --resolution 64 --hu-min <explicit-min> --hu-max <explicit-max> --hu-provenance user_supplied

python scripts/24_export_baseline_metadata.py --rendered-ledger data/processed/consendxformer_phase2_render64/rendered_rois.csv --rendered-root data/processed/consendxformer_phase2_render64 --view paper_aligned_available_reader_mean --output-csv data/processed/consendxformer_phase2_render64/baseline_metadata.csv
```

The exporter also provides a distinct `paper_aligned_strict_four_reader` candidate view; it is never silently substituted for the available-reader view.

After installing the exact official runtime, its bundled annotation database can exercise the real clustering and consensus algorithms without accessing DICOM pixels:

```text
python scripts/25_smoke_pylidc_runtime.py
```

This command uses `PylidcProvider(raw_root=None)`, reports NumPy/pylidc and compatibility metadata, and prints `PYLIDC_RUNTIME_SMOKE=PASS` only after validating real clustered annotations and aligned non-empty individual/soft masks. It explicitly reports that DICOM/source readiness was not tested. A missing official dependency produces `NOT_RUN`; an installed but failing runtime produces `FAIL` with the exact exception.

## Verification

The tests use synthetic provider records only and require neither internet, GPU, LIDC files, pydicom, nor pylidc:

```text
python -m pytest -q
python -m compileall -q src scripts
```

They cover frozen-manifest/config immutability, split propagation and leakage, exact target rules, stable identity, aligned masks and representative slices, deterministic artifacts, DICOM/HU/crop behavior, strict/partial reconciliation, explicit cohort export, protected paths, and the provider call contract.

## Limitations requiring later audit

- A real source inventory/extraction was not run as part of repository implementation. Actual cluster, reader-count, label, and boundary distributions remain unknown until an audited source is available.
- Exact paper HU endpoints and crop extent remain unspecified. They must be selected on independent methodological grounds and recorded as project/user supplied, never tuned on the test set.
- No output count is forced to 4,095, and Phase 2 does not claim exact reproduction of that cohort.
- Official `pylidc==0.2.3` is old and optional. Compatibility with the real execution environment must be demonstrated before a full run; no fork is approved automatically.
- Source verification checks identifiers and representative DICOM headers, not every raw file's cryptographic digest or independent XML byte equivalence.
- Reader/session order is not treated as a cross-patient identity, and the final minimum-reader inclusion rule remains deliberately unresolved.

Methodological context for the 2D paper-aligned view is the [LungDxFormer article](https://doi.org/10.1038/s41598-026-50449-7). This reference does not resolve the missing HU-window or crop details above.
