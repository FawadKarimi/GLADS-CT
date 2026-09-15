"""Additive, descriptive Phase 4 evidence exports. Never train or select models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_evaluation import CLASS_ORDER
from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .phase4_ablation_contracts import ID_FIELDS, identity_rows
from .phase4_ablation_runner import (
    assert_supervision, guard_new_namespace, load_ablation_source, reference_record,
)
from .phase4_boundary import (
    BOUNDARY_LIMITATIONS, load_repository_phase4_boundary_source,
    assert_boundary_source_snapshot_unchanged, _phase2_mean_boundary,
    BOUNDARY_EXCLUSION_FIELDS, BOUNDARY_EVALUATED_POPULATION, BOUNDARY_SOURCE_VALIDITY_NOTE,
)
from .phase4_evaluation_metrics import (
    UNCERTAINTY_DEFINITION, RISK_COVERAGE_DEFINITION, predictive_entropy,
    spearman_association, high_low_disagreement_comparison, risk_coverage_curve,
    _descriptive_group,
)
from .phase4_evaluation_runner import (
    PROJECT_ROOT, FINAL_SEEDS, DEFAULT_EVALUATION_CONFIG, _probabilities, _test_disagreement,
    _training_disagreement_threshold, assert_source_snapshot_unchanged,
    load_phase4_evaluation_config, load_phase4_final_prediction_source,
    validate_final_source_descriptors, validate_seeded_test_cohorts,
    validate_evaluator_repository, create_evaluation_layout,
    aggregate_seed_records,
)
from .phase4_posthoc import load_frozen_final_best_model, predict_validated_boundary_cohort

NAMESPACE = PROJECT_ROOT / "artifacts/phase4/evidence_gaps"
WARNING = (
    "Exploratory post-hoc analysis; no causal superiority claim, model selection, "
    "training or tuning. Repeated seeds are not independent observations. "
    "Nodules can share patients; summaries and paired differences are descriptive, "
    "without independence-based p-values or confidence intervals."
)
BOUNDARY_WARNING = (
    "Descriptive stress testing ONLY: boundary patients may overlap training patients. "
    "This is NOT an independent held-out test set and must not be used for model "
    "selection or primary test headlines. No primary hard targets are assigned."
)
THRESHOLD_PROVENANCE = (
    "median disagreement_entropy of the frozen primary training split; "
    "computed without predictions or test outcomes"
)


def _exact_seeds(sources, key):
    seeds = [key(source) for source in sources]
    if len(seeds) != 3 or any(type(seed) is not int for seed in seeds) or sorted(seeds) != list(FINAL_SEEDS):
        raise ValueError("Exactly seeds 42/43/44 are required, once each")
    return dict(zip(seeds, sources))


def _ordered(frame):
    # Validate before sorting: never coerce missing/invalid identifiers to text.
    identity_rows(frame, label="true_class")
    return frame.sort_values(list(ID_FIELDS), kind="mergesort").reset_index(drop=True)


def uncertainty_comparison(full_frames, off_frames, supervision):
    """Replay saved probabilities; pool MEAN ENTROPY per nodule, not 3N rows."""
    cohort = validate_seeded_test_cohorts(full_frames, off_frames)
    threshold = _training_disagreement_threshold(supervision)
    rows, per_seed, curves = [], [], {}
    for seed in FINAL_SEEDS:
        for model, frames in (("full", full_frames), ("disagreement_weighting_off", off_frames)):
            frame = _ordered(frames[seed])
            probabilities = _probabilities(frame)
            # Existing helper compares Series dtypes as well as values. Normalize
            # only validated textual labels on private copies, not historical code.
            metadata = supervision.copy()
            frame["true_class"] = frame["true_class"].astype(object)
            metadata["label"] = metadata["label"].astype(object)
            disagreement = _test_disagreement(frame, metadata)
            entropy = predictive_entropy(probabilities)
            groups = high_low_disagreement_comparison(
                probabilities, disagreement, threshold=threshold,
                threshold_provenance=THRESHOLD_PROVENANCE,
            )
            identifiers = ["|".join(str(row[field]) for field in
                           ("patient_id", "series_instance_uid", "nodule_id"))
                           for row in frame.to_dict("records")]
            curve = {ranking: risk_coverage_curve(
                frame["true_class"].tolist(), probabilities, ranking=ranking,
                identifiers=identifiers,
            ) for ranking in ("entropy", "confidence")}
            curves[f"{model}/{seed}"] = curve
            per_seed.append({"model": model, "seed": seed,
                             "spearman": spearman_association(entropy, disagreement),
                             "high_low_disagreement": groups,
                             **{f"{k}_aurc": v["aurc"] for k, v in curve.items()}})
            for index, record in enumerate(frame.to_dict("records")):
                rows.append({**record, "model": model, "seed": seed,
                             "predictive_entropy_nats": float(entropy[index]),
                             "confidence": float(probabilities[index].max()),
                             "disagreement_entropy": float(disagreement[index]),
                             "disagreement_group": groups["groups"][index],
                             "disagreement_threshold": threshold})
    raw = pd.DataFrame(rows)
    keys = list(ID_FIELDS)
    pooled = raw.groupby(["model", *keys], sort=True, as_index=False).agg(
        mean_predictive_entropy_nats=("predictive_entropy_nats", "mean"),
        disagreement_entropy=("disagreement_entropy", "first"),
        seed_count=("seed", "nunique"),
    )
    pooled["disagreement_group"] = np.where(pooled.disagreement_entropy >= threshold, "high", "low")
    pooled_associations = {model: spearman_association(
        group.mean_predictive_entropy_nats, group.disagreement_entropy,
    ) for model, group in pooled.groupby("model", sort=True)}
    paired = raw.loc[raw.model == "full"].merge(
        raw.loc[raw.model == "disagreement_weighting_off"],
        on=[*keys, "seed"], suffixes=("_full", "_off"), validate="one_to_one",
    )
    paired["entropy_delta_full_minus_off"] = paired.predictive_entropy_nats_full - paired.predictive_entropy_nats_off
    paired["confidence_delta_full_minus_off"] = paired.confidence_full - paired.confidence_off
    for name in CLASS_ORDER:
        paired[f"prob_{name}_delta_full_minus_off"] = paired[f"prob_{name}_full"] - paired[f"prob_{name}_off"]
    deltas = [column for column in paired if column.endswith("_delta_full_minus_off")]
    paired_nodules = paired.groupby(keys, as_index=False, sort=True)[deltas].mean()
    paired_seed = []
    for seed in FINAL_SEEDS:
        records = {r["model"]: r for r in per_seed if r["seed"] == seed}
        paired_seed.append({"seed": seed,
            **{column: float(paired.loc[paired.seed == seed, column].mean()) for column in deltas},
            **{f"{ranking}_aurc_delta_full_minus_off": records["full"][f"{ranking}_aurc"] -
               records["disagreement_weighting_off"][f"{ranking}_aurc"] for ranking in ("entropy", "confidence")}})
    summary = {
        "cohort": cohort, "unique_nodules": len(paired_nodules),
        "unique_patients": int(raw.patient_id.nunique()),
        "seed_nodule_pairs": len(paired), "warning": WARNING,
        "uncertainty_definition": UNCERTAINTY_DEFINITION,
        "risk_coverage_definition": RISK_COVERAGE_DEFINITION,
        "threshold": threshold, "threshold_provenance": THRESHOLD_PROVENANCE,
        "per_seed": per_seed, "pooled_spearman": pooled_associations,
        "pooled_high_low": {model: {
            name: _descriptive_group(group.loc[group.disagreement_group == name,
                "mean_predictive_entropy_nats"].to_numpy(dtype=float)) for name in ("low", "high")
        } for model, group in pooled.groupby("model", sort=True)},
        "aurc_seed_summary": {model: aggregate_seed_records(
            [row for row in per_seed if row["model"] == model],
            metric_fields=("entropy_aurc", "confidence_aurc"))
            for model in ("full", "disagreement_weighting_off")},
        "pooling": "One mean predictive entropy across seeds per nodule, NOT entropy of mean probabilities; descriptive only.",
        "paired_per_seed": paired_seed,
        "paired_nodule_mean_differences": {column: float(paired_nodules[column].mean()) for column in deltas},
        "risk_coverage_pooling": "Per-seed curves only; no concatenated seed-nodule risk curve or ensemble substitution.",
    }
    return summary, {"uncertainty_per_nodule_per_seed.csv": raw,
                     "uncertainty_per_nodule_seed_mean.csv": pooled,
                     "paired_per_nodule_per_seed.csv": paired,
                     "paired_per_nodule_seed_mean.csv": paired_nodules}, curves


def boundary_summary(results, supervision):
    """Describe validated boundary predictions without constructing hard labels."""
    by_seed = _exact_seeds(results, lambda result: result["seed"])
    keys = ["patient_id", "nodule_id", "series_instance_uid", "split", "boundary_stratum",
            "mean_malignancy", "malignancy_score_sum", "n_valid_malignancy_ratings"]
    frames, reference = [], None
    for seed in FINAL_SEEDS:
        result = by_seed[seed]
        frame = pd.DataFrame(result["rows"])
        if frame.empty or {"true_class", "label"} & set(frame):
            raise ValueError("Boundary predictions must be nonempty and have no primary targets")
        if frame.duplicated(["patient_id", "nodule_id", "series_instance_uid"]).any():
            raise ValueError("Duplicate boundary identity")
        identities = frame[keys].to_dict("records")
        for row in identities:
            member, region = _phase2_mean_boundary(float(row["mean_malignancy"]))
            if not member or region != row["boundary_stratum"]:
                raise ValueError("Prediction violates the authoritative Phase 2 boundary definition")
        if reference is not None and identities != reference:
            raise ValueError("Boundary cohort/order differs across seeds")
        reference = identities
        probabilities = _probabilities(frame)
        frame["predictive_entropy_nats"] = predictive_entropy(probabilities)
        if frame.predicted_class.tolist() != [CLASS_ORDER[int(index)] for index in probabilities.argmax(axis=1)]:
            raise ValueError("Boundary predicted class differs from probabilities")
        frame["confidence"] = probabilities.max(axis=1)
        frame["seed"] = seed
        frame["experiment_id"] = result["experiment_id"]
        frame["checkpoint_sha256"] = result["checkpoint_sha256"]
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    descriptive = []
    for (seed, region), group in raw.groupby(["seed", "boundary_stratum"], sort=True):
        descriptive.append({"seed": int(seed), "boundary_stratum": region,
            "n_nodules": len(group), "n_patients": int(group.patient_id.nunique()),
            **{f"predicted_{name}_count": int((group.predicted_class == name).sum()) for name in CLASS_ORDER},
            **{f"mean_prob_{name}": float(group[f"prob_{name}"].mean()) for name in CLASS_ORDER},
            "mean_confidence": float(group.confidence.mean()), "median_confidence": float(group.confidence.median()),
            "mean_entropy_nats": float(group.predictive_entropy_nats.mean()),
            "median_entropy_nats": float(group.predictive_entropy_nats.median())})
    values = [*[f"prob_{name}" for name in CLASS_ORDER], "confidence", "predictive_entropy_nats"]
    pooled = raw.groupby(keys, as_index=False, sort=True)[values].mean()
    patients = set(pooled.patient_id)
    overlap = {}
    for split in ("train", "val", "test"):
        shared = sorted(patients & set(supervision.loc[supervision.split == split, "patient_id"]))
        overlap[split] = {"patient_ids": shared, "patient_count": len(shared),
                          "boundary_nodule_count": int(pooled.patient_id.isin(shared).sum())}
    aggregate = []
    for region, group in pooled.groupby("boundary_stratum", sort=True):
        aggregate.append({"boundary_stratum": region, "n_nodules": len(group),
                          "n_patients": int(group.patient_id.nunique()),
                          **{f"mean_{column}": float(group[column].mean()) for column in values}})
    return {"warning": BOUNDARY_WARNING, "limitations": list(BOUNDARY_LIMITATIONS),
            "evaluated_population": BOUNDARY_EVALUATED_POPULATION,
            "source_validity_note": BOUNDARY_SOURCE_VALIDITY_NOTE,
            "n_nodules": len(pooled), "n_patients": len(patients),
            "patient_overlap_with_primary": overlap, "per_seed_regions": descriptive,
            "boundary_patients_absent_from_primary": sorted(patients - set(supervision.patient_id)),
            "original_split_nodule_counts": {str(key): int(value) for key, value in pooled.split.value_counts().items()},
            "aggregate_regions": aggregate,
            "pooling": "One mean per nodule across three seeds; mean entropy/confidence, not ensemble entropy/confidence.",
            "primary_metrics_computed": False, "independent_held_out_test": False}, {
                "boundary_predictions_per_seed.csv": raw,
                "boundary_per_nodule_seed_mean.csv": pooled,
                "boundary_regions_per_seed.csv": pd.DataFrame(descriptive),
                "boundary_regions_aggregate.csv": pd.DataFrame(aggregate)}


def run_evidence_gap(*, analysis, full_source_dirs, off_source_dirs=(), output_root=None):
    """Validate historical sources, export a NEW bundle, recheck, then complete."""
    if analysis not in {"uncertainty", "boundary"}:
        raise ValueError("Unknown evidence gap")
    if len(full_source_dirs) != 3 or (analysis == "uncertainty" and len(off_source_dirs) != 3):
        raise ValueError("Exactly three sources per requested model are required")
    if analysis == "boundary" and off_source_dirs:
        raise ValueError("Boundary analysis accepts full models only")
    git = validate_evaluator_repository(PROJECT_ROOT)
    config = load_phase4_evaluation_config(DEFAULT_EVALUATION_CONFIG)
    full = _exact_seeds([load_phase4_final_prediction_source(path) for path in full_source_dirs],
                        lambda source: source.descriptor["seed"])
    validate_final_source_descriptors([source.descriptor for source in full.values()])
    snapshots, records = {}, []
    protected = []
    def record_source(source, identity):
        for path, digest in source.source_snapshot.items():
            if path in snapshots and snapshots[path] != digest:
                raise ValueError("Conflicting source snapshots")
            snapshots[path] = digest
        protected.append(source.source_dir)
        records.append({"source_dir": str(source.source_dir), "identity": identity,
                        "snapshot": dict(source.source_snapshot),
                        "best_checkpoint_sha256": sha256_file(source.source_dir / "checkpoints/best.pt"),
                        "test_predictions_sha256": sha256_file(source.source_dir / "predictions/test.csv")})
    for seed in FINAL_SEEDS:
        record_source(full[seed], full[seed].descriptor)
        assert_supervision(full[seed].supervision, full[42].descriptor)
    boundary = None
    off = {}
    if analysis == "uncertainty":
        off = _exact_seeds([load_ablation_source(path) for path in off_source_dirs], lambda source: source.identity["seed"])
        for seed in FINAL_SEEDS:
            source = off[seed]
            if source.identity["variant"] != "disagreement_weighting":
                raise ValueError("Only the completed disagreement_weighting-OFF ablation is allowed")
            if source.identity["full_model_reference"] != reference_record(full[seed]):
                raise ValueError("Ablation full-model reference differs from supplied full source")
            assert_supervision(source.supervision, full[seed].descriptor)
            record_source(source, source.identity)
    else:
        boundary = load_repository_phase4_boundary_source(project_root=PROJECT_ROOT)
        for path, digest in boundary.source_snapshot.items():
            if path in snapshots and snapshots[path] != digest:
                raise ValueError("Conflicting boundary snapshot")
            snapshots[path] = digest
        protected.extend(Path(path).parent for path in boundary.source_snapshot)
    identity = {"schema_version": 1, "analysis": analysis, "git": git,
                "seeds": list(FINAL_SEEDS), "sources": records, "device": "cpu",
                "evaluation_config": config, "warning": WARNING,
                "input_snapshot": snapshots,
                "boundary_provenance": None if boundary is None else dict(boundary.provenance)}
    evaluation_id = f"phase4-evaluation-{sha256_json(identity)[:16]}"
    root = guard_new_namespace(NAMESPACE / analysis if output_root is None else output_root)
    # Custom output roots must still use the distinct analysis namespace.
    if not root.is_relative_to(NAMESPACE.resolve()):
        raise ValueError("Output must remain under artifacts/phase4/evidence_gaps")
    guard_new_namespace(root / evaluation_id)
    layout = create_evaluation_layout(root, evaluation_id, protected_sources=protected)
    assert_source_snapshot_unchanged(snapshots)
    curves = {}
    if analysis == "uncertainty":
        summary, tables, curves = uncertainty_comparison(
            {seed: full[seed].predictions for seed in FINAL_SEEDS},
            {seed: off[seed].predictions["test"] for seed in FINAL_SEEDS}, full[42].supervision)
    else:
        import torch
        deterministic, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        results = []
        try:
            for seed in FINAL_SEEDS:
                assert_source_snapshot_unchanged(snapshots)
                frozen = load_frozen_final_best_model(full[seed].source_dir, device="cpu")
                if frozen.source.descriptor != full[seed].descriptor or frozen.source.source_snapshot != full[seed].source_snapshot:
                    raise ValueError("Reloaded frozen full source differs")
                results.append(predict_validated_boundary_cohort(frozen, boundary))
                del frozen
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        summary, tables = boundary_summary(results, full[42].supervision)
        accounting = boundary.provenance["source_validity_accounting"]
        if summary["n_nodules"] != accounting["eligible_subset_count"]:
            raise ValueError("Boundary summary count differs from the source-valid subset")
        summary["source_validity_accounting"] = accounting
        tables["boundary_source_invalid_exclusions.csv"] = pd.DataFrame(
            accounting["exclusion_records"], columns=BOUNDARY_EXCLUSION_FIELDS)
    paths = [atomic_write_json(layout.output_dir / "evaluation.json", {
        "evaluation_id": evaluation_id, "analysis": analysis, "summary": summary,
        "training_performed": False, "model_selection_performed": False,
        "causal_superiority_claimed": False, "exploratory_post_hoc": True}),
        atomic_write_json(layout.provenance_dir / "provenance.json", identity),
        atomic_write_text(layout.output_dir / "README.md", WARNING + "\n\n" +
                          (BOUNDARY_WARNING + "\n\n" + BOUNDARY_SOURCE_VALIDITY_NOTE + "\n" +
                           f"Full boundary membership: {accounting['full_boundary_membership_count']}; "
                           f"source-invalid exclusions: {accounting['source_invalid_exclusion_count']}; "
                           f"evaluated source-valid subset: {accounting['eligible_subset_count']}."
                           if boundary is not None else summary["pooling"]) + "\n")]
    for name, table in tables.items():
        paths.append(atomic_write_csv(layout.tables_dir / name, table))
    if curves:
        paths.append(atomic_write_json(layout.curves_dir / "risk_coverage.json", curves))
    def recheck():
        assert_source_snapshot_unchanged(snapshots)
        if boundary is not None:
            assert_boundary_source_snapshot_unchanged(boundary.source_snapshot)
        if validate_evaluator_repository(PROJECT_ROOT) != git:
            raise ValueError("Evaluator repository changed")
    recheck()
    files = {path.relative_to(layout.output_dir).as_posix(): sha256_file(path) for path in paths}
    manifest = atomic_write_json(layout.output_dir / "artifact_manifest.json", {
        "schema_version": 1, "evaluation_id": evaluation_id, "files": files,
        "bundle_sha256": sha256_json(files), "source_artifacts_unchanged": True,
        "training_performed": False, "exploratory_post_hoc": True})
    for relative, digest in files.items():
        if sha256_file(layout.output_dir / relative) != digest:
            raise ValueError("Output changed before completion")
    recheck()
    completion = atomic_write_json(layout.output_dir / "COMPLETED.json", {
        "schema_version": 1, "evaluation_id": evaluation_id,
        "artifact_manifest_sha256": sha256_file(manifest), "training_performed": False,
        "source_artifacts_unchanged": True, "exploratory_post_hoc": True})
    try:
        recheck()
    except BaseException:
        completion.unlink(missing_ok=True)
        raise
    return layout.output_dir


def evidence_gap_cli(argv=None):
    parser = argparse.ArgumentParser(description="Analysis-only Phase 4 evidence gaps; no training or tuning.")
    parser.add_argument("analysis", choices=("uncertainty", "boundary"))
    parser.add_argument("--full-source", action="append", required=True)
    parser.add_argument("--off-source", action="append", default=[])
    parser.add_argument("--output-root", default=None, help="Optional new directory under artifacts/phase4/evidence_gaps")
    args = parser.parse_args(argv)
    print(run_evidence_gap(analysis=args.analysis, full_source_dirs=args.full_source,
                           off_source_dirs=args.off_source, output_root=args.output_root))
    return 0
