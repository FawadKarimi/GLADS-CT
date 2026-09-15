"""Separate, opt-in analysis of completed ablations against immutable full models."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_json, atomic_write_csv
from .phase4_ablations import FINAL_SEEDS
from .phase4_ablation_contracts import NAMESPACE, ID_FIELDS, PROBS, identity_rows, prediction_metrics, execution_protocol
from .phase4_ablation_runner import (ROOT, read_json, load_ablation_source, frozen_bundle, revalidate_inputs,
    assert_supervision, reference_record, _require_project_state, guard_new_namespace, guard_image_paths)
from .phase4_evaluation_runner import load_phase4_final_prediction_source, _safe_source_member, snapshot_source_files, assert_source_snapshot_unchanged
from .phase4_predictions import load_phase4_prediction_csv
from .phase4_bootstrap import bootstrap_paired_differences
from .phase4_attention_evidence import (array_record, map_evidence, paired_alignment_bootstrap, write_map_archive,
                                      supplementary_protocol, FLAGGED_NODULE)
from .phase4_attention_runtime import _saved_replay


def load_fixed_evidence(directory, sources):
    """Reuse the already-frozen prior and full-model arrays, never refit a prior."""
    directory = Path(directory).resolve()
    manifest, completion = read_json(directory / "artifact_manifest.json"), read_json(directory / "COMPLETED.json")
    expected_id = execution_protocol()["supplementary_reference_id"]
    if (manifest.get("evaluation_id") != expected_id or completion.get("evaluation_id") != expected_id
        or completion.get("artifact_manifest_sha256") != sha256_file(directory / "artifact_manifest.json")
        or completion.get("replay_consistent") is not True or completion.get("comparison_interpretation_permitted") is not True
        or manifest.get("replay_consistent") is not True or manifest.get("bundle_sha256") != sha256_json(manifest["files"])):
        raise ValueError("Completed, replay-consistent frozen supplementary reference required")
    snapshot = snapshot_source_files([directory / "artifact_manifest.json", directory / "COMPLETED.json"])
    for relative, digest in manifest["files"].items():
        canonical, member = _safe_source_member(directory, relative)
        if canonical != relative or sha256_file(member) != digest:
            raise ValueError("Frozen supplementary evidence checksum changed")
        snapshot[str(member)] = digest
    provenance = read_json(directory / "provenance/provenance.json")
    evaluation = read_json(directory / "evaluation.json")
    if evaluation["protocol"] != supplementary_protocol() or evaluation.get("replay_consistent") is not True:
        raise ValueError("Supplementary evidence protocol/replay differs")
    records = [r for r in provenance["source_experiments"] if r["model"] == "ConsenDxFormer"]
    if sorted(r["seed"] for r in records) != list(FINAL_SEEDS):
        raise ValueError("Supplementary full-model seeds are not exact")
    for record in records:
        if record["descriptor"] != sources[record["seed"]].identity["full_model_reference"]["descriptor"]:
            raise ValueError("Supplementary arrays bind another full-model reference")
    archives = {}
    for record in provenance["archives"]:
        name = record["file"]
        canonical, member = _safe_source_member(directory, name)
        if canonical != name or name in archives or snapshot.get(str(member)) != record["sha256"]:
            raise ValueError("Supplementary array ledger is not manifest-bound")
        with np.load(member, allow_pickle=False) as loaded:
            if set(loaded.files) != set(record["arrays"]):
                raise ValueError("Supplementary array keys differ")
            archives[name] = {key: loaded[key].copy() for key in loaded.files}
        if any(array_record(value) != record["arrays"][key] for key, value in archives[name].items()):
            raise ValueError("Supplementary array dtype/shape/checksum differs")
    prior = archives["location_prior.npz"]["prior"]
    if array_record(prior) != provenance["prior"]["array"] or provenance["prior"]["test_masks_used"] is not False:
        raise ValueError("Frozen training-only prior provenance differs")
    training = sources[42].supervision.loc[sources[42].supervision["split"] == "train"]
    expected = sorted(identity_rows(training), key=lambda row: tuple(row[k] for k in ID_FIELDS))
    if provenance["prior"]["training_identities"] != expected or provenance["prior"]["n_nodules"] != 807:
        raise ValueError("Frozen prior is not the exact 807-nodule training aggregate")
    if sha256_json(expected) != provenance["prior"]["training_identities_sha256"]:
        raise ValueError("Frozen prior identity hash differs")
    assert_source_snapshot_unchanged(provenance["input_file_snapshot"])
    snapshot.update(provenance["input_file_snapshot"])
    assert_source_snapshot_unchanged(snapshot)
    return archives, provenance, snapshot


def infer_ablation_maps(source, archive_inputs, *, device):
    import torch
    from .phase4_execution import _model, _build_test_dataset, _loader
    from .phase4_ablation_execution import ExactEvalLoader

    _, lock = frozen_bundle()
    _, readiness, live = revalidate_inputs(source.config, lock)
    assert_supervision(live, source.identity["inputs"])
    if any(readiness.to_dict()[key] != value for key, value in source.identity["inputs"].items()):
        raise ValueError("Ablation analysis readiness differs from trained source")
    frame = source.supervision.loc[source.supervision["split"] == "test"].reset_index(drop=True)
    guard_image_paths(frame)
    ids = pd.DataFrame({key: archive_inputs[key] for key in ID_FIELDS})
    if identity_rows(ids, label="true_class") != identity_rows(frame):
        raise ValueError("Frozen supplementary evidence cohort/order differs")
    model = _model(source.config)
    model.load_state_dict(source.best_record["model_state"], strict=True)
    model.to(device).eval().requires_grad_(False)
    dataset = _build_test_dataset(frame, source.config, ROOT / source.config["data"]["rendered_dir"])
    loader = ExactEvalLoader(_loader(dataset, source.config, seed=source.identity["seed"], shuffle=False), frame)
    attention, logits, probabilities = [], [], []
    position = 0
    with torch.no_grad():
        for batch in loader:
            n = len(batch["image"])
            if (not np.array_equal(batch["image"].numpy(), archive_inputs["ct_crops"][position:position+n])
                or not np.array_equal(batch["soft_consensus_mask"].numpy(), archive_inputs["soft_masks_input"][position:position+n])):
                raise ValueError("Live CT/consensus differs from frozen supplementary arrays")
            output = model(batch["image"].to(device))  # Images only; masks are targets, never model inputs.
            attention.append(output["attention_map"].cpu().numpy().copy())
            logits.append(output["logits"].cpu().numpy().copy())
            probabilities.append(output["probs"].cpu().numpy().copy())
            position += n
    maps, values, probs = np.concatenate(attention), np.concatenate(logits), np.concatenate(probabilities)
    if maps.ndim != 4 or maps.shape[:2] != (len(ids), 1):
        raise ValueError("Ablation must emit native [N,1,H,W] DSA attention")
    replay = _saved_replay(source.predictions["test"], ids, values, probs)
    assert_source_snapshot_unchanged(source.source_snapshot)
    return maps, values, probs, replay


def analyze_ablations(source_dirs, supplementary_source, *, authorize_analysis=False, device="cpu"):
    if authorize_analysis is not True:
        raise ValueError("Separate explicit ablation analysis authorization required")
    if device not in ("cpu", "cuda"):
        raise ValueError("Unsupported analysis device")
    if len(source_dirs) != 3 or len({str(Path(p).resolve()) for p in source_dirs}) != 3:
        raise ValueError("Analysis requires exactly three distinct ablation sources")
    git = _require_project_state(ROOT)
    loaded = [load_ablation_source(path) for path in source_dirs]
    if sorted(s.identity["seed"] for s in loaded) != list(FINAL_SEEDS) or len({s.identity["variant"] for s in loaded}) != 1:
        raise ValueError("Analyze one exact variant across all three seeds; no favorable subset")
    sources = {s.identity["seed"]: s for s in loaded}
    archives, fixed_provenance, snapshot = load_fixed_evidence(supplementary_source, sources)
    for source in loaded:
        snapshot.update(source.source_snapshot)
    identity = {"schema": "phase4_ablation_analysis_v1", "evaluator_commit": git["commit"], "device": device,
                "sources": [s.identity for s in sorted(loaded, key=lambda s: s.identity["seed"])],
                "supplementary_manifest_sha256": sha256_file(Path(supplementary_source) / "artifact_manifest.json"),
                "protocol": execution_protocol()}
    output = ROOT / NAMESPACE / "analysis" / f"phase4-ablation-analysis-{sha256_json(identity)[:16]}"
    guard_new_namespace(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, summaries, archive_records = [], [], []
    inputs, prior = archives["test_inputs.npz"], archives["location_prior.npz"]["prior"]
    ids = pd.DataFrame({key: inputs[key] for key in ID_FIELDS})
    fixed = map_evidence(np.broadcast_to(prior, (len(ids), *prior.shape)), inputs["soft_masks_input"], ids.to_dict("records"))
    if not np.array_equal(fixed["resized_soft_masks"], inputs["soft_masks_native_grid"]) or not np.array_equal(fixed["target_distributions"], inputs["target_distributions"]):
        raise ValueError("Frozen supplementary consensus targets differ from existing metric definition")
    archive_records.append(write_map_archive(output / "test_inputs.npz", inputs))
    archive_records.append(write_map_archive(output / "location_prior.npz", {"prior": prior}))
    for seed in FINAL_SEEDS:
        source = sources[seed]
        assert_source_snapshot_unchanged(snapshot)
        full = load_phase4_final_prediction_source(source.identity["full_model_reference"]["source_dir"])
        if reference_record(full) != source.identity["full_model_reference"]:
            raise ValueError("Full-model source changed during ablation analysis")
        snapshot.update(full.source_snapshot)
        maps, logits, probabilities, replay = infer_ablation_maps(source, inputs, device=device)
        full_maps = archives[f"seed_{seed}_model_arrays.npz"]["cdx_attention"]
        if maps.shape != full_maps.shape or tuple(maps.shape[-2:]) != prior.shape:
            raise ValueError("Native attention grid differs; adaptation is forbidden")
        a = map_evidence(maps, inputs["soft_masks_input"], ids.to_dict("records"))
        b = map_evidence(full_maps, inputs["soft_masks_input"], ids.to_dict("records"))
        archive_records.append(write_map_archive(output / f"seed_{seed}_maps.npz", {
            "ablation_attention": maps, "full_attention": full_maps, "ablation_logits": logits, "ablation_probabilities": probabilities,
            **{key: inputs[key] for key in ID_FIELDS}}))
        classification = {}
        for split in ("val", "test"):
            reference_predictions = load_phase4_prediction_csv(full.source_dir / f"predictions/{split}.csv")
            # Existing paired metric implementation is reused mathematically; its
            # historical B1/CDX parameter names do not label these source models.
            from .phase4_bootstrap import validate_paired_prediction_rows
            validate_paired_prediction_rows(reference_predictions, source.predictions[split], expected_split=split)
            classification[split] = {"full": prediction_metrics(reference_predictions), "ablation": prediction_metrics(source.predictions[split])}
        ci = bootstrap_paired_differences(full.predictions, source.predictions["test"], seed=20260914, n_resamples=10000, confidence_level=.95)
        summaries.append({"seed": seed, "variant": source.identity["variant"], "replay": replay,
            "classification_calibration": classification, "classification_comparison_direction": "ablation_minus_full",
            "classification_difference": ci["point_estimates"]["difference"], "classification_difference_ci": ci["confidence_intervals"]["difference"],
            "classification_bootstrap": ci["bootstrap"], "attention_shape": list(maps.shape),
            "attention_ablation_minus_full": paired_alignment_bootstrap(pd.DataFrame(a["rows"]), pd.DataFrame(b["rows"])),
            "attention_ablation_minus_fixed_prior": paired_alignment_bootstrap(pd.DataFrame(a["rows"]), pd.DataFrame(fixed["rows"]))})
        for model, evidence in (("ablation", a), ("full", b)):
            rows.extend({**row, "model": model, "seed": seed, "variant": source.identity["variant"]} for row in evidence["rows"])
    table = pd.DataFrame(rows)
    atomic_write_csv(output / "alignment_per_nodule.csv", table)
    atomic_write_csv(output / "flagged_case.csv", table.loc[table["nodule_id"] == FLAGGED_NODULE])
    consistent = all(row["replay"]["consistent"] for row in summaries)
    atomic_write_json(output / "analysis.json", {"identity": identity, "seed_results": summaries,
        "replay_consistent": consistent, "comparison_interpretation_permitted": consistent,
        "chronology": execution_protocol()["chronology"], "preregistered": False, "no_cross_variant_loss_comparison": True,
        "prior_unchanged": True, "training_performed": False, "causal_or_population_significance_claimed": False})
    atomic_write_json(output / "provenance.json", {"input_snapshot": snapshot, "archives": archive_records,
        "frozen_prior": fixed_provenance["prior"], "prior_recomputed": False})
    assert_source_snapshot_unchanged(snapshot)
    if _require_project_state(ROOT) != git:
        raise ValueError("Git state changed during ablation analysis")
    files = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    manifest = atomic_write_json(output / "artifact_manifest.json", {"identity": identity, "files": files, "bundle_sha256": sha256_json(files)})
    assert_source_snapshot_unchanged(snapshot)
    marker = atomic_write_json(output / "COMPLETED.json", {"schema": "phase4_ablation_analysis_completion_v1",
        "artifact_manifest_sha256": sha256_file(manifest), "replay_consistent": consistent, "training_performed": False})
    try:
        assert_source_snapshot_unchanged(snapshot)
        if _require_project_state(ROOT) != git:
            raise ValueError("Git state changed before analysis completion")
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    return output
