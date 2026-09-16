"""Synthetic saved-artifact integration. Source-loader stubs never weaken production."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase3_artifacts import atomic_write_json
from consendxformer.phase4_ablation_contracts import identity_rows, execution_protocol
from consendxformer.phase4_attention_evidence import array_record, write_map_archive, supplementary_protocol
from consendxformer.phase4_evaluation_runner import relative_source_snapshot
from consendxformer.phase5_descriptive import CLASS_ORDER, FLAGGED_NODULE, IDENTITY, MODELS, SEEDS, VARIANTS
import consendxformer.phase5_descriptive_runner as runner


def _seal(directory):
    files = {p.relative_to(directory).as_posix(): sha256_file(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name not in ("COMPLETED.json", "artifact_manifest.json")}
    eid = execution_protocol()["supplementary_reference_id"]
    atomic_write_json(directory / "artifact_manifest.json", {"evaluation_id": eid, "files": files,
        "bundle_sha256": sha256_json(files), "replay_consistent": True}, overwrite=True)
    atomic_write_json(directory / "COMPLETED.json", {"evaluation_id": eid,
        "artifact_manifest_sha256": sha256_file(directory / "artifact_manifest.json"),
        "replay_consistent": True, "comparison_interpretation_permitted": True}, overwrite=True)


def _setup(tmp_path, monkeypatch):
    # 84 patients with two nodules, 15 with one; entirely invented test data.
    patient_numbers = [i // 2 if i < 168 else i - 84 for i in range(183)]
    ids = pd.DataFrame({"patient_id": [f"synthetic-p{i:03}" for i in patient_numbers],
        "nodule_id": [FLAGGED_NODULE, *[f"synthetic-n{i:03}" for i in range(1, 183)]],
        "series_instance_uid": [f"synthetic-s{i:03}" for i in patient_numbers],
        "split": ["test"]*183, "true_class": [CLASS_ORDER[i % 3] for i in range(183)]})
    train = ids.iloc[[0]*807].reset_index(drop=True)
    train["patient_id"] = [f"train-p{i:03}" for i in range(807)]
    train["nodule_id"] = [f"train-n{i:03}" for i in range(807)]
    train["series_instance_uid"] = [f"train-s{i:03}" for i in range(807)]
    train["split"] = "train"
    supervision = pd.concat([train, ids], ignore_index=True).rename(columns={"true_class": "label"})
    full, b1, off, all_sources = {}, {}, {}, {}
    for model in MODELS:
        for seed in SEEDS:
            path = tmp_path / "sources" / f"{model}-{seed}"
            (path / "checkpoints").mkdir(parents=True)
            (path / "checkpoints/best.pt").write_bytes(f"SYNTHETIC NOT A CHECKPOINT {model} {seed}".encode())
            frame = ids.copy()
            # Deliberately nontrivial, different hard predictions without a model.
            chosen = np.array([(i + (seed - 42) + MODELS.index(model)) % 3 for i in range(183)])
            probs = np.full((183, 3), .125, dtype=np.float32)
            probs[np.arange(183), chosen] = .75
            logits = np.log(probs).astype(np.float32)
            frame["predicted_class"] = [CLASS_ORDER[c] for c in chosen]
            for j, name in enumerate(CLASS_ORDER):
                frame[f"prob_{name}"] = probs[:, j]
                frame[f"logit_{name}"] = logits[:, j]
            (path / "predictions").mkdir()
            frame.to_csv(path / "predictions/test.csv", index=False)
            atomic_write_json(path / "artifact_manifest.json", {"fixture": "synthetic"})
            snapshot = {str(p.resolve()): sha256_file(p) for p in path.rglob("*") if p.is_file()}
            source = SimpleNamespace(source_dir=path.resolve(), predictions=frame, supervision=supervision,
                source_snapshot=snapshot, descriptor={"seed": seed, "experiment_id": path.name, "model": model})
            if model == "ConsenDxFormer": full[seed] = source
            elif model == "B1": b1[seed] = source
            else:
                variant = model.removesuffix("_off")
                source.identity = {"variant": variant, "seed": seed,
                                   "full_model_reference": runner.reference_record(full[seed])}
                source.predictions = {"test": frame}
                off[(variant, seed)] = source
            all_sources[(model, seed)] = source
    source_snapshot = {k: v for s in [*full.values(), *b1.values()] for k, v in s.source_snapshot.items()}
    inputs = SimpleNamespace(phase4=full, b1=b1, supervision=supervision, source_snapshot=source_snapshot)
    monkeypatch.setattr(runner, "load_saved_prediction_inputs", lambda a, b: inputs)
    monkeypatch.setattr(runner, "load_ablation_source", lambda p: next(s for s in off.values() if s.source_dir == Path(p)))
    monkeypatch.setattr(runner, "validate_evaluator_repository", lambda root: {"commit": "b"*40, "clean": True})
    monkeypatch.setattr(runner, "NAMESPACE", tmp_path / "new-subset-output")
    # Production protocol/proposal constants and cohort counts remain unchanged.
    evidence = tmp_path / "saved-attention"
    masks = np.zeros((183, 8, 8), dtype=np.float32)
    masks[1:, :4, :4] = .5
    masks[0, :2, :2] = .1  # known empty binary target stays positive as a soft mask
    targets = masks / masks.sum(axis=(1, 2), keepdims=True)
    prior = np.full((8, 8), 1/64, dtype=np.float64)
    archived_inputs = {**{k: ids[k].to_numpy(dtype=np.str_) for k in IDENTITY},
        "soft_masks_native_grid": masks, "target_distributions": targets,
        "ct_crops": np.zeros((183, 1, 64, 64), dtype=np.float32),
        "soft_masks_input": np.repeat(np.repeat(masks, 8, axis=1), 8, axis=2)[:, None]}
    archives = [write_map_archive(evidence / "test_inputs.npz", archived_inputs),
                write_map_archive(evidence / "location_prior.npz", {"prior": prior})]
    for seed in SEEDS:
        arrays = {}
        for model, prefix in (("ConsenDxFormer", "cdx"), ("B1", "b1")):
            arrays[f"{prefix}_attention"] = np.full((183, 1, 8, 8), 1/64, dtype=np.float32)
            arrays[f"{prefix}_logits"] = all_sources[(model, seed)].predictions[[f"logit_{c}" for c in CLASS_ORDER]].to_numpy()
            arrays[f"{prefix}_probabilities"] = all_sources[(model, seed)].predictions[[f"prob_{c}" for c in CLASS_ORDER]].to_numpy()
        archives.append(write_map_archive(evidence / f"seed_{seed}_model_arrays.npz", arrays))
    train_ids = sorted(identity_rows(supervision.loc[supervision.split == "train"]), key=lambda r: tuple(r[k] for k in IDENTITY))
    source_records = [{"model": model, "seed": seed, "descriptor": s.descriptor,
                       "source_snapshot": relative_source_snapshot(s)}
                      for (model, seed), s in all_sources.items() if model in MODELS[:2]]
    provenance = {"archives": archives, "source_experiments": source_records,
        "prior": {"array": array_record(prior), "test_masks_used": False, "n_nodules": 807,
                  "training_identities": train_ids, "training_identities_sha256": sha256_json(train_ids)},
        "input_file_snapshot": source_snapshot, "input_file_snapshot_sha256": sha256_json(source_snapshot)}
    atomic_write_json(evidence / "provenance/provenance.json", provenance)
    atomic_write_json(evidence / "provenance/supplementary_protocol.json", supplementary_protocol())
    atomic_write_json(evidence / "evaluation.json", {"protocol": supplementary_protocol(),
        "replay_consistent": True, "comparison_interpretation_permitted": True})
    _seal(evidence)
    kwargs = dict(full_source_dirs=[s.source_dir for s in full.values()], b1_source_dirs=[s.source_dir for s in b1.values()],
                  ablation_source_dirs=[s.source_dir for s in off.values()], attention_source=evidence,
                  authorize_analysis=True)
    return kwargs, all_sources, evidence


def test_subset_end_to_end_zero_forwards_deterministic_exports_and_completion(tmp_path, monkeypatch):
    kwargs, sources, evidence = _setup(tmp_path, monkeypatch)
    import consendxformer.phase4_bootstrap as bootstrap
    import consendxformer.phase4_ablation_analysis as old_analysis
    import consendxformer.phase4_attention_runtime as runtime
    import consendxformer.phase4_posthoc as posthoc
    import consendxformer.phase4_execution as execution
    import consendxformer.phase4_ablation_runner as ablation
    import consendxformer.phase4_evaluation_runner as evaluation
    def forbidden(*args, **kw): raise AssertionError("Forbidden inference/training/old-suite path")
    for module, names in ((bootstrap, ("bootstrap_model_metrics", "bootstrap_paired_differences", "_metric_vector")),
        (old_analysis, ("infer_ablation_maps", "map_evidence", "build_location_prior")),
        (runtime, ("infer_matched_attention",)), (posthoc, ("load_frozen_final_best_model",)),
        (execution, ("_model",)), (ablation, ("run_ablation",)),
        (evaluation, ("compute_saved_prediction_evaluation", "run_phase4_saved_evaluation"))):
        for name in names:
            if hasattr(module, name): monkeypatch.setattr(module, name, forbidden)
    # Intercept every PyTorch Module call as well, not merely one known runner.
    import torch
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    snapshot = {str(p): sha256_file(p) for p in (tmp_path / "sources").rglob("*") if p.is_file()}
    snapshot.update({str(p): sha256_file(p) for p in evidence.rglob("*") if p.is_file()})
    first = runner.run_descriptive(**kwargs)
    result, marker = runner._json(first / "evaluation.json"), runner._json(first / "COMPLETED.json")
    assert marker["scope"] == "pointing_and_accuracy_only" and marker["subset_completed"]
    assert not marker["full_supplemental_completion"] and not marker["formal_tests_computed"]
    assert marker["model_forwards"] == 0 and not marker["training_performed"]
    assert len(marker["unresolved_hypotheses"]) == 15
    assert not any(k in r for r in marker["unresolved_hypotheses"] for k in ("Q", "p_raw", "p_adj", "p_value"))
    assert result["individual_intervals"] == 18 and result["paired_intervals"] == 15
    raw = pd.read_csv(first / "tables/pointing_per_nodule.csv")
    flagged = pd.read_csv(first / "tables/flagged_case.csv")
    assert len(raw) == 1281 and len(flagged) == 7
    assert flagged.hit.eq(0).all() and flagged.empty_target.all()
    assert len(pd.read_csv(first / "tables/pointing_paired_descriptive.csv")) == 9
    with np.load(first / "accuracy_bootstrap.npz", allow_pickle=False) as saved:
        assert saved["draw_indices"].shape == (10000, 99)
        assert saved["replicate_estimates"].shape == (10000, 33)
    second = runner.run_descriptive(**kwargs, output_root=runner.NAMESPACE / "another-new-root")
    assert {p.relative_to(first).as_posix(): sha256_file(p) for p in first.rglob("*") if p.is_file()} == {
        p.relative_to(second).as_posix(): sha256_file(p) for p in second.rglob("*") if p.is_file()}
    runner.assert_source_snapshot_unchanged(snapshot)
    with pytest.raises(FileExistsError): runner.run_descriptive(**kwargs)


@pytest.mark.parametrize("change", ["bytes", "array_metadata", "b1_binding", "full_binding", "training",
    "replay", "permission", "unbound_provenance", "input_snapshot", "archive_identity", "grid", "flagged", "saved_probability", "normalized_target"])
def test_saved_evidence_corruption_fails_before_scoring(tmp_path, monkeypatch, change):
    kwargs, _, evidence = _setup(tmp_path, monkeypatch)
    path = evidence / "provenance/provenance.json"
    provenance = runner._json(path)
    if change == "bytes":
        (evidence / "location_prior.npz").write_bytes(b"corrupted")
    elif change in {"archive_identity", "grid", "flagged", "saved_probability", "normalized_target"}:
        name = "seed_42_model_arrays.npz" if change in {"grid", "saved_probability"} else "test_inputs.npz"
        with np.load(evidence / name, allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        if change == "archive_identity": arrays["patient_id"][0] = "wrong"
        elif change == "grid": arrays["b1_attention"] = np.ones((183, 1, 1, 1))
        elif change == "flagged":
            arrays["soft_masks_native_grid"][0] = .5
            arrays["soft_masks_input"][0] = .5
        elif change == "normalized_target": arrays["soft_masks_native_grid"] = arrays["target_distributions"].copy()
        else: arrays["b1_probabilities"][0] = [.5, .25, .25]
        # Re-seal file and array digests to exercise semantic validation, not just checksums.
        np.savez(evidence / name, **arrays)
        for r in provenance["archives"]:
            if r["file"] == name:
                r["sha256"] = sha256_file(evidence / name)
                r["arrays"] = {k: array_record(v) for k, v in arrays.items()}
    elif change == "array_metadata": provenance["archives"][0]["arrays"]["ct_crops"]["shape"] = [1]
    elif change in {"b1_binding", "full_binding"}:
        record = next(r for r in provenance["source_experiments"] if r["model"] == ("B1" if change == "b1_binding" else "ConsenDxFormer"))
        record["source_snapshot"]["checkpoints/best.pt"] = "0"*64
    elif change == "training": provenance["prior"]["training_identities"][0]["split"] = "test"
    elif change == "input_snapshot": provenance["input_file_snapshot_sha256"] = "0"*64
    elif change in {"replay", "permission"}:
        evaluation = runner._json(evidence / "evaluation.json")
        evaluation["replay_consistent" if change == "replay" else "comparison_interpretation_permitted"] = False
        atomic_write_json(evidence / "evaluation.json", evaluation, overwrite=True)
    atomic_write_json(path, provenance, overwrite=True)
    if change != "bytes": _seal(evidence)
    if change == "unbound_provenance":
        manifest = runner._json(evidence / "artifact_manifest.json")
        del manifest["files"]["provenance/provenance.json"]
        manifest["bundle_sha256"] = sha256_json(manifest["files"])
        atomic_write_json(evidence / "artifact_manifest.json", manifest, overwrite=True)
        marker = runner._json(evidence / "COMPLETED.json")
        marker["artifact_manifest_sha256"] = sha256_file(evidence / "artifact_manifest.json")
        atomic_write_json(evidence / "COMPLETED.json", marker, overwrite=True)
    monkeypatch.setattr(runner, "pointing_tables", lambda *a: pytest.fail("Scoring before source validation"))
    with pytest.raises((ValueError, OSError)): runner.run_descriptive(**kwargs)
    assert not runner.NAMESPACE.exists()


def test_source_pairs_exact_seeds_counts_and_opt_in(tmp_path, monkeypatch):
    kwargs, sources, _ = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="authorize"):
        runner.run_descriptive(**{**kwargs, "authorize_analysis": False})
    with pytest.raises(ValueError, match="18 distinct"):
        runner.run_descriptive(**{**kwargs, "ablation_source_dirs": kwargs["ablation_source_dirs"][:-1]})
    source = sources[("ordinal_auxiliary_off", 44)]
    source.identity["full_model_reference"]["best_checkpoint_sha256"] = "0"*64
    with pytest.raises(ValueError, match="different complete-model"):
        runner.run_descriptive(**kwargs)


def test_export_failure_keeps_partial_no_completion_and_no_overwrite(tmp_path, monkeypatch):
    kwargs, _, _ = _setup(tmp_path, monkeypatch)
    original = runner._write_csv
    monkeypatch.setattr(runner, "_write_csv", lambda *a: (_ for _ in ()).throw(RuntimeError("synthetic export failure")))
    with pytest.raises(RuntimeError, match="export failure"): runner.run_descriptive(**kwargs)
    outputs = list(runner.NAMESPACE.iterdir())
    assert len(outputs) == 1 and not (outputs[0] / "COMPLETED.json").exists()
    monkeypatch.setattr(runner, "_write_csv", original)
    with pytest.raises(FileExistsError): runner.run_descriptive(**kwargs)


@pytest.mark.parametrize("mutation_point", ["evaluation.json", "COMPLETED.json"])
def test_source_mutation_and_post_marker_mutation_invalidate_completion(tmp_path, monkeypatch, mutation_point):
    kwargs, sources, _ = _setup(tmp_path, monkeypatch)
    original = runner.atomic_write_json
    changed = sources[("B1", 42)].source_dir / "checkpoints/best.pt"
    triggered = []
    def mutate(path, payload, **kw):
        result = original(path, payload, **kw)
        if Path(path).name == mutation_point:
            changed.write_bytes(b"synthetic concurrent mutation")
            triggered.append(True)
        return result
    monkeypatch.setattr(runner, "atomic_write_json", mutate)
    with pytest.raises(RuntimeError, match="Source artifact changed"): runner.run_descriptive(**kwargs)
    assert triggered == [True]
    assert not list(runner.NAMESPACE.rglob("COMPLETED.json"))


def test_protected_output_and_completed_parent_rejected(tmp_path, monkeypatch):
    kwargs, _, _ = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="namespace"):
        runner.run_descriptive(**kwargs, output_root=tmp_path / "sources")
    parent = runner.NAMESPACE / "already-complete"
    parent.mkdir(parents=True)
    (parent / "COMPLETED.json").write_text("{}")
    with pytest.raises(ValueError, match="completed bundle"):
        runner.run_descriptive(**kwargs, output_root=parent / "nested")


def test_completion_rejects_full_scope_claim_and_artifact_tampering(tmp_path, monkeypatch):
    kwargs, _, _ = _setup(tmp_path, monkeypatch)
    output = runner.run_descriptive(**kwargs)
    marker = runner._json(output / "COMPLETED.json")
    original = dict(marker)
    marker["full_supplemental_completion"] = True
    atomic_write_json(output / "COMPLETED.json", marker, overwrite=True)
    with pytest.raises(ValueError, match="only the approved descriptive subset"):
        runner.validate_descriptive_completion(output)
    atomic_write_json(output / "COMPLETED.json", original, overwrite=True)
    (output / "tables/accuracy_intervals.csv").write_text("synthetic corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.validate_descriptive_completion(output)


def test_cli_without_explicit_authorization_never_loads_sources(monkeypatch):
    monkeypatch.setattr(runner, "load_descriptive_inputs", lambda *a: pytest.fail("Unauthorized source access"))
    with pytest.raises(ValueError, match="authorize-analysis"):
        runner.descriptive_cli(["--full-source", "absent", "--b1-source", "absent",
                                "--ablation-source", "absent", "--attention-source", "absent"])
