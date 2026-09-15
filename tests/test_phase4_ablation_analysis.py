"""Synthetic saved-evidence replay; actual tiny image-only inference, never LIDC."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch = pytest.importorskip("torch")

from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase3_artifacts import atomic_write_json
from consendxformer.phase4_dataset import _read_image, _read_soft_mask
from consendxformer.phase4_ablation_contracts import ID_FIELDS, RUN_KIND, build_predictions, identity_rows
from consendxformer.phase4_ablation_runner import AblationSource, reference_record
from consendxformer.phase4_attention_evidence import array_record, write_map_archive, supplementary_protocol, FLAGGED_NODULE
import consendxformer.phase4_ablation_analysis as analysis
import consendxformer.phase4_execution as primitives
from test_phase4_posthoc import _fixture, TinyPosthocModel


def _seal(directory):
    files = {p.relative_to(directory).as_posix(): sha256_file(p) for p in directory.rglob("*")
             if p.is_file() and p.name not in ("artifact_manifest.json", "COMPLETED.json")}
    manifest = {"evaluation_id": "synthetic-attention-reference", "files": files, "bundle_sha256": sha256_json(files), "replay_consistent": True}
    atomic_write_json(directory / "artifact_manifest.json", manifest, overwrite=True)
    atomic_write_json(directory / "COMPLETED.json", {"evaluation_id": manifest["evaluation_id"],
        "artifact_manifest_sha256": sha256_file(directory / "artifact_manifest.json"), "replay_consistent": True,
        "comparison_interpretation_permitted": True}, overwrite=True)


def _setup(tmp_path, monkeypatch):
    fixture, _, _ = _fixture(tmp_path / "tiny")
    config = json.loads((fixture.source_dir / "config/effective_config.json").read_text())
    test = fixture.supervision.copy()
    test.loc[0, "nodule_id"] = FLAGGED_NODULE
    mask_path = Path(config["data"]["rendered_dir"]) / test.iloc[0]["rendered_mask_path"]
    np.savez(mask_path, soft_consensus_mask=np.full((2, 2), .1, dtype=np.float32))
    train = pd.concat([test.iloc[[0]].copy() for _ in range(807)], ignore_index=True)
    train["split"] = "train"
    train["nodule_id"] = [f"synthetic-train-n{i:04d}" for i in range(807)]
    train["patient_id"] = [f"synthetic-train-p{i:04d}" for i in range(807)]
    train["series_instance_uid"] = [f"synthetic-train-s{i:04d}" for i in range(807)]
    val = test.copy(); val["split"] = "val"; val["nodule_id"] = ["synthetic-v1", "synthetic-v2"]
    supervision = pd.concat([train, val, test], ignore_index=True)
    images = np.stack([_read_image(path, 2) for path in test["image_path"]])[:, None]
    masks = np.stack([_read_soft_mask(Path(config["data"]["rendered_dir"]) / path, 2) for path in test["rendered_mask_path"]])[:, None]
    model = TinyPosthocModel().eval().requires_grad_(False)
    with torch.no_grad(): output = model(torch.from_numpy(images))
    ids = pd.DataFrame(identity_rows(test))
    from consendxformer.phase4_attention_evidence import map_evidence
    maps = output["attention_map"].numpy()
    evidence = map_evidence(maps, masks, ids.to_dict("records"))
    inputs = {**{field: np.asarray(ids[field].tolist(), dtype=np.str_) for field in ID_FIELDS},
              "ct_crops": images, "soft_masks_input": masks, "soft_masks_native_grid": evidence["resized_soft_masks"],
              "target_distributions": evidence["target_distributions"]}
    snapshot = {str(p): sha256_file(p) for p in [*map(Path, test["image_path"]),
                 *[Path(config["data"]["rendered_dir"]) / path for path in test["rendered_mask_path"]]]}
    sources, full_sources = {}, {}
    for seed in (42, 43, 44):
        full = replace(fixture, descriptor={**fixture.descriptor, "seed": seed})
        identity = {"run_kind": RUN_KIND, "variant": "attention_alignment", "seed": seed, "lambda_ordinal": .5,
                    "lambda_attention": .05, "experiment_id": f"synthetic-ablation-{seed}",
                    "inputs": {"supervision_sha256": "a" * 64}, "full_model_reference": reference_record(full)}
        predictions = {split: build_predictions(frame, output["logits"].numpy(), identity) for split, frame in (("val", val), ("test", test))}
        source_path = tmp_path / f"synthetic-ablation-{seed}"; source_path.mkdir()
        sources[seed] = AblationSource(source_path, identity, config, supervision, predictions, snapshot, {"model_state": model.state_dict()})
        full_sources[seed] = replace(full, predictions=predictions["test"])
    reference = tmp_path / "synthetic-evidence"
    prior = np.full((2, 2), .25)
    archives = [write_map_archive(reference / "location_prior.npz", {"prior": prior}), write_map_archive(reference / "test_inputs.npz", inputs)]
    for seed in (42, 43, 44):
        archives.append(write_map_archive(reference / f"seed_{seed}_model_arrays.npz", {"cdx_attention": maps}))
    training_ids = sorted(identity_rows(train), key=lambda row: tuple(row[field] for field in ID_FIELDS))
    provenance = {"fixture": "synthetic CPU unit test, not research evidence", "archives": archives,
                  "source_experiments": [{"model": "ConsenDxFormer", "seed": seed, "descriptor": sources[seed].identity["full_model_reference"]["descriptor"]} for seed in sources],
                  "prior": {"array": array_record(prior), "test_masks_used": False, "n_nodules": 807,
                            "training_identities": training_ids, "training_identities_sha256": sha256_json(training_ids)},
                  "input_file_snapshot": snapshot}
    atomic_write_json(reference / "provenance/provenance.json", provenance)
    atomic_write_json(reference / "evaluation.json", {"protocol": supplementary_protocol(), "replay_consistent": True})
    _seal(reference)
    actual_protocol = analysis.execution_protocol()
    monkeypatch.setattr(analysis, "execution_protocol", lambda: {**actual_protocol, "supplementary_reference_id": "synthetic-attention-reference"})
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "_require_project_state", lambda *args: {"commit": "b" * 40, "branch": "phase4-consendxformer"})
    monkeypatch.setattr(analysis, "load_ablation_source", lambda path: next(s for s in sources.values() if s.source_dir == Path(path)))
    monkeypatch.setattr(analysis, "frozen_bundle", lambda: (None, None))
    monkeypatch.setattr(analysis, "revalidate_inputs", lambda *args: (None, SimpleNamespace(to_dict=lambda: {"supervision_sha256": "a" * 64}), supervision))
    monkeypatch.setattr(analysis, "assert_supervision", lambda *args: None)
    monkeypatch.setattr(primitives, "_model", lambda config: TinyPosthocModel())
    config["determinism"] = {"dataloader_worker_seeding": True, "seeded_dataloader_generator": True}
    # Full-source binding is tested by the strict runner tests; here the same
    # synthetic source directory is reused, with explicitly bound seed records.
    state = {"seed": 42}
    def full_loader(path):
        seed = state["seed"]; state["seed"] += 1
        return full_sources[seed]
    monkeypatch.setattr(analysis, "load_phase4_final_prediction_source", full_loader)
    monkeypatch.setattr(analysis, "load_phase4_prediction_csv", lambda path: sources[42].predictions[Path(path).stem])
    def classification_ci(full, ablation, **kwargs):
        assert kwargs == {"seed": 20260914, "n_resamples": 10000, "confidence_level": .95}
        assert full["nodule_id"].tolist() == ablation["nodule_id"].tolist()
        return {"point_estimates": {"difference": {}}, "confidence_intervals": {"difference": {}},
                "bootstrap": {"fixture": "wiring only; shared bootstrap math covered by existing CPU tests"}}
    monkeypatch.setattr(analysis, "bootstrap_paired_differences", classification_ci)
    return sources, reference, inputs


def test_completed_ablation_analysis_reuses_fixed_prior_and_preserves_empty_foreground(tmp_path, monkeypatch):
    sources, reference, inputs = _setup(tmp_path, monkeypatch)
    prior_hash = sha256_file(reference / "location_prior.npz")
    output = analysis.analyze_ablations([s.source_dir for s in sources.values()], reference, authorize_analysis=True)
    result = json.loads((output / "analysis.json").read_text())
    assert result["replay_consistent"] and result["prior_unchanged"] and not result["training_performed"]
    assert len(result["seed_results"]) == 3
    assert sha256_file(reference / "location_prior.npz") == prior_hash == sha256_file(output / "location_prior.npz")
    flagged = pd.read_csv(output / "flagged_case.csv")
    assert len(flagged) == 6 and flagged["dice"].eq(0).all() and flagged["iou"].eq(0).all()
    assert flagged["zero_overlap_reason"].eq("empty_target_foreground").all()
    assert flagged["consensus_foreground_count"].eq(0).all()
    assert (output / "COMPLETED.json").is_file()
    with np.load(output / "seed_42_maps.npz", allow_pickle=False) as saved:
        assert saved["ablation_attention"].shape == saved["full_attention"].shape == (2, 1, 2, 2)
        np.testing.assert_array_equal(saved["nodule_id"], inputs["nodule_id"])
    with pytest.raises(FileExistsError):
        analysis.analyze_ablations([s.source_dir for s in sources.values()], reference, authorize_analysis=True)


@pytest.mark.parametrize("change", ["array", "source_binding", "training_identity", "replay"])
def test_resealed_prior_or_evidence_corruption_fails_closed(tmp_path, monkeypatch, change):
    sources, reference, inputs = _setup(tmp_path, monkeypatch)
    if change == "array":
        np.savez(reference / "location_prior.npz", prior=np.ones((2, 2)))
    else:
        path = reference / "provenance/provenance.json"
        record = json.loads(path.read_text())
        if change == "source_binding": record["source_experiments"][0]["descriptor"]["seed"] = 43
        elif change == "training_identity": record["prior"]["training_identities"][0]["split"] = "test"
        else:
            path = reference / "evaluation.json"; record = json.loads(path.read_text()); record["replay_consistent"] = False
        path.write_text(json.dumps(record))
    _seal(reference)
    with pytest.raises(ValueError):
        analysis.load_fixed_evidence(reference, sources)


def test_analysis_requires_separate_authorization_all_seeds_and_one_variant(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="authorization"):
        analysis.analyze_ablations([], "missing")
    sources, reference, inputs = _setup(tmp_path, monkeypatch)
    paths = [s.source_dir for s in sources.values()]
    with pytest.raises(ValueError, match="three distinct"):
        analysis.analyze_ablations(paths[:2], reference, authorize_analysis=True)
    sources[44].identity["variant"] = "ordinal_auxiliary"
    with pytest.raises(ValueError, match="one exact variant"):
        analysis.analyze_ablations(paths, reference, authorize_analysis=True)


def test_map_replay_rejects_changed_live_input_and_reports_saved_prediction_discrepancy(tmp_path, monkeypatch):
    sources, reference, inputs = _setup(tmp_path, monkeypatch)
    source = sources[42]
    with torch.no_grad():
        source.best_record["model_state"]["classifier.weight"].add_(1)
    maps, logits, probabilities, replay = analysis.infer_ablation_maps(source, inputs, device="cpu")
    # The synthetic crops have nonzero entries: changed weights differ from saved logits.
    assert not replay["consistent"]
    changed = dict(inputs); changed["ct_crops"] = inputs["ct_crops"] + 1
    with pytest.raises(ValueError, match="Live CT/consensus"):
        analysis.infer_ablation_maps(source, changed, device="cpu")
