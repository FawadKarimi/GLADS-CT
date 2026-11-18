"""Tiny CPU integration: trust gates replaced, real inference/maps/writer exercised."""

from contextlib import contextmanager
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
from consendxformer.phase4_evaluation_runner import SavedPredictionInputs, SavedPredictionSource
from consendxformer.phase4_attention_evidence import FLAGGED_NODULE, build_location_prior, supplementary_protocol
from consendxformer.phase4_attention_runtime import FrozenB1
from consendxformer.phase4_dataset import _read_image, _read_soft_mask
import consendxformer.phase4_attention_comparison as runner
import consendxformer.phase4_posthoc as posthoc
from test_phase4_posthoc import _fixture, TinyPosthocModel, _attention_identity_batch, _sealed_attention_loader


def _reference(path, rows):
    atomic_write_json(path / "evaluation.json", {"attention_alignment": {"per_seed": rows}})
    atomic_write_json(path / "provenance/provenance.json", {"fixture": "synthetic only"})
    _reseal(path)


def _reseal(path):
    files = {name: sha256_file(path / name) for name in ("evaluation.json", "provenance/provenance.json")}
    manifest = atomic_write_json(path / "artifact_manifest.json", {
        "schema_version": 1, "evaluation_id": "phase4-evaluation-synthetic-reference", "research_results_eligible": True,
        "files": files, "bundle_sha256": sha256_json(files),
    }, overwrite=True)
    atomic_write_json(path / "COMPLETED.json", {
        "schema_version": 1, "evaluation_id": "phase4-evaluation-synthetic-reference", "research_results_eligible": True,
        "artifact_manifest": manifest.name, "artifact_manifest_sha256": sha256_file(manifest),
    }, overwrite=True)


def _install(tmp_path, monkeypatch):
    sources = [_fixture(tmp_path / str(seed), seed=seed)[0] for seed in (42, 43, 44)]
    shared = sources[0].supervision.copy()
    shared.loc[0, "nodule_id"] = FLAGGED_NODULE
    sources = [replace(source, supervision=shared.copy()) for source in sources]
    by_dir = {source.source_dir: source for source in sources}
    monkeypatch.setattr(posthoc, "load_phase4_final_prediction_source", lambda path: by_dir[Path(path)])
    frozen = {}
    for source in sources:
        predictions = shared[["patient_id", "nodule_id", "series_instance_uid", "split"]].copy()
        predictions["true_class"] = shared["label"]
        source = replace(source, predictions=predictions)
        by_dir[source.source_dir] = source
        loaded = posthoc.load_frozen_final_best_model(source.source_dir, model_factory=lambda config: TinyPosthocModel())
        images = torch.from_numpy(np.stack([_read_image(path, 2) for path in shared["image_path"]])).unsqueeze(1)
        with torch.no_grad():
            outputs = loaded.model(images)
        for i, name in enumerate(("benign", "indeterminate", "malignant")):
            predictions[f"logit_{name}"] = outputs["logits"][:, i].numpy()
            predictions[f"prob_{name}"] = outputs["probs"][:, i].numpy()
        frozen[source.descriptor["seed"]] = loaded

    def loader(value):
        fields = _attention_identity_batch(value)
        # Actual fixture crop/mask bytes; masks never become model arguments.
        image = torch.from_numpy(np.stack([_read_image(path, 2) for path in fields["path"]])).unsqueeze(1)
        config = json.loads((value.source.source_dir / "config/effective_config.json").read_text())
        masks = torch.from_numpy(np.stack([_read_soft_mask(Path(config["data"]["rendered_dir"]) / path, 2) for path in fields["rendered_mask_path"]])).unsqueeze(1)
        return _sealed_attention_loader(value, [{"image": image, "soft_consensus_mask": masks,
                                                "true_class": shared["label"].tolist(), **fields}])

    phase4 = {seed: value.source for seed, value in frozen.items()}
    b1_sources, b1_models = {}, {}
    for seed, value in frozen.items():
        path = tmp_path / f"b1-{seed}"; path.mkdir()
        anchor = path / "synthetic-source.txt"; anchor.write_text(f"untrained fixture {seed}")
        source = SavedPredictionSource("B1", path, {**value.source.descriptor, "model": "B1", "experiment_id": f"b1-fixture-{seed}"},
                                       value.source.predictions.copy(), None, {}, {str(anchor): sha256_file(anchor)})
        model = TinyPosthocModel(); model.load_state_dict(value.model.state_dict()); model.eval().requires_grad_(False)
        b1_sources[seed] = source
        b1_models[path] = FrozenB1(model, source, "b" * 64, {})
    train = shared.copy()
    train["split"] = "train"; train["nodule_id"] = ["train-n1", "train-n2"]
    train["patient_id"] = ["train-p1", "train-p2"]
    supervision = pd.concat([train, shared], ignore_index=True)
    snapshots = {path: digest for source in (*phase4.values(), *b1_sources.values()) for path, digest in source.source_snapshot.items()}
    inputs = SavedPredictionInputs(phase4, b1_sources, supervision, snapshots)
    monkeypatch.setattr(runner, "load_saved_prediction_inputs", lambda *args: inputs)
    monkeypatch.setattr(runner, "validate_evaluator_repository", lambda root: {"commit": "c" * 40, "branch": "phase4-consendxformer", "clean": True})
    protocol = supplementary_protocol()
    protocol.update(test_nodules=2, test_patients=2, training_nodules=2)
    config = tmp_path / "synthetic_protocol.json"; config.write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "PROTOCOL_PATH", config)
    monkeypatch.setattr(runner, "supplementary_protocol", lambda: protocol)
    monkeypatch.setattr(runner, "load_frozen_final_best_model", lambda path, **kwargs: frozen[by_dir[Path(path)].descriptor["seed"]])
    monkeypatch.setattr(runner, "build_frozen_attention_loader", loader)
    monkeypatch.setattr(runner, "load_frozen_b1", lambda path, *args, **kwargs: b1_models[Path(path)])

    @contextmanager
    def fake_api(*args):
        yield TinyPosthocModel, {"fixture": "synthetic-only; pinned API tested separately"}

    monkeypatch.setattr(runner, "pinned_b1_api", fake_api)
    training_masks = np.array([[[1., 0], [0, 0]], [[0., 0], [0, 1.]]])
    training_file = tmp_path / "train_only_masks.npy"; np.save(training_file, training_masks)

    def prior(value, grid):
        array, provenance = build_location_prior(train.rename(columns={"label": "true_class"}), np.load(training_file, allow_pickle=False), grid, expected_count=2)
        return array, provenance, {str(training_file): sha256_file(training_file)}

    monkeypatch.setattr(runner, "_training_prior", prior)
    reference = tmp_path / "reference"
    reference_rows = [posthoc.evaluate_attention_alignment_from_loader(value, loader(value)) for value in frozen.values()]
    _reference(reference, reference_rows)
    kwargs = {"phase4_source_dirs": [source.source_dir for source in phase4.values()],
              "b1_source_dirs": [source.source_dir for source in b1_sources.values()],
              "reference_attention_bundle": reference, "output_root": tmp_path / "output"}
    return kwargs, inputs, reference


def test_full_evidence_bundle_is_deterministic_and_source_immutable(tmp_path, monkeypatch):
    kwargs, inputs, reference = _install(tmp_path, monkeypatch)
    before = {path: sha256_file(path) for path in inputs.source_snapshot}
    output = runner.run_attention_comparison(**kwargs)
    result = json.loads((output / "evaluation.json").read_text())
    assert result["replay_consistent"] is True
    assert result["status"] == "complete"
    assert len(result["patient_bootstrap"]) == 9
    assert len(result["seed_results"]) == 3
    assert len(list(output.glob("location_prior.npz"))) == 1
    assert (output / "overlay_panel.png").is_file()
    assert (output / "tables/flagged_case.csv").is_file()
    with np.load(output / "seed_42_model_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["b1_attention"].shape == (2, 1, 2, 2)
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        assert sha256_file(output / relative) == digest
    assert before == {path: sha256_file(path) for path in inputs.source_snapshot}
    assert (output / "COMPLETED.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run_attention_comparison(**kwargs)
    again = runner.run_attention_comparison(**{**kwargs, "output_root": tmp_path / "second"})
    assert output.name == again.name
    assert (output / "artifact_manifest.json").read_bytes() == (again / "artifact_manifest.json").read_bytes()


def test_reference_source_binding_rejects_resealed_wrong_checkpoint(tmp_path, monkeypatch):
    kwargs, inputs, reference = _install(tmp_path, monkeypatch)
    data = json.loads((reference / "evaluation.json").read_text())
    data["attention_alignment"]["per_seed"][0]["checkpoint_sha256"] = "0" * 64
    (reference / "evaluation.json").write_text(json.dumps(data)); _reseal(reference)
    with pytest.raises(ValueError, match="another frozen checkpoint"):
        runner.run_attention_comparison(**kwargs)
    assert not kwargs["output_root"].exists()


def test_scalar_discrepancy_is_exported_without_changing_previous_bundle(tmp_path, monkeypatch):
    kwargs, inputs, reference = _install(tmp_path, monkeypatch)
    data = json.loads((reference / "evaluation.json").read_text())
    data["attention_alignment"]["per_seed"][0]["metrics"]["per_sample"][0]["dice"] = -1.
    (reference / "evaluation.json").write_text(json.dumps(data)); _reseal(reference)
    digest = sha256_file(reference / "evaluation.json")
    output = runner.run_attention_comparison(**kwargs)
    result = json.loads((output / "evaluation.json").read_text())
    assert result["status"] == "complete_with_replay_discrepancies"
    assert result["comparison_interpretation_permitted"] is False
    assert sha256_file(reference / "evaluation.json") == digest


def test_source_mutation_cannot_produce_completion(tmp_path, monkeypatch):
    kwargs, inputs, reference = _install(tmp_path, monkeypatch)
    original = runner.infer_matched_attention

    def mutate(*args):
        result = original(*args)
        Path(next(iter(inputs.b1[42].source_snapshot))).write_bytes(b"synthetic tampering")
        return result

    monkeypatch.setattr(runner, "infer_matched_attention", mutate)
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        runner.run_attention_comparison(**kwargs)
    assert not list(kwargs["output_root"].rglob("COMPLETED.json"))


def test_new_output_cannot_be_nested_in_any_completed_bundle(tmp_path):
    existing = tmp_path / "existing"; existing.mkdir()
    (existing / "COMPLETED.json").write_text("{}")
    with pytest.raises(ValueError, match="inside an existing completed bundle"):
        runner.run_attention_comparison(phase4_source_dirs=[], b1_source_dirs=[],
                                        reference_attention_bundle=tmp_path / "reference",
                                        output_root=existing / "child")


@pytest.mark.parametrize("change", ["duplicate_seed", "wrong_cohort", "changed_definition"])
def test_previous_bundle_cannot_be_resealed_to_change_bindings(tmp_path, monkeypatch, change):
    kwargs, inputs, reference = _install(tmp_path, monkeypatch)
    data = json.loads((reference / "evaluation.json").read_text())
    rows = data["attention_alignment"]["per_seed"]
    if change == "duplicate_seed":
        rows[1]["seed"] = 42
    elif change == "wrong_cohort":
        rows[1]["metrics"]["per_sample"][0]["nodule_id"] = "not-the-frozen-nodule"
    else:
        rows[1]["definition"]["dice_iou"]["attention_quantile"] = .8
    (reference / "evaluation.json").write_text(json.dumps(data)); _reseal(reference)
    with pytest.raises(ValueError):
        runner.run_attention_comparison(**kwargs)
    assert not kwargs["output_root"].exists()


def _prior_fixture(tmp_path, monkeypatch):
    import consendxformer.manifest_guard as manifest_guard
    import consendxformer.phase4_readiness as readiness_module

    source = _fixture(tmp_path)[0]
    training = source.supervision.copy()
    training["split"] = "train"
    training["nodule_id"] = ["train-n1", "train-n2"]
    training["patient_id"] = ["train-p1", "train-p2"]
    testing = source.supervision.copy()
    # These deliberately nonexistent targets must never be read into the prior.
    testing["rendered_mask_path"] = ["forbidden-test-1.npz", "forbidden-test-2.npz"]
    supervision = pd.concat([training, testing], ignore_index=True)
    source = replace(source, supervision=supervision)
    descriptor = source.descriptor
    readiness = SimpleNamespace(
        **{field: descriptor[field] for field in (
            "manifest_bundle_sha256", "supervision_sha256", "ordered_supervision_sha256")},
        phase3_input_fingerprint=SimpleNamespace(to_dict=lambda: descriptor["phase3_input_fingerprint"]),
        to_dict=lambda: {"fixture": "readiness boundary mocked; prior file loading is real"},
    )
    monkeypatch.setattr(manifest_guard, "validate_frozen_manifest_bundle", lambda path: "synthetic-manifest")
    monkeypatch.setattr(readiness_module, "validate_phase4_readiness", lambda *args, **kwargs: (readiness, supervision))
    monkeypatch.setattr(runner, "build_location_prior", lambda rows, masks, grid: build_location_prior(rows, masks, grid, expected_count=2))
    return SimpleNamespace(source=source), readiness, supervision


def test_prior_runtime_reads_only_readiness_bound_training_masks(tmp_path, monkeypatch):
    frozen, readiness, supervision = _prior_fixture(tmp_path, monkeypatch)
    prior, provenance, snapshot = runner._training_prior(frozen, (2, 2))
    np.testing.assert_array_equal(prior, np.full((2, 2), .25))
    assert len(snapshot) == len(provenance["mask_sources"]) == 2
    assert {row["nodule_id"] for row in provenance["training_identities"]} == {"train-n1", "train-n2"}
    assert provenance["test_masks_used"] is False
    assert all("forbidden-test" not in path for path in snapshot)
    assert all(sha256_file(path) == digest for path, digest in snapshot.items())


@pytest.mark.parametrize("change", ["readiness_hash", "training_membership", "mask_during_read"])
def test_prior_runtime_rejects_changed_readiness_membership_or_masks(tmp_path, monkeypatch, change):
    import consendxformer.phase4_dataset as dataset

    frozen, readiness, supervision = _prior_fixture(tmp_path, monkeypatch)
    if change == "readiness_hash":
        readiness.supervision_sha256 = "0" * 64
    elif change == "training_membership":
        # Separate the live frame from the persisted source before tampering.
        frozen.source = replace(frozen.source, supervision=supervision.copy())
        supervision.loc[0, "nodule_id"] = "substituted-train-nodule"
    else:
        original = dataset._read_soft_mask

        def mutate(path, resolution):
            mask = original(path, resolution)
            Path(path).write_bytes(b"changed synthetic training target after reading")
            return mask

        monkeypatch.setattr(dataset, "_read_soft_mask", mutate)
    with pytest.raises((ValueError, RuntimeError), match="differs|exact readiness-approved|changed during evaluation"):
        runner._training_prior(frozen, (2, 2))
