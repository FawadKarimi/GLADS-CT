"""CPU-only ablation contracts and tiny integration; no university sources/jobs."""
from copy import deepcopy
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
from consendxformer.phase4_config import load_phase4_config
from consendxformer.phase4_ablations import phase4_ablation_specifications, phase4_ablation_protocol_record
from consendxformer.phase4_training import compute_phase4_loss, compute_training_class_weights
from consendxformer.phase4_artifacts import Phase4ExperimentLayout
from consendxformer.phase4_ablation_contracts import *
import consendxformer.phase4_ablation_runner as runner
import consendxformer.phase4_ablation_execution as execution
from test_phase4_execution import TinyDataset, TinyModel, _supervision

ORIGINAL_PREPARE = runner.prepare_ablation


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def bundle():
    return load_phase4_config(ROOT / "configs/phase4_consensdxformer.yaml")


def test_four_single_toggles_and_twelve_frozen_run_specs_do_not_mutate_history():
    original = bundle()
    before = deepcopy(original.phase4_config)
    historical = phase4_ablation_protocol_record()
    for spec in phase4_ablation_specifications():
        for seed in (42, 43, 44):
            run = AblationRunSpec(spec.name, seed)
            config = effective_ablation_config(original, run)
            assert [key for key, value in config["ablation"].items() if not value] == [spec.disabled_toggle]
            assert config["model"]["use_spatial_attention"] is True
            assert config["model"]["use_hierarchical_fusion"] == config["ablation"]["use_hierarchical_fusion"]
            assert config["execution"]["lambda_ordinal"] == .5
            assert config["execution"]["lambda_attention"] == .05
            for section in ("training", "dataset", "augmentation", "data", "determinism", "losses", "inference"):
                assert config[section] == before[section]
            assert config["augmentation"] == original.b1_anchor_config["augmentation"]
    assert original.phase4_config == before
    assert historical == phase4_ablation_protocol_record()
    assert all(s.training_execution_authorized is False for s in phase4_ablation_specifications())


@pytest.mark.parametrize("variant,seed", [("all_off", 42), ("attention_alignment", 41), ("ordinal_auxiliary", True)])
def test_invalid_specs_fail_closed(variant, seed):
    with pytest.raises(ValueError):
        AblationRunSpec(variant, seed).specification()


@pytest.mark.parametrize("variant", [s.name for s in phase4_ablation_specifications()])
def test_exact_remaining_loss_composition_and_native_model_paths(variant):
    config = effective_ablation_config(bundle(), AblationRunSpec(variant, 42))
    # Explicit tiny untrained architecture fixture; no production config is edited.
    config["model"].update(cnn_channels=[2, 4, 8], embed_dim=8, transformer_heads=2, transformer_layers=1,
                           classifier_hidden_dim=8, dropout=0.)
    model = execution._model(config).eval()
    outputs = model(torch.linspace(0, 1, 64 * 64).reshape(1, 1, 64, 64))
    assert outputs["attention_map"].shape == (1, 1, 8, 8)
    assert outputs["ordinal_logits"].shape == (1, 4)  # Loss ablation, not head redesign.
    assert model.use_hierarchical_fusion == (variant != "hierarchical_fusion")
    if variant == "hierarchical_fusion":
        assert outputs["fused_features"] is outputs["deep_fused_features"]
    else:
        assert outputs["fused_features"].shape[-2:] == (32, 32)
    target = {"label": torch.tensor([2]), "soft_label": torch.tensor([[.2, .3, .5]]),
              "disagreement": torch.tensor([.7]), "ordinal_target": torch.tensor([[1., .8, .5, .2]]),
              "soft_consensus_mask": torch.ones(1, 1, 64, 64)}
    weights = torch.tensor([1., 2., 3.])
    all_on = {key: True for key in config["ablation"]}
    baseline = compute_phase4_loss(outputs, target, lambda_ordinal=.5, lambda_attention=.05, class_weights=weights, ablation=all_on)
    loss = compute_phase4_loss(outputs, target, lambda_ordinal=.5, lambda_attention=.05, class_weights=weights, ablation=config["ablation"])
    diagnostic = torch.nn.functional.cross_entropy(outputs["logits"], target["label"], weight=weights) if variant == "disagreement_weighting" else baseline.diagnostic
    ordinal = baseline.ordinal * (variant != "ordinal_auxiliary")
    attention = baseline.attention * (variant != "attention_alignment")
    torch.testing.assert_close(loss.diagnostic, diagnostic)
    torch.testing.assert_close(loss.ordinal, ordinal)
    torch.testing.assert_close(loss.attention, attention)
    torch.testing.assert_close(loss.total, diagnostic + .5 * ordinal + .05 * attention)
    if variant == "attention_alignment":
        loss.total.backward()
        assert any("attention" in name and parameter.grad is not None for name, parameter in model.named_parameters())


class ExactTinyDataset(TinyDataset):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        row = self.frame.iloc[index]
        result.update(split=row["split"], rendered_mask_path=row["rendered_mask_path"])
        return result


def install_tiny(tmp_path, monkeypatch, variant="attention_alignment", seed=42):
    original = bundle()
    small = deepcopy(original.phase4_config)
    small["training"]["epochs"] = 2
    small["dataset"]["batch_size"] = 3
    small["data"]["rendered_dir"] = str(tmp_path)
    configured = replace(original, phase4_config=small)
    spec = AblationRunSpec(variant, seed)
    config = effective_ablation_config(configured, spec)
    supervision = _supervision()
    supervision["image_path"] = [str(tmp_path / path) for path in supervision["image_path"]]
    snapshot_path = tmp_path / "immutable-full-source.txt"
    snapshot_path.write_text("synthetic fixture; never a university checkpoint")
    snapshot = {str(snapshot_path): sha256_file(snapshot_path)}
    lock = SimpleNamespace(to_dict=lambda: {"fixture": "tiny only"})
    ready = {"manifest_bundle_sha256": "a" * 64, "supervision_sha256": "b" * 64,
             "ordered_supervision_sha256": "c" * 64, "phase3_input_fingerprint": {"ordered_samples_sha256": "d" * 64},
             "cohort_view": "paper_aligned_multi_reader_2_to_4", "expected_resolution": 64,
             "sample_count": 12, "split_counts": {"train": 6, "val": 3, "test": 3}}
    ref = {"source_dir": str(tmp_path / "full"), "descriptor": {**ready, "seed": seed}, "best_checkpoint_sha256": "e" * 64}
    identity = make_identity(spec, config, "f" * 40, ready, ref)
    output = tmp_path / NAMESPACE / "runs" / identity["experiment_id"]
    project = {"commit": "f" * 40, "branch": "phase4-consendxformer", "working_tree_clean": True}
    prepared = runner.PreparedAblation(spec, config, identity, supervision, ready, {"fixture": "manifest"}, lock,
                                      project, {}, {}, snapshot, output)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "frozen_bundle", lambda: (configured, lock))
    monkeypatch.setattr(runner, "revalidate_inputs", lambda *args: (SimpleNamespace(to_dict=lambda: prepared.manifest), SimpleNamespace(to_dict=lambda: ready), supervision))
    monkeypatch.setattr(runner, "_require_project_state", lambda *args: project)
    monkeypatch.setattr(runner, "assert_supervision", lambda frame, expected: None)  # Real hash/readiness guards tested separately.
    monkeypatch.setattr(runner, "validate_phase4_input_lock", lambda *args: None)
    monkeypatch.setattr(runner, "recheck_prepared", lambda p: runner.assert_source_snapshot_unchanged(p.source_snapshot))

    def prepare(*args):
        if output.exists():
            raise FileExistsError("Ablation identity already exists")
        return prepared

    monkeypatch.setattr(runner, "prepare_ablation", prepare)
    reference = SimpleNamespace(source_dir=Path(ref["source_dir"]), descriptor=ref["descriptor"], source_snapshot=snapshot)
    monkeypatch.setattr(runner, "load_phase4_final_prediction_source", lambda *args: reference)
    monkeypatch.setattr(runner, "reference_record", lambda source: ref)
    original_build = execution.build_phase4_initial_datasets
    monkeypatch.setattr(execution, "build_phase4_initial_datasets", lambda *args, **kwargs: original_build(*args, **kwargs, dataset_factory=ExactTinyDataset))
    monkeypatch.setattr(execution, "_build_test_dataset", lambda frame, config, root: ExactTinyDataset(frame, image_size=2, in_channels=1, rendered_root=root))
    monkeypatch.setattr(execution, "_model", lambda config: TinyModel())
    TinyDataset.constructions = []; TinyDataset.event_log = []; TinyModel.event_log = TinyDataset.event_log
    return prepared


def test_tiny_complete_run_persisted_snapshots_predictions_and_separate_loader(tmp_path, monkeypatch):
    prepared = install_tiny(tmp_path, monkeypatch)
    source_bytes = Path(next(iter(prepared.source_snapshot))).read_bytes()
    path = runner.run_ablation(prepared.spec, "unused", authorize_training=True, device="cpu")
    source = runner.load_ablation_source(path)
    assert source.identity == prepared.identity
    assert source.identity["run_kind"] == RUN_KIND
    assert "run_mode" not in source.identity
    assert TinyDataset.event_log.index("best:loaded") < TinyDataset.event_log.index("construct:test")
    assert set(source.predictions) == {"val", "test"}
    assert all(tuple(frame.columns) == PREDICTION_COLUMNS for frame in source.predictions.values())
    assert Path(next(iter(prepared.source_snapshot))).read_bytes() == source_bytes
    assert TinyDataset.constructions[0][1].__dict__["enabled"] is True
    assert all(augmenter is None for _, augmenter in TinyDataset.constructions[1:])
    from consendxformer.phase4_evaluation_runner import load_phase4_final_prediction_source
    with pytest.raises((ValueError, RuntimeError)):
        load_phase4_final_prediction_source(path)
    with pytest.raises(FileExistsError):
        runner.run_ablation(prepared.spec, "unused", authorize_training=True)


def test_best_last_snapshot_aliasing_and_best_reload_cannot_change_last(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    identity = {"run_kind": RUN_KIND, "variant": "attention_alignment", "seed": 42}
    history = []
    for epoch, loss in ((1, .5), (2, .8)):
        with torch.no_grad():
            model.weight.fill_(epoch)
        metrics = {"loss": loss, "total_loss": loss}
        history.append({"epoch": epoch, "train": {"loss": 1.}, "validation": metrics})
        record = checkpoint_payload(identity, kind="best" if epoch == 1 else "last", epoch=epoch,
            best_epoch=1, best_loss=.5, validation=metrics, model=model, optimizer=optimizer, scheduler=scheduler)
        save_checkpoint(record, tmp_path / f"{'best' if epoch == 1 else 'last'}.pt")
    with torch.no_grad():
        model.weight.fill_(99)
    best, last = load_checkpoint_pair(tmp_path, identity, history)
    assert best["model_state"]["weight"].item() == 1
    assert last["model_state"]["weight"].item() == 2
    model.load_state_dict(best["model_state"])
    assert last["model_state"]["weight"].item() == 2
    tampered = deepcopy(identity); tampered["seed"] = 43
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint_pair(tmp_path, tampered, history)


@pytest.mark.parametrize("mutation", ["lr", "dsa", "second_toggle", "lambda"])
def test_unauthorized_config_overrides_rejected(tmp_path, monkeypatch, mutation):
    prepared = install_tiny(tmp_path, monkeypatch)
    config = deepcopy(prepared.config)
    if mutation == "lr": config["training"]["lr"] *= 2
    elif mutation == "dsa": config["model"]["use_spatial_attention"] = False
    elif mutation == "lambda": config["losses"]["selected_lambda_attention"] = .1
    else: config["ablation"]["use_ordinal_head"] = False
    with pytest.raises(ValueError, match="unauthorized"):
        validate_identity(prepared.identity, config, runner.frozen_bundle()[0])


def test_prediction_matching_handles_text_dtypes_but_rejects_identity_label_changes(tmp_path, monkeypatch):
    p = install_tiny(tmp_path, monkeypatch)
    frame = p.supervision.loc[p.supervision["split"] == "val"]
    predictions = build_predictions(frame, np.zeros((3, 3)), p.identity)
    for col in ID_FIELDS:
        predictions[col] = predictions[col].astype("string")
    validate_predictions(predictions, frame.astype({key: object for key in frame.columns}), p.identity)
    for changed in (predictions.iloc[::-1], pd.concat([predictions.iloc[:2], predictions.iloc[:1]]), predictions.assign(true_class="malignant")):
        with pytest.raises(ValueError):
            validate_predictions(changed, frame, p.identity)
    np.testing.assert_array_equal(compute_training_class_weights(p.supervision), compute_training_class_weights(p.supervision.assign(label=np.where(p.supervision["split"] == "train", p.supervision["label"], "malignant"))))


def test_readiness_failure_precedes_data_or_model_construction(tmp_path, monkeypatch):
    p = install_tiny(tmp_path, monkeypatch)
    def reject(*args): raise ValueError("readiness denied")
    monkeypatch.setattr(runner, "revalidate_inputs", reject)
    with pytest.raises(ValueError, match="readiness denied"):
        ORIGINAL_PREPARE(p.spec, "unused")
    with pytest.raises(ValueError, match="readiness denied"):
        runner.run_ablation(p.spec, "unused", authorize_training=True)
    assert TinyDataset.constructions == []
    assert not list(tmp_path.rglob("COMPLETED.json"))


@pytest.mark.parametrize("failure", ["interrupt", "source_mutation", "invalid_predictions"])
def test_interrupted_invalid_or_mutating_execution_cannot_complete(tmp_path, monkeypatch, failure):
    p = install_tiny(tmp_path, monkeypatch)
    original = execution.execute_ablation
    def fail(*args, **kwargs):
        if failure == "interrupt": raise KeyboardInterrupt("fixture interruption")
        result = original(*args, **kwargs)
        if failure == "source_mutation": Path(next(iter(p.source_snapshot))).write_text("tampered fixture")
        else: result["predictions"]["test"]["true_class"] = "malignant"
        return result
    monkeypatch.setattr(execution, "execute_ablation", fail)
    with pytest.raises((KeyboardInterrupt, ValueError, RuntimeError)):
        runner.run_ablation(p.spec, "unused", authorize_training=True)
    assert not list(tmp_path.rglob("COMPLETED.json"))
    assert list(tmp_path.rglob("FAILED.json"))


def test_separate_training_authorization_is_required_before_any_preflight(monkeypatch):
    monkeypatch.setattr(runner, "prepare_ablation", lambda *args: pytest.fail("unauthorized preflight"))
    with pytest.raises(ValueError, match="authorization"):
        runner.run_ablation(AblationRunSpec("attention_alignment", 42), "missing")


def reseal_test_bundle(directory):
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    manifest["files"] = {name: sha256_file(directory / name) for name in manifest["files"]}
    manifest["bundle_sha256"] = sha256_json(manifest["files"])
    (directory / "artifact_manifest.json").write_text(json.dumps(manifest))
    completion = json.loads((directory / "COMPLETED.json").read_text())
    completion["artifact_manifest_sha256"] = sha256_file(directory / "artifact_manifest.json")
    (directory / "COMPLETED.json").write_text(json.dumps(completion))


@pytest.mark.parametrize("change", ["config", "checkpoint_identity", "prediction_label", "metric", "loss_composition", "rng", "authorization"])
def test_separate_source_loader_rejects_resealed_semantic_changes(tmp_path, monkeypatch, change):
    p = install_tiny(tmp_path, monkeypatch)
    directory = runner.run_ablation(p.spec, "unused", authorize_training=True)
    if change == "checkpoint_identity":
        path = directory / "checkpoints/best.pt"
        payload = torch.load(path, weights_only=False)
        payload["identity"]["seed"] = 43
        torch.save(payload, path)
    elif change == "prediction_label":
        path = directory / "predictions/test.csv"
        frame = pd.read_csv(path); frame.loc[0, "true_class"] = "malignant"
        frame.to_csv(path, index=False)
    else:
        relative = {"config": "config/effective_config.json", "metric": "metrics/test.json", "loss_composition": "metrics/test.json",
                    "rng": "provenance/run.json", "authorization": "provenance/run.json"}[change]
        path = directory / relative
        record = json.loads(path.read_text())
        if change == "config": record["model"]["use_spatial_attention"] = False
        elif change == "metric": record["prediction_metrics"]["classification"]["accuracy"] = -1
        elif change == "loss_composition": record["within_variant_evaluation_losses"]["attention_loss"] = 1.
        elif change == "rng": record["seed_controls"]["deterministic_algorithms_requested"] = False
        else: record["runtime_training_explicitly_authorized"] = False
        path.write_text(json.dumps(record))
    reseal_test_bundle(directory)
    with pytest.raises((ValueError, RuntimeError)):
        runner.load_ablation_source(directory)


def test_new_namespace_refuses_nesting_inside_completed_artifacts(tmp_path):
    (tmp_path / "COMPLETED.json").write_text("synthetic existing bundle")
    with pytest.raises(ValueError, match="completed bundle"):
        runner.guard_new_namespace(tmp_path / "nested" / "run")
