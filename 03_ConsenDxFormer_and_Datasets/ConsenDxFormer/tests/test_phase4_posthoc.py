from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase4_checkpoints import (
    Phase4CheckpointContext,
    build_phase4_checkpoint,
    save_phase4_checkpoint,
)
from consendxformer.phase4_evaluation_runner import (
    SOURCE_FINAL_MODEL_COMMIT,
    SavedPredictionSource,
)
from consendxformer.phase4_posthoc import (
    FrozenAttentionLoader,
    FrozenFinalModel,
    Phase4PosthocError,
    aggregate_attention_seed_results,
    build_frozen_attention_loader,
    evaluate_attention_alignment_from_loader,
    load_frozen_final_best_model,
    predict_validated_boundary_cohort,
)
import consendxformer.phase4_posthoc as phase4_posthoc


class TinyPosthocModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(4, 3, bias=False)

    def forward(self, image):
        logits = self.classifier(torch.nn.functional.adaptive_avg_pool2d(image, 2).flatten(1))
        attention = torch.softmax(image.flatten(2), dim=2).reshape_as(image)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "attention_map": attention,
        }


def _attention_identity_batch(
    loaded: FrozenFinalModel, *, reverse: bool = False
) -> dict[str, list[str]]:
    source = loaded.source.supervision
    assert isinstance(source, pd.DataFrame)
    rows = source.iloc[::-1] if reverse else source
    mapping = {
        "patient_id": "patient_id",
        "nodule_id": "nodule_id",
        "nodule_sha256": "nodule_sha256",
        "series_instance_uid": "series_instance_uid",
        "split": "split",
        "image_path": "path",
        "rendered_mask_path": "rendered_mask_path",
    }
    return {
        target: rows[source_field].astype(str).tolist()
        for source_field, target in mapping.items()
    }


def _sealed_attention_loader(
    loaded: FrozenFinalModel, batches
) -> FrozenAttentionLoader:
    fields = _attention_identity_batch(loaded)
    records = [
        {field: fields[field][index] for field in fields}
        for index in range(len(fields["nodule_id"]))
    ]
    effective = json.loads(
        (loaded.source.source_dir / "config" / "effective_config.json").read_text(
            encoding="utf-8"
        )
    )
    rendered_root = Path(effective["data"]["rendered_dir"])
    paths = []
    for record in records:
        paths.extend(
            (
                Path(record["path"]).resolve(),
                (rendered_root / record["rendered_mask_path"]).resolve(),
            )
        )
    snapshot = {str(path): sha256_file(path) for path in paths}
    return FrozenAttentionLoader(
        loader=batches,
        input_snapshot=snapshot,
        ordered_test_cohort_sha256=sha256_json(records),
    )


def _fixture(tmp_path: Path, *, seed: int = 42):
    source_dir = tmp_path / f"phase4-final-s{seed}-source"
    (source_dir / "checkpoints").mkdir(parents=True)
    (source_dir / "config").mkdir()
    (source_dir / "history").mkdir()
    rendered_root = tmp_path / "rendered"
    rendered_resolution = rendered_root / "resolution_64"
    rendered_resolution.mkdir(parents=True)
    image_paths: list[Path] = []
    for index in (1, 2):
        image_path = rendered_resolution / f"n{index}.npy"
        mask_path = rendered_resolution / f"n{index}.npz"
        np.save(image_path, np.arange(4, dtype=np.float32).reshape(2, 2))
        np.savez_compressed(
            mask_path,
            soft_consensus_mask=np.ones((2, 2), dtype=np.float32),
        )
        image_paths.append(image_path.resolve())
    context = Phase4CheckpointContext(
        experiment_id=f"phase4-final-s{seed}-source",
        run_mode="final",
        seed=seed,
        lambda_ordinal=0.5,
        lambda_attention=0.05,
        project_commit=SOURCE_FINAL_MODEL_COMMIT,
        config_sha256="3" * 64,
        manifest_bundle_sha256="4" * 64,
        phase3_input_fingerprint={"ordered_samples_sha256": "5" * 64},
        supervision_sha256="6" * 64,
        ordered_supervision_sha256="7" * 64,
        input_lock_sha256="8" * 64,
        cohort_view="paper_aligned_multi_reader_2_to_4",
        expected_resolution=64,
        sample_count=4,
        split_counts={"train": 1, "val": 1, "test": 2},
    )
    trained = TinyPosthocModel()
    with torch.no_grad():
        trained.classifier.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
        )
    checkpoint = build_phase4_checkpoint(
        checkpoint_kind="best",
        context=context,
        model_state=trained.state_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        scheduler_state=None,
        epoch=1,
        best_epoch=1,
        best_validation_loss=0.25,
        validation_metrics={"loss": 0.25, "total_loss": 0.25},
    )
    best_path = save_phase4_checkpoint(
        checkpoint, source_dir / "checkpoints" / "best.pt"
    )
    # Deliberately invalid: loading it would prove the wrong checkpoint was used.
    (source_dir / "checkpoints" / "last.pt").write_bytes(b"not-a-checkpoint")
    identity = {
        "experiment_id": context.experiment_id,
        "run_mode": context.run_mode,
        "seed": context.seed,
        "lambda_ordinal": context.lambda_ordinal,
        "lambda_attention": context.lambda_attention,
        "project_commit": context.project_commit,
        "configuration_sha256": context.config_sha256,
        "manifest_bundle_sha256": context.manifest_bundle_sha256,
        "phase3_input_fingerprint": dict(context.phase3_input_fingerprint),
        "supervision_sha256": context.supervision_sha256,
        "ordered_supervision_sha256": context.ordered_supervision_sha256,
        "input_lock_sha256": context.input_lock_sha256,
        "cohort_view": context.cohort_view,
        "expected_resolution": context.expected_resolution,
        "sample_count": context.sample_count,
        "split_counts": dict(context.split_counts),
    }
    (source_dir / "config" / "run_identity.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    (source_dir / "config" / "effective_config.json").write_text(
        json.dumps(
            {
                "model": "tiny",
                "data": {
                    "rendered_dir": str(rendered_root.resolve()),
                    "frozen_split_dir": str((tmp_path / "manifests").resolve()),
                },
                "dataset": {
                    "image_size": 2,
                    "in_channels": 1,
                    "batch_size": 2,
                    "num_workers": 0,
                },
                "augmentation": {"eval": {"enabled": False}},
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "history" / "epoch_history.json").write_text(
        json.dumps(
            [
                {
                    "epoch": 1,
                    "train": {"loss": 0.5, "total_loss": 0.5},
                    "validation": {"loss": 0.25, "total_loss": 0.25},
                }
            ]
        ),
        encoding="utf-8",
    )
    snapshot = {
        str(path.resolve()): sha256_file(path)
        for path in source_dir.rglob("*")
        if path.is_file()
    }
    test_rows = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "nodule_id": "n1",
                "nodule_sha256": "a" * 64,
                "series_instance_uid": "s1",
                "split": "test",
                "image_path": str(image_paths[0]),
                "rendered_mask_path": "resolution_64/n1.npz",
                "label": "benign",
                "resolution": 2,
                "q_benign": 1.0,
                "q_indeterminate": 0.0,
                "q_malignant": 0.0,
                "disagreement_entropy": 0.0,
                "ordinal_t1": 0.0,
                "ordinal_t2": 0.0,
                "ordinal_t3": 0.0,
                "ordinal_t4": 0.0,
            },
            {
                "patient_id": "p2",
                "nodule_id": "n2",
                "nodule_sha256": "b" * 64,
                "series_instance_uid": "s2",
                "split": "test",
                "image_path": str(image_paths[1]),
                "rendered_mask_path": "resolution_64/n2.npz",
                "label": "malignant",
                "resolution": 2,
                "q_benign": 0.0,
                "q_indeterminate": 0.0,
                "q_malignant": 1.0,
                "disagreement_entropy": 0.0,
                "ordinal_t1": 1.0,
                "ordinal_t2": 1.0,
                "ordinal_t3": 1.0,
                "ordinal_t4": 1.0,
            },
        ]
    )
    source = SavedPredictionSource(
        model="ConsenDxFormer",
        source_dir=source_dir.resolve(),
        descriptor={
            "experiment_id": context.experiment_id,
            "seed": seed,
            "project_commit": SOURCE_FINAL_MODEL_COMMIT,
            "cohort_view": "paper_aligned_multi_reader_2_to_4",
            "sample_count": 4,
            "test_sample_count": 2,
            "manifest_bundle_sha256": "4" * 64,
            "phase3_input_fingerprint": {"ordered_samples_sha256": "5" * 64},
            "supervision_sha256": "6" * 64,
            "ordered_supervision_sha256": "7" * 64,
            "input_lock_sha256": "8" * 64,
            "artifact_manifest_sha256": "9" * 64,
            "artifact_bundle_sha256": "a" * 64,
        },
        predictions=test_rows.loc[
            :, ["patient_id", "nodule_id", "series_instance_uid", "split"]
        ].copy(),
        supervision=test_rows,
        verification={},
        source_snapshot=snapshot,
    )
    return source, best_path.resolve(), trained.classifier.weight.detach().clone()


def _load(tmp_path: Path, monkeypatch) -> FrozenFinalModel:
    source, _path, _weight = _fixture(tmp_path)
    monkeypatch.setattr(
        phase4_posthoc,
        "load_phase4_final_prediction_source",
        lambda _directory: source,
    )
    return load_frozen_final_best_model(
        source.source_dir,
        device="cpu",
        model_factory=lambda _config: TinyPosthocModel(),
    )


def test_loads_only_persisted_final_best_and_freezes_model(tmp_path, monkeypatch):
    source, best_path, expected_weight = _fixture(tmp_path)
    monkeypatch.setattr(
        phase4_posthoc,
        "load_phase4_final_prediction_source",
        lambda _directory: source,
    )
    original_load = torch.load
    loaded_paths = []

    def tracked_load(path, *args, **kwargs):
        loaded_paths.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    loaded = load_frozen_final_best_model(
        source.source_dir,
        device="cpu",
        model_factory=lambda _config: TinyPosthocModel(),
    )
    assert loaded_paths == [best_path]
    assert loaded.checkpoint_path.name == "best.pt"
    assert loaded.checkpoint_record["checkpoint_kind"] == "best"
    assert loaded.model.training is False
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    torch.testing.assert_close(loaded.model.classifier.weight, expected_weight)
    for path, digest in source.source_snapshot.items():
        assert sha256_file(path) == digest


def test_rejects_nonfinal_source_seed_before_inference(tmp_path, monkeypatch):
    source, _best, _weight = _fixture(tmp_path)
    bad = SavedPredictionSource(
        **{**source.__dict__, "descriptor": {**source.descriptor, "seed": 45}}
    )
    monkeypatch.setattr(
        phase4_posthoc,
        "load_phase4_final_prediction_source",
        lambda _directory: bad,
    )
    with pytest.raises(Phase4PosthocError, match="seed"):
        load_frozen_final_best_model(
            source.source_dir,
            model_factory=lambda _config: TinyPosthocModel(),
        )


def test_attention_inference_is_no_grad_noncausal_and_preserves_identifiers(
    tmp_path, monkeypatch
):
    loaded = _load(tmp_path, monkeypatch)
    batch = {
        "image": torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0]]], [[[3.0, 2.0], [1.0, 0.0]]]]
        ),
        "soft_consensus_mask": torch.tensor(
            [[[[0.0, 0.25], [0.75, 1.0]]], [[[1.0, 0.75], [0.25, 0.0]]]]
        ),
        **_attention_identity_batch(loaded),
    }
    result = evaluate_attention_alignment_from_loader(
        loaded, _sealed_attention_loader(loaded, [batch])
    )
    assert result["causal_explanation_claimed"] is False
    assert result["checkpoint_kind"] == "best"
    assert [row["nodule_id"] for row in result["metrics"]["per_sample"]] == [
        "n1",
        "n2",
    ]
    assert all(row["split"] == "test" for row in result["metrics"]["per_sample"])
    assert all(parameter.grad is None for parameter in loaded.model.parameters())


def test_attention_aggregate_requires_exact_final_seeds(tmp_path, monkeypatch):
    loaded = _load(tmp_path, monkeypatch)
    batch = {
        "image": torch.ones((2, 1, 2, 2)),
        "soft_consensus_mask": torch.ones((2, 1, 2, 2)),
        **_attention_identity_batch(loaded),
    }
    base = evaluate_attention_alignment_from_loader(
        loaded, _sealed_attention_loader(loaded, [batch])
    )
    results = []
    for seed in (42, 43, 44):
        result = copy.deepcopy(base)
        result["seed"] = seed
        result["experiment_id"] = f"phase4-final-s{seed}-source"
        result["checkpoint_sha256"] = f"{seed:064x}"
        results.append(result)
    aggregate = aggregate_attention_seed_results(results)
    assert aggregate["seeds"] == [42, 43, 44]
    assert aggregate["standard_deviation_ddof"] == 1
    with pytest.raises(Phase4PosthocError, match="exactly"):
        aggregate_attention_seed_results(results[:2])

    duplicate_source = copy.deepcopy(results)
    duplicate_source[1]["experiment_id"] = duplicate_source[0]["experiment_id"]
    with pytest.raises(Phase4PosthocError, match="unique source experiment"):
        aggregate_attention_seed_results(duplicate_source)


def test_attention_reconciles_saved_csv_sort_without_changing_inference_order(tmp_path, monkeypatch):
    loaded = _load(tmp_path, monkeypatch)
    loaded = replace(loaded, source=replace(
        loaded.source, predictions=loaded.source.predictions.iloc[::-1].copy(),
    ))
    batch = {
        "image": torch.ones((2, 1, 2, 2)),
        "soft_consensus_mask": torch.ones((2, 1, 2, 2)),
        **_attention_identity_batch(loaded),
    }
    result = evaluate_attention_alignment_from_loader(loaded, _sealed_attention_loader(loaded, [batch]))
    assert [row["nodule_id"] for row in result["metrics"]["per_sample"]] == ["n1", "n2"]


def test_attention_rejects_subset_reordering_and_whitespace_identity(
    tmp_path, monkeypatch
):
    loaded = _load(tmp_path, monkeypatch)
    batch = {
        "image": torch.ones((2, 1, 2, 2)),
        "soft_consensus_mask": torch.ones((2, 1, 2, 2)),
        **_attention_identity_batch(loaded, reverse=True),
    }
    with pytest.raises(Phase4PosthocError, match="frozen test cohort order"):
        evaluate_attention_alignment_from_loader(
            loaded, _sealed_attention_loader(loaded, [batch])
        )

    whitespace = copy.deepcopy(batch)
    whitespace["patient_id"] = [" p1 ", "p2"]
    with pytest.raises(Phase4PosthocError, match="noncanonical"):
        evaluate_attention_alignment_from_loader(
            loaded, _sealed_attention_loader(loaded, [whitespace])
        )


def test_repository_native_attention_loader_is_unaugmented_and_exact(
    tmp_path, monkeypatch
):
    loaded = _load(tmp_path, monkeypatch)
    descriptor = loaded.source.descriptor
    report = SimpleNamespace(
        cohort_view="paper_aligned_multi_reader_2_to_4",
        sample_count=4,
        manifest_bundle_sha256=descriptor["manifest_bundle_sha256"],
        supervision_sha256=descriptor["supervision_sha256"],
        ordered_supervision_sha256=descriptor["ordered_supervision_sha256"],
        phase3_input_fingerprint=SimpleNamespace(
            to_dict=lambda: dict(descriptor["phase3_input_fingerprint"])
        ),
    )
    monkeypatch.setattr(
        phase4_posthoc,
        "validate_frozen_manifest_bundle",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        phase4_posthoc,
        "validate_phase4_readiness",
        lambda *_args, **_kwargs: (report, loaded.source.supervision.copy(deep=True)),
    )
    loader = build_frozen_attention_loader(loaded)

    result = evaluate_attention_alignment_from_loader(loaded, loader)

    assert result["source_cohort"]["sample_count"] == 2
    assert [row["nodule_id"] for row in result["metrics"]["per_sample"]] == [
        "n1",
        "n2",
    ]


def test_boundary_inference_stays_separate_and_has_no_primary_target(
    tmp_path, monkeypatch
):
    from test_phase4_boundary_runtime import _fixture_source

    loaded = _load(tmp_path / "final", monkeypatch)
    boundary, primary, _calls = _fixture_source(tmp_path / "boundary", monkeypatch)
    binding = boundary.provenance["primary_cohort_binding"]
    loaded = replace(
        loaded,
        source=replace(
            loaded.source,
            supervision=primary,
            descriptor={**loaded.source.descriptor, **binding},
        ),
    )
    result = predict_validated_boundary_cohort(loaded, boundary)
    assert result["primary_three_class_targets_assigned"] is False
    assert result["primary_metrics_computed"] is False
    assert result["primary_headline_inclusion_permitted"] is False
    assert result["rows"][0]["mean_malignancy"] == 2.5
    assert "true_class" not in result["rows"][0]

    with pytest.raises(TypeError, match="verified Phase 2 loader"):
        predict_validated_boundary_cohort(loaded, {"ordered_identities": []})
    changed = copy.deepcopy(boundary.provenance)
    changed["sample_count"] += 1
    with pytest.raises(ValueError, match="sample count"):
        predict_validated_boundary_cohort(loaded, replace(boundary, provenance=changed))
