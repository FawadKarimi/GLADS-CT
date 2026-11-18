from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_config import load_phase4_config
import consendxformer.phase4_execution as phase4_execution
from consendxformer.phase4_execution import (
    Phase4ExecutionContext,
    build_phase4_initial_datasets,
    execute_phase4_single_run,
)
from consendxformer.reproducibility import seed_dataloader_worker


def _config() -> dict:
    config = deepcopy(
        load_phase4_config(ROOT / "configs" / "phase4_consensdxformer.yaml").phase4_config
    )
    config["training"]["epochs"] = 2
    config["training"]["early_stopping_patience"] = 10
    config["training"]["scheduler_patience"] = 1
    config["dataset"]["batch_size"] = 2
    config["dataset"]["num_workers"] = 0
    return config


def _supervision() -> pd.DataFrame:
    rows = []
    labels = ("benign", "indeterminate", "malignant")
    for split, count in (("train", 6), ("val", 3), ("test", 3)):
        for index in range(count):
            label = labels[index % 3]
            rows.append(
                {
                    "patient_id": f"{split}-p{index}",
                    "nodule_id": f"{split}-n{index}",
                    "series_instance_uid": f"{split}-s{index}",
                    "split": split,
                    "label": label,
                    "image_path": f"unused/{split}-{index}.npy",
                    "rendered_mask_path": f"unused/{split}-{index}.npz",
                    "resolution": 64,
                    "q_benign": float(label == "benign"),
                    "q_indeterminate": float(label == "indeterminate"),
                    "q_malignant": float(label == "malignant"),
                    "disagreement_entropy": 0.0,
                    "ordinal_t1": float(index % 3 >= 1),
                    "ordinal_t2": float(index % 3 >= 1),
                    "ordinal_t3": float(index % 3 >= 2),
                    "ordinal_t4": float(index % 3 >= 2),
                }
            )
    return pd.DataFrame(rows)


class TinyDataset(torch.utils.data.Dataset):
    constructions: list[tuple[tuple[str, ...], object]] = []
    event_log: list[str] = []

    def __init__(
        self,
        frame,
        *,
        image_size,
        in_channels,
        rendered_root,
        augmenter=None,
    ):
        self.frame = frame.reset_index(drop=True).copy(deep=True)
        self.augmenter = augmenter
        splits = tuple(sorted(set(self.frame["split"])))
        self.constructions.append((splits, augmenter))
        self.event_log.append(f"construct:{splits[0]}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        label = ("benign", "indeterminate", "malignant").index(row["label"])
        image = torch.full((1, 2, 2), float(label) / 2.0)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "soft_label": torch.nn.functional.one_hot(
                torch.tensor(label), num_classes=3
            ).float(),
            "disagreement": torch.tensor(0.0),
            "ordinal_target": torch.tensor(
                [label >= 1, label >= 1, label >= 2, label >= 2],
                dtype=torch.float32,
            ),
            "soft_consensus_mask": torch.ones((1, 2, 2)),
            "patient_id": row["patient_id"],
            "nodule_id": row["nodule_id"],
            "series_instance_uid": row["series_instance_uid"],
            "true_class": row["label"],
            "path": row["image_path"],
        }


class TinyModel(torch.nn.Module):
    event_log: list[str] = []

    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = torch.nn.Linear(4, 3)
        self.ordinal = torch.nn.Linear(4, 4)

    def forward(self, image):
        flattened = image.flatten(1)
        logits = self.classifier(flattened)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "ordinal_logits": self.ordinal(flattened),
            "attention_map": torch.softmax(image.flatten(2), dim=2).reshape_as(image),
        }

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.event_log.append("best:loaded")
        return super().load_state_dict(state_dict, *args, **kwargs)


def _context(tmp_path: Path, *, mode: str) -> Phase4ExecutionContext:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(parents=True)
    config = _config()
    if mode == "final":
        config["losses"]["selected_lambda_ordinal"] = 0.1
        config["losses"]["selected_lambda_attention"] = 0.05
    supervision = _supervision()
    if mode == "tuning":
        supervision = supervision.loc[supervision["split"] != "test"].reset_index(
            drop=True
        )
    return Phase4ExecutionContext(
        experiment_id=f"phase4-{mode}-s42-0123456789abcdef",
        run_mode=mode,
        seed=42,
        lambda_ordinal=0.1,
        lambda_attention=0.05,
        configuration_sha256="a" * 64,
        project_commit="b" * 40,
        manifest_bundle_sha256="c" * 64,
        phase3_input_fingerprint={"ordered_samples_sha256": "d" * 64},
        supervision_sha256="e" * 64,
        ordered_supervision_sha256="f" * 64,
        input_lock_sha256="1" * 64,
        cohort_view="paper_aligned_multi_reader_2_to_4",
        expected_resolution=64,
        sample_count=12,
        split_counts={"train": 6, "val": 3, "test": 3},
        effective_config=config,
        supervision=supervision,
        rendered_root=tmp_path,
        layout=SimpleNamespace(checkpoints_dir=checkpoints),
    )


@pytest.fixture(autouse=True)
def _tiny_runtime(monkeypatch):
    TinyDataset.constructions = []
    TinyDataset.event_log = []
    TinyModel.event_log = TinyDataset.event_log
    monkeypatch.setattr(phase4_execution, "ConsenDxDataset", TinyDataset)
    monkeypatch.setattr(phase4_execution, "ConsenDxFormer", TinyModel)


def test_initial_dataset_construction_preserves_splits_and_uses_paired_train_only(tmp_path):
    frames, datasets = build_phase4_initial_datasets(
        _supervision(),
        _config(),
        rendered_root=tmp_path,
        dataset_factory=TinyDataset,
    )
    assert tuple(frames) == ("train", "val", "test")
    assert tuple(datasets) == ("train", "val")
    assert frames["train"]["nodule_id"].tolist() == [f"train-n{i}" for i in range(6)]
    assert frames["val"]["nodule_id"].tolist() == [f"val-n{i}" for i in range(3)]
    assert frames["test"]["nodule_id"].tolist() == [f"test-n{i}" for i in range(3)]
    assert datasets["train"].augmenter is not None
    assert datasets["val"].augmenter is None
    assert not any(splits == ("test",) for splits, _ in TinyDataset.constructions)


def test_loader_carries_explicit_worker_and_generator_seed_controls():
    dataset = torch.utils.data.TensorDataset(torch.arange(4))
    loader = phase4_execution._loader(dataset, _config(), seed=42, shuffle=False)
    assert loader.worker_init_fn is seed_dataloader_worker
    assert loader.generator is not None
    assert loader.generator.initial_seed() == 42
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)


def test_tuning_execution_never_constructs_or_evaluates_test(tmp_path):
    result = execute_phase4_single_run(_context(tmp_path, mode="tuning"))

    assert set(result.predictions) == {"val"}
    assert set(result.metrics) == {"val"}
    assert not any(splits == ("test",) for splits, _ in TinyDataset.constructions)
    assert "test_evaluation_started_after_final_selection_and_best_reload" not in result.events
    assert result.predictions["val"]["split"].unique().tolist() == ["val"]
    assert not (tmp_path / "predictions" / "test.csv").exists()


def test_final_execution_reloads_persisted_best_before_test_construction(tmp_path):
    result = execute_phase4_single_run(_context(tmp_path, mode="final"))

    assert set(result.predictions) == {"val", "test"}
    assert set(result.metrics) == {"val", "test"}
    assert TinyDataset.event_log.index("best:loaded") < TinyDataset.event_log.index(
        "construct:test"
    )
    assert result.events.index("persisted_checkpoints_reloaded_and_validated") < result.events.index(
        "best_checkpoint_reloaded_for_final_validation_and_test"
    )
    assert result.events.index("best_checkpoint_reloaded_for_final_validation_and_test") < result.events.index(
        "test_evaluation_started_after_final_selection_and_best_reload"
    )
    assert result.predictions["test"]["nodule_id"].tolist() == [
        "test-n0",
        "test-n1",
        "test-n2",
    ]


def test_seeded_cpu_execution_is_repeatable(tmp_path):
    first = execute_phase4_single_run(_context(tmp_path / "first", mode="tuning"))
    second = execute_phase4_single_run(_context(tmp_path / "second", mode="tuning"))

    assert first.history == second.history
    pd.testing.assert_frame_equal(first.predictions["val"], second.predictions["val"])
