from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_dataset import ConsenDxDataset


def _frame(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    rendered_root = tmp_path / "rendered_64"
    mask_dir = rendered_root / "resolution_64"
    mask_dir.mkdir(parents=True)

    image = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    image_path = mask_dir / "example.npy"
    np.save(image_path, image)

    soft_mask = np.zeros((4, 4), dtype=np.float32)
    soft_mask[1:3, 1:3] = 1.0
    mask_path = mask_dir / "example_masks.npz"
    np.savez_compressed(
        mask_path,
        soft_consensus_mask=soft_mask,
        binary_consensus_50=(soft_mask >= 0.5).astype(np.uint8),
        individual_masks=np.stack(
            [
                (soft_mask >= 0.5).astype(np.uint8),
                (soft_mask >= 0.5).astype(np.uint8),
            ],
            axis=0,
        ),
    )

    frame = pd.DataFrame(
        [
            {
                "patient_id": "LIDC-IDRI-0001",
                "image_path": str(image_path.resolve()),
                "label": "indeterminate",
                "nodule_id": "LIDC-IDRI-0001__nodule-abc123",
                "nodule_sha256": "abc123",
                "series_instance_uid": "series-1",
                "split": "train",
                "resolution": 4,
                "rendered_mask_path": "resolution_64/example_masks.npz",
                "q_benign": 0.25,
                "q_indeterminate": 0.50,
                "q_malignant": 0.25,
                "disagreement_entropy": 0.75,
                "ordinal_t1": 1.00,
                "ordinal_t2": 0.75,
                "ordinal_t3": 0.25,
                "ordinal_t4": 0.25,
            }
        ]
    )
    return frame, rendered_root


def test_dataset_returns_complete_multirater_supervision(tmp_path):
    frame, rendered_root = _frame(tmp_path)

    dataset = ConsenDxDataset(
        frame,
        image_size=4,
        in_channels=1,
        rendered_root=rendered_root,
        augmenter=None,
    )

    sample = dataset[0]

    assert sample["image"].shape == (1, 4, 4)
    assert sample["image"].dtype == torch.float32

    assert sample["label"].dtype == torch.long
    assert sample["label"].item() == 1

    assert sample["soft_label"].shape == (3,)
    assert sample["soft_label"].dtype == torch.float32
    assert sample["soft_label"].tolist() == pytest.approx([0.25, 0.50, 0.25])
    assert float(sample["soft_label"].sum()) == pytest.approx(1.0)

    assert sample["disagreement"].dtype == torch.float32
    assert sample["disagreement"].item() == pytest.approx(0.75)

    assert sample["ordinal_target"].shape == (4,)
    assert sample["ordinal_target"].tolist() == pytest.approx(
        [1.00, 0.75, 0.25, 0.25]
    )

    assert sample["soft_consensus_mask"].shape == (1, 4, 4)
    assert sample["soft_consensus_mask"].dtype == torch.float32
    assert sample["soft_consensus_mask"].min().item() >= 0.0
    assert sample["soft_consensus_mask"].max().item() <= 1.0

    assert sample["patient_id"] == "LIDC-IDRI-0001"
    assert sample["nodule_id"] == "LIDC-IDRI-0001__nodule-abc123"
    assert sample["series_instance_uid"] == "series-1"
    assert sample["true_class"] == "indeterminate"


def test_dataset_preserves_baseline_label_order(tmp_path):
    frame, rendered_root = _frame(tmp_path)

    expected = {
        "benign": 0,
        "indeterminate": 1,
        "malignant": 2,
    }

    for label, label_id in expected.items():
        candidate = frame.copy()
        candidate.loc[0, "label"] = label

        dataset = ConsenDxDataset(
            candidate,
            image_size=4,
            in_channels=1,
            rendered_root=rendered_root,
            augmenter=None,
        )

        assert dataset[0]["label"].item() == label_id


def test_dataset_rejects_missing_soft_consensus_mask(tmp_path):
    frame, rendered_root = _frame(tmp_path)

    mask_path = rendered_root / frame.iloc[0]["rendered_mask_path"]
    with np.load(mask_path, allow_pickle=False) as archive:
        binary = archive["binary_consensus_50"]

    np.savez_compressed(mask_path, binary_consensus_50=binary)

    dataset = ConsenDxDataset(
        frame,
        image_size=4,
        in_channels=1,
        rendered_root=rendered_root,
        augmenter=None,
    )

    with pytest.raises(ValueError, match="soft_consensus_mask"):
        dataset[0]
