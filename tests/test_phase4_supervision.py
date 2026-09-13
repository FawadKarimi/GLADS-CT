from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_supervision import (
    PairedAugmenter,
    build_supervision_frame,
)


def _primary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "LIDC-IDRI-0001",
                "image_path": "resolution_64/example.png",
                "label": "indeterminate",
                "nodule_id": "LIDC-IDRI-0001__nodule-abc123",
                "series_instance_uid": "series-1",
                "split": "train",
                "resolution": 64,
                "n_valid_malignancy_ratings": 4,
                "cohort_view": "paper_aligned_multi_reader_2_to_4",
            }
        ]
    )


def _rendered_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nodule_id": "LIDC-IDRI-0001__nodule-abc123",
                "nodule_sha256": "abc123",
                "series_instance_uid": "series-1",
                "split": "train",
                "resolution": 64,
                "rendered_mask_path": "resolution_64/example_masks.npz",
            }
        ]
    )


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_instance_uid": "series-1",
                "split": "train",
                "mask_path": "masks/abc123.npz",
                "q_benign": 0.25,
                "q_indeterminate": 0.50,
                "q_malignant": 0.25,
                "disagreement_entropy": 0.75,
                "ordinal_t1": 1.00,
                "ordinal_t2": 0.75,
                "ordinal_t3": 0.25,
                "ordinal_t4": 0.25,
                "source_validity_status": "valid",
                "n_valid_malignancy_ratings": 4,
            }
        ]
    )


def test_build_supervision_frame_preserves_exact_primary_identity():
    frame = build_supervision_frame(
        _primary_frame(),
        _rendered_frame(),
        _canonical_frame(),
    )

    assert len(frame) == 1
    row = frame.iloc[0]

    assert row["nodule_id"] == "LIDC-IDRI-0001__nodule-abc123"
    assert row["nodule_sha256"] == "abc123"
    assert row["label"] == "indeterminate"
    assert row["rendered_mask_path"] == "resolution_64/example_masks.npz"

    assert (
        row["q_benign"]
        + row["q_indeterminate"]
        + row["q_malignant"]
    ) == pytest.approx(1.0)

    assert (
        row["ordinal_t1"],
        row["ordinal_t2"],
        row["ordinal_t3"],
        row["ordinal_t4"],
    ) == pytest.approx((1.0, 0.75, 0.25, 0.25))


def test_build_supervision_frame_rejects_missing_canonical_match():
    canonical = _canonical_frame().copy()
    canonical.loc[0, "mask_path"] = "masks/different-sha.npz"

    with pytest.raises(ValueError, match="canonical supervision"):
        build_supervision_frame(
            _primary_frame(),
            _rendered_frame(),
            canonical,
        )


def test_paired_augmenter_applies_identical_horizontal_flip(monkeypatch):
    image = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:, :2] = 1.0

    draws = iter([0.0, 1.0])
    monkeypatch.setattr(
        "consendxformer.phase4_supervision.random.random",
        lambda: next(draws),
    )
    monkeypatch.setattr(
        "consendxformer.phase4_supervision.random.uniform",
        lambda a, b: 0.0,
    )
    monkeypatch.setattr(
        "consendxformer.phase4_supervision.np.random.normal",
        lambda *args, **kwargs: np.zeros(kwargs["size"], dtype=np.float32),
    )

    augmenter = PairedAugmenter(
        hflip=True,
        vflip=True,
        rotation_deg=0.0,
        translation_frac=0.0,
        intensity_jitter=0.0,
        gaussian_noise_std=0.0,
        enabled=True,
    )

    augmented_image, augmented_mask = augmenter(image, mask)

    assert np.array_equal(augmented_image, np.fliplr(image))
    assert np.array_equal(augmented_mask, np.fliplr(mask))
    assert augmented_image.dtype == np.float32
    assert augmented_mask.dtype == np.float32
