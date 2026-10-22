"""Phase 4 ConsenDxFormer dataset with multi-rater supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_TO_ID = {
    "benign": 0,
    "indeterminate": 1,
    "malignant": 2,
}

_REQUIRED_COLUMNS = {
    "patient_id",
    "image_path",
    "label",
    "nodule_id",
    "nodule_sha256",
    "series_instance_uid",
    "split",
    "resolution",
    "rendered_mask_path",
    "q_benign",
    "q_indeterminate",
    "q_malignant",
    "disagreement_entropy",
    "ordinal_t1",
    "ordinal_t2",
    "ordinal_t3",
    "ordinal_t4",
}


def _read_image(path: str | Path, image_size: int) -> np.ndarray:
    """Match the released LungDxFormer grayscale loading semantics."""

    source = Path(path)
    if source.suffix.lower() == ".npy":
        image = np.load(source, allow_pickle=False)
    else:
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    image = np.asarray(image, dtype=np.float32)
    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    if image.max() > 1.0:
        image = image / 255.0

    return image.astype(np.float32)


def _read_soft_mask(
    path: str | Path,
    image_size: int,
) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Rendered mask archive does not exist: {source}")

    with np.load(source, allow_pickle=False) as archive:
        if "soft_consensus_mask" not in archive.files:
            raise ValueError(
                f"Rendered mask archive lacks soft_consensus_mask: {source}"
            )
        mask = np.asarray(
            archive["soft_consensus_mask"],
            dtype=np.float32,
        )

    if mask.ndim != 2:
        raise ValueError(
            f"soft_consensus_mask must be a 2D array; got shape {mask.shape}"
        )

    mask = cv2.resize(
        mask,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    return mask


def _label_id(value: Any) -> int:
    if isinstance(value, str):
        label = value.strip().lower()
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unsupported diagnostic class: {value!r}")
        return LABEL_TO_ID[label]

    result = int(value)
    if result not in LABEL_TO_ID.values():
        raise ValueError(f"Unsupported diagnostic class id: {result}")
    return result


class ConsenDxDataset(Dataset):
    """Return baseline-compatible images plus Phase 4 supervision targets."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_size: int,
        in_channels: int = 1,
        rendered_root: str | Path,
        augmenter: Callable[
            [np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray],
        ]
        | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")

        missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                f"Phase 4 dataset metadata is missing required columns: {missing}"
            )

        if type(image_size) is not int or image_size <= 0:
            raise ValueError("image_size must be a positive integer")

        if type(in_channels) is not int or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")

        root = Path(rendered_root).resolve()
        if not root.is_dir():
            raise ValueError(f"rendered_root does not exist: {root}")

        metadata = frame.reset_index(drop=True).copy()

        if metadata["nodule_id"].duplicated().any():
            raise ValueError("Phase 4 dataset metadata contains duplicate nodule_id values")

        self.frame = metadata
        self.image_size = image_size
        self.in_channels = in_channels
        self.rendered_root = root
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]

        image = _read_image(
            row["image_path"],
            self.image_size,
        )

        mask_reference = Path(str(row["rendered_mask_path"]))
        mask_path = (
            mask_reference
            if mask_reference.is_absolute()
            else self.rendered_root / mask_reference
        )
        soft_mask = _read_soft_mask(
            mask_path,
            self.image_size,
        )

        if image.shape != soft_mask.shape:
            raise ValueError(
                "Image and soft consensus mask must have identical spatial shape"
            )

        if self.augmenter is not None:
            image, soft_mask = self.augmenter(
                image,
                soft_mask,
            )

        image = np.asarray(image, dtype=np.float32)
        soft_mask = np.asarray(soft_mask, dtype=np.float32)

        if image.shape != (self.image_size, self.image_size):
            raise ValueError(
                f"Augmented image has unexpected shape {image.shape}"
            )
        if soft_mask.shape != (self.image_size, self.image_size):
            raise ValueError(
                f"Augmented soft consensus mask has unexpected shape {soft_mask.shape}"
            )

        if self.in_channels == 1:
            image_tensor = torch.from_numpy(
                np.ascontiguousarray(image)
            ).unsqueeze(0)
        else:
            image_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    np.stack(
                        [image] * self.in_channels,
                        axis=0,
                    )
                )
            )

        soft_label = torch.tensor(
            [
                float(row["q_benign"]),
                float(row["q_indeterminate"]),
                float(row["q_malignant"]),
            ],
            dtype=torch.float32,
        )

        if not torch.isfinite(soft_label).all():
            raise ValueError("soft_label contains non-finite values")
        if torch.any(soft_label < 0) or torch.any(soft_label > 1):
            raise ValueError("soft_label probabilities must lie in [0, 1]")
        if not torch.isclose(
            soft_label.sum(),
            torch.tensor(1.0, dtype=torch.float32),
            atol=1e-6,
        ):
            raise ValueError("soft_label probabilities must sum to 1")

        disagreement = torch.tensor(
            float(row["disagreement_entropy"]),
            dtype=torch.float32,
        )
        if (
            not torch.isfinite(disagreement)
            or disagreement.item() < 0.0
            or disagreement.item() > 1.0
        ):
            raise ValueError("disagreement must lie in [0, 1]")

        ordinal_target = torch.tensor(
            [
                float(row["ordinal_t1"]),
                float(row["ordinal_t2"]),
                float(row["ordinal_t3"]),
                float(row["ordinal_t4"]),
            ],
            dtype=torch.float32,
        )

        if not torch.isfinite(ordinal_target).all():
            raise ValueError("ordinal_target contains non-finite values")
        if torch.any(ordinal_target < 0) or torch.any(ordinal_target > 1):
            raise ValueError("ordinal_target values must lie in [0, 1]")
        if torch.any(ordinal_target[1:] > ordinal_target[:-1] + 1e-7):
            raise ValueError(
                "ordinal_target values must be non-increasing"
            )

        label_text = str(row["label"]).strip().lower()

        return {
            "image": image_tensor.float(),
            "label": torch.tensor(
                _label_id(row["label"]),
                dtype=torch.long,
            ),
            "soft_label": soft_label,
            "disagreement": disagreement,
            "ordinal_target": ordinal_target,
            "soft_consensus_mask": torch.from_numpy(
                np.ascontiguousarray(soft_mask)
            ).unsqueeze(0).float(),
            "patient_id": str(row["patient_id"]),
            "nodule_id": str(row["nodule_id"]),
            "nodule_sha256": str(row["nodule_sha256"]),
            "series_instance_uid": str(row["series_instance_uid"]),
            "split": str(row["split"]),
            "true_class": label_text,
            "path": str(row["image_path"]),
            "rendered_mask_path": str(row["rendered_mask_path"]),
        }
