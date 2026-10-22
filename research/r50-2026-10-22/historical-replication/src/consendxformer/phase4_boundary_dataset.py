"""No-target Dataset for frozen-model Phase 4 boundary inference."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .lidc_targets import paper_mean_label
from .phase4_boundary import (
    BOUNDARY_SOURCE_VALIDITY_STATUSES,
    FROZEN_BOUNDARY_RESOLUTION,
)
from .phase4_dataset import _read_image


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SPLITS = frozenset({"train", "val", "test"})
_REQUIRED_COLUMNS = {
    "patient_id",
    "nodule_id",
    "nodule_sha256",
    "study_instance_uid",
    "series_instance_uid",
    "split",
    "resolution",
    "image_path",
    "rendered_mask_path",
    "mean_malignancy",
    "malignancy_score_sum",
    "n_valid_malignancy_ratings",
    "paper_label",
    "paper_primary_eligible",
    "boundary_zone",
    "boundary_stratum",
    "source_validity_status",
}
_FORBIDDEN_TARGET_COLUMNS = frozenset({"label", "true_class"})


class Phase4BoundaryDatasetError(ValueError):
    """Raised when boundary inference data could introduce an unsafe target."""


def _canonical_text(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Phase4BoundaryDatasetError(
            f"{role} must be a non-empty canonical string without whitespace"
        )
    return value


def _canonical_relative_path(value: Any, role: str) -> str:
    text = _canonical_text(value, role)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise Phase4BoundaryDatasetError(f"{role} must be a canonical relative path")
    return text


def _exact_integer(value: Any, role: str) -> int:
    if isinstance(value, bool):
        raise Phase4BoundaryDatasetError(f"{role} must be an exact integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        return int(value)
    raise Phase4BoundaryDatasetError(f"{role} must be an exact canonical integer")


def _canonical_boolean(value: Any, role: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in {"True", "False", "true", "false"}:
        return value.casefold() == "true"
    raise Phase4BoundaryDatasetError(f"{role} must be a canonical boolean")


class Phase4BoundaryDataset(Dataset):
    """Apply frozen evaluation preprocessing without emitting a class target.

    The source images have already passed Phase 3/4 readiness. This Dataset
    resolves only canonical paths below the validated rendered root, uses the
    same grayscale image reader as ``ConsenDxDataset``, performs no
    augmentation, and deliberately has no ``label`` or ``true_class`` output.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        rendered_root: str | Path,
        image_size: int = FROZEN_BOUNDARY_RESOLUTION,
        in_channels: int = 1,
    ) -> None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise Phase4BoundaryDatasetError("frame must be a non-empty DataFrame")
        missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise Phase4BoundaryDatasetError(
                f"Boundary dataset metadata lacks required columns: {missing}"
            )
        forbidden = sorted(_FORBIDDEN_TARGET_COLUMNS & set(frame.columns))
        if forbidden:
            raise Phase4BoundaryDatasetError(
                f"Boundary dataset must not receive primary target columns: {forbidden}"
            )
        if type(image_size) is not int or image_size != FROZEN_BOUNDARY_RESOLUTION:
            raise Phase4BoundaryDatasetError(
                "Boundary evaluation image_size must be exactly 64"
            )
        if type(in_channels) is not int or in_channels != 1:
            raise Phase4BoundaryDatasetError(
                "Boundary evaluation in_channels must be exactly 1"
            )
        root = Path(rendered_root).resolve()
        if not root.is_dir():
            raise Phase4BoundaryDatasetError(
                f"rendered_root does not exist: {root}"
            )

        metadata = frame.reset_index(drop=True).copy(deep=True)
        identities: list[tuple[str, str]] = []
        for position, row in metadata.iterrows():
            for field in (
                "patient_id",
                "nodule_id",
                "study_instance_uid",
                "series_instance_uid",
                "split",
                "paper_label",
                "boundary_stratum",
                "source_validity_status",
            ):
                _canonical_text(row[field], f"boundary row {position} {field}")
            digest = _canonical_text(
                row["nodule_sha256"], f"boundary row {position} nodule_sha256"
            )
            if _SHA256_RE.fullmatch(digest) is None:
                raise Phase4BoundaryDatasetError(
                    f"boundary row {position} nodule_sha256 is invalid"
                )
            if row["split"] not in _SPLITS:
                raise Phase4BoundaryDatasetError(
                    f"boundary row {position} split is noncanonical"
                )
            if row["source_validity_status"] not in BOUNDARY_SOURCE_VALIDITY_STATUSES:
                raise Phase4BoundaryDatasetError(
                    f"boundary row {position} source status is not renderable"
                )
            if (
                row["paper_label"] != "boundary_zone"
                or _canonical_boolean(
                    row["paper_primary_eligible"],
                    f"boundary row {position} paper_primary_eligible",
                )
                or not _canonical_boolean(
                    row["boundary_zone"],
                    f"boundary row {position} boundary_zone",
                )
            ):
                raise Phase4BoundaryDatasetError(
                    f"boundary row {position} is not an excluded Phase 2 boundary row"
                )
            if _exact_integer(
                row["resolution"], f"boundary row {position} resolution"
            ) != image_size:
                raise Phase4BoundaryDatasetError(
                    f"boundary row {position} resolution differs from frozen preprocessing"
                )
            score_sum = _exact_integer(
                row["malignancy_score_sum"],
                f"boundary row {position} malignancy_score_sum",
            )
            readers = _exact_integer(
                row["n_valid_malignancy_ratings"],
                f"boundary row {position} n_valid_malignancy_ratings",
            )
            try:
                mean = float(row["mean_malignancy"])
            except (TypeError, ValueError) as exc:
                raise Phase4BoundaryDatasetError(
                    f"boundary row {position} mean_malignancy is invalid"
                ) from exc
            label, _eligible, boundary = paper_mean_label(score_sum, readers)
            expected_stratum = (
                "lower_boundary" if score_sum < 3 * readers else "upper_boundary"
            )
            if (
                not math.isfinite(mean)
                or not math.isclose(
                    mean, score_sum / readers, rel_tol=0.0, abs_tol=1e-12
                )
                or label != "boundary_zone"
                or not boundary
                or row["boundary_stratum"] != expected_stratum
            ):
                raise Phase4BoundaryDatasetError(
                    f"boundary row {position} violates the exact Phase 2 integer rule"
                )
            for field in ("image_path", "rendered_mask_path"):
                relative = _canonical_relative_path(
                    row[field], f"boundary row {position} {field}"
                )
                target = root.joinpath(*PurePosixPath(relative).parts).resolve()
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise Phase4BoundaryDatasetError(
                        f"boundary row {position} {field} escapes rendered_root"
                    ) from exc
                if not target.is_file():
                    raise Phase4BoundaryDatasetError(
                        f"boundary row {position} {field} is missing: {relative}"
                    )
            identities.append((row["nodule_id"], digest))
        if len(set(identities)) != len(identities):
            raise Phase4BoundaryDatasetError(
                "Boundary dataset metadata contains duplicate identities"
            )

        self.frame = metadata
        self.rendered_root = root
        self.image_size = image_size
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        relative = PurePosixPath(str(row["image_path"]))
        image_path = self.rendered_root.joinpath(*relative.parts).resolve()
        image = _read_image(image_path, self.image_size)
        if (
            image.shape != (self.image_size, self.image_size)
            or image.dtype != np.float32
            or not bool(np.isfinite(image).all())
            or bool((image < 0.0).any())
            or bool((image > 1.0).any())
        ):
            raise Phase4BoundaryDatasetError(
                f"Boundary image failed frozen preprocessing validation: {image_path}"
            )
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        return {
            "image": image_tensor.float(),
            "patient_id": str(row["patient_id"]),
            "nodule_id": str(row["nodule_id"]),
            "nodule_sha256": str(row["nodule_sha256"]),
            "study_instance_uid": str(row["study_instance_uid"]),
            "series_instance_uid": str(row["series_instance_uid"]),
            "split": str(row["split"]),
            "path": str(row["image_path"]),
            "rendered_mask_path": str(row["rendered_mask_path"]),
            "mean_malignancy": float(row["mean_malignancy"]),
            "malignancy_score_sum": _exact_integer(
                row["malignancy_score_sum"], "malignancy_score_sum"
            ),
            "n_valid_malignancy_ratings": _exact_integer(
                row["n_valid_malignancy_ratings"],
                "n_valid_malignancy_ratings",
            ),
            "boundary_stratum": str(row["boundary_stratum"]),
        }


__all__ = ["Phase4BoundaryDataset", "Phase4BoundaryDatasetError"]
