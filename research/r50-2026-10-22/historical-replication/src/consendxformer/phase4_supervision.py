"""Phase 4 multi-rater supervision and paired image-mask augmentation."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


_PRIMARY_REQUIRED = {
    "patient_id",
    "image_path",
    "label",
    "nodule_id",
    "series_instance_uid",
    "split",
    "resolution",
    "n_valid_malignancy_ratings",
    "cohort_view",
}

_RENDERED_REQUIRED = {
    "nodule_id",
    "nodule_sha256",
    "series_instance_uid",
    "split",
    "resolution",
    "rendered_mask_path",
}

_CANONICAL_REQUIRED = {
    "series_instance_uid",
    "split",
    "mask_path",
    "q_benign",
    "q_indeterminate",
    "q_malignant",
    "disagreement_entropy",
    "ordinal_t1",
    "ordinal_t2",
    "ordinal_t3",
    "ordinal_t4",
    "source_validity_status",
    "n_valid_malignancy_ratings",
}

_SUPERVISION_COLUMNS = (
    "q_benign",
    "q_indeterminate",
    "q_malignant",
    "disagreement_entropy",
    "ordinal_t1",
    "ordinal_t2",
    "ordinal_t3",
    "ordinal_t4",
    "mask_path",
)


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    role: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{role} is missing required columns: {missing}")


def _canonical_with_nodule_sha(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.copy()
    canonical["nodule_sha256"] = canonical["mask_path"].map(
        lambda value: Path(str(value)).stem
    )
    if canonical["nodule_sha256"].duplicated().any():
        raise ValueError("canonical supervision contains duplicate nodule SHA identities")
    return canonical


def build_supervision_frame(
    primary: pd.DataFrame,
    rendered: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """Join the locked Phase 3 cohort to rendered and canonical Phase 2 evidence.

    The primary cohort remains authoritative for sample membership and ordering.
    Every primary sample must resolve one-to-one to both the rendered artifact
    and canonical multi-rater supervision.
    """

    for frame, required, role in (
        (primary, _PRIMARY_REQUIRED, "primary cohort"),
        (rendered, _RENDERED_REQUIRED, "rendered ledger"),
        (canonical, _CANONICAL_REQUIRED, "canonical supervision"),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{role} must be a pandas DataFrame")
        _require_columns(frame, required, role)

    if primary["nodule_id"].duplicated().any():
        raise ValueError("primary cohort contains duplicate nodule identities")
    if rendered["nodule_id"].duplicated().any():
        raise ValueError("rendered ledger contains duplicate nodule identities")

    canonical_sha = _canonical_with_nodule_sha(canonical)

    rendered_projection = rendered.loc[
        :,
        [
            "nodule_id",
            "nodule_sha256",
            "series_instance_uid",
            "split",
            "resolution",
            "rendered_mask_path",
        ],
    ].copy()

    joined = primary.merge(
        rendered_projection,
        on=["nodule_id", "series_instance_uid", "split", "resolution"],
        how="left",
        validate="one_to_one",
    )

    if joined["nodule_sha256"].isna().any():
        raise ValueError("rendered supervision is missing for one or more primary samples")

    canonical_projection = canonical_sha.loc[
        :,
        [
            "nodule_sha256",
            "series_instance_uid",
            "split",
            "n_valid_malignancy_ratings",
            "q_benign",
            "q_indeterminate",
            "q_malignant",
            "disagreement_entropy",
            "ordinal_t1",
            "ordinal_t2",
            "ordinal_t3",
            "ordinal_t4",
            "mask_path",
            "source_validity_status",
        ],
    ].copy()

    joined = joined.merge(
        canonical_projection,
        on=["nodule_sha256", "series_instance_uid", "split"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_canonical"),
    )

    if joined[list(_SUPERVISION_COLUMNS)].isna().any(axis=1).any():
        raise ValueError(
            "canonical supervision is missing for one or more primary samples"
        )

    if not joined["source_validity_status"].astype(str).str.startswith("valid").all():
        raise ValueError("canonical supervision contains source-invalid samples")

    if "n_valid_malignancy_ratings_canonical" in joined.columns:
        left = pd.to_numeric(
            joined["n_valid_malignancy_ratings"], errors="raise"
        ).astype(int)
        right = pd.to_numeric(
            joined["n_valid_malignancy_ratings_canonical"], errors="raise"
        ).astype(int)
        if not left.equals(right):
            raise ValueError(
                "canonical supervision reader counts disagree with the primary cohort"
            )

    q = joined.loc[:, ["q_benign", "q_indeterminate", "q_malignant"]].astype(float)
    if ((q < 0.0) | (q > 1.0)).any().any():
        raise ValueError("canonical supervision class probabilities must lie in [0, 1]")
    if not np.allclose(q.sum(axis=1).to_numpy(), 1.0, atol=1e-6):
        raise ValueError("canonical supervision class probabilities must sum to 1")

    disagreement = pd.to_numeric(
        joined["disagreement_entropy"], errors="raise"
    ).to_numpy(dtype=float)
    if np.any((disagreement < 0.0) | (disagreement > 1.0)):
        raise ValueError("canonical supervision disagreement entropy must lie in [0, 1]")

    ordinal = joined.loc[
        :, ["ordinal_t1", "ordinal_t2", "ordinal_t3", "ordinal_t4"]
    ].astype(float)
    ordinal_values = ordinal.to_numpy()
    if np.any((ordinal_values < 0.0) | (ordinal_values > 1.0)):
        raise ValueError("canonical supervision ordinal targets must lie in [0, 1]")
    if np.any(np.diff(ordinal_values, axis=1) > 1e-7):
        raise ValueError(
            "canonical supervision ordinal targets must be non-increasing"
        )

    return joined.reset_index(drop=True)


class PairedAugmenter:
    """Apply baseline-equivalent geometry jointly to image and soft mask.

    Geometric operations are shared exactly between image and mask. Intensity
    jitter and Gaussian noise are image-only, matching the released baseline's
    augmentation semantics.
    """

    def __init__(
        self,
        hflip: bool = True,
        vflip: bool = False,
        rotation_deg: float = 15,
        translation_frac: float = 0.05,
        intensity_jitter: float = 0.05,
        gaussian_noise_std: float = 0.01,
        enabled: bool = True,
    ) -> None:
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)
        self.rotation_deg = float(rotation_deg)
        self.translation_frac = float(translation_frac)
        self.intensity_jitter = float(intensity_jitter)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.enabled = bool(enabled)

        for name, value in (
            ("rotation_deg", self.rotation_deg),
            ("translation_frac", self.translation_frac),
            ("intensity_jitter", self.intensity_jitter),
            ("gaussian_noise_std", self.gaussian_noise_std),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        img = np.asarray(image, dtype=np.float32).copy()
        msk = np.asarray(mask, dtype=np.float32).copy()

        if img.ndim != 2 or msk.ndim != 2:
            raise ValueError("paired augmentation requires 2D image and mask arrays")
        if img.shape != msk.shape:
            raise ValueError("paired augmentation requires image and mask shapes to match")

        if not self.enabled:
            return img, msk

        h, w = img.shape

        if self.hflip and random.random() < 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)

        if self.vflip and random.random() < 0.5:
            img = np.flipud(img)
            msk = np.flipud(msk)

        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        tx = random.uniform(-self.translation_frac, self.translation_frac) * w
        ty = random.uniform(-self.translation_frac, self.translation_frac) * h

        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        matrix[:, 2] += [tx, ty]

        img = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        msk = cv2.warpAffine(
            msk,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        scale = 1.0 + random.uniform(
            -self.intensity_jitter,
            self.intensity_jitter,
        )
        shift = random.uniform(
            -self.intensity_jitter,
            self.intensity_jitter,
        )
        img = img * scale + shift

        noise = np.random.normal(
            0,
            self.gaussian_noise_std,
            size=img.shape,
        ).astype(np.float32)
        img = img + noise

        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        msk = np.clip(msk, 0.0, 1.0).astype(np.float32)

        return img, msk
