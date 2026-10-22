"""Explicit Phase 2 cohort views compatible with the Phase 1 metadata adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .lidc_reconciliation import frozen_series_index
from .manifest_guard import ManifestGuardReport
from .phase2_io import write_csv_new


COHORT_VIEWS = (
    "paper_aligned_available_reader_mean",
    "paper_aligned_multi_reader_2_to_4",
    "paper_aligned_strict_four_reader",
)


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    unknown = ~normalized.isin(("true", "false", "1", "0"))
    if bool(unknown.any()):
        raise ValueError(f"Column {name} contains invalid booleans")
    return normalized.isin(("true", "1"))


def export_baseline_metadata(
    rendered_ledger: str | Path | pd.DataFrame,
    manifest: ManifestGuardReport,
    *,
    cohort_view: str,
    rendered_root: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    if cohort_view not in COHORT_VIEWS:
        raise ValueError(f"cohort_view must be one of {COHORT_VIEWS}")
    if isinstance(rendered_ledger, pd.DataFrame):
        frame = rendered_ledger.copy(deep=True)
    else:
        path = Path(rendered_ledger).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Rendered ROI ledger does not exist: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "nodule_id",
        "patient_id",
        "series_instance_uid",
        "split",
        "resolution",
        "image_path",
        "paper_label",
        "paper_primary_eligible",
        "n_valid_malignancy_ratings",
        "has_four_valid_readers",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Rendered ROI ledger lacks required columns: {missing}")
    frozen = frozen_series_index(manifest)
    for row in frame.itertuples(index=False):
        identity = frozen.get(str(row.series_instance_uid))
        if identity is None:
            raise ValueError(f"Unknown series in primary export: {row.series_instance_uid}")
        if identity.patient_id != str(row.patient_id) or identity.split != str(row.split):
            raise ValueError(f"Frozen identity mismatch in primary export: {row.nodule_id}")
    eligible = _as_bool(frame["paper_primary_eligible"], "paper_primary_eligible")
    selected = frame.loc[eligible].copy()
    if cohort_view == "paper_aligned_multi_reader_2_to_4":
        reader_counts = pd.to_numeric(
            selected["n_valid_malignancy_ratings"], errors="raise"
        )
        selected = selected.loc[reader_counts.between(2, 4)].copy()
    if cohort_view == "paper_aligned_strict_four_reader":
        selected = selected.loc[
            _as_bool(selected["has_four_valid_readers"], "has_four_valid_readers")
        ].copy()
    if not selected["paper_label"].isin(("benign", "indeterminate", "malignant")).all():
        raise ValueError("Primary export contains a boundary/unrated label")
    root = Path(rendered_root).resolve()
    image_paths: list[str] = []
    for value in selected["image_path"]:
        target = (root / str(value)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Rendered image path escapes rendered root: {value}") from exc
        image_paths.append(str(target))
    output = pd.DataFrame(
        {
            "patient_id": selected["patient_id"].astype(str).tolist(),
            "image_path": image_paths,
            "label": selected["paper_label"].astype(str).tolist(),
            "nodule_id": selected["nodule_id"].astype(str).tolist(),
            "series_instance_uid": selected["series_instance_uid"].astype(str).tolist(),
            "split": selected["split"].astype(str).tolist(),
            "resolution": selected["resolution"].astype(str).tolist(),
            "n_valid_malignancy_ratings": selected[
                "n_valid_malignancy_ratings"
            ].astype(str).tolist(),
            "cohort_view": cohort_view,
        }
    )
    output = output.sort_values("nodule_id", kind="stable").reset_index(drop=True)
    if output_csv is not None:
        write_csv_new(output_csv, output)
    return output
