from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.baseline_metadata_export import export_baseline_metadata
from consendxformer.manifest_guard import validate_frozen_manifest_bundle


@pytest.fixture(scope="module")
def manifest():
    return validate_frozen_manifest_bundle(ROOT / "artifacts" / "phase0" / "manifests")


def _rendered(manifest):
    row = manifest.series_split.iloc[0]
    common = {
        "patient_id": row.patient_id,
        "series_instance_uid": row.series_instance_uid,
        "split": row.split,
        "resolution": 64,
        "paper_primary_eligible": True,
        "n_valid_malignancy_ratings": 2,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "nodule_id": "eligible-two-reader",
                "image_path": "resolution_64/eligible.npy",
                "paper_label": "benign",
                "has_four_valid_readers": False,
            },
            {
                **common,
                "nodule_id": "eligible-four-reader",
                "image_path": "resolution_64/four.npy",
                "paper_label": "malignant",
                "n_valid_malignancy_ratings": 4,
                "has_four_valid_readers": True,
            },
            {
                **common,
                "nodule_id": "boundary",
                "image_path": "resolution_64/boundary.npy",
                "paper_label": "boundary_zone",
                "paper_primary_eligible": False,
                "has_four_valid_readers": True,
            },
        ]
    )


def test_available_reader_and_strict_four_reader_views_remain_separate(tmp_path, manifest):
    rendered = _rendered(manifest)
    available = export_baseline_metadata(
        rendered,
        manifest,
        cohort_view="paper_aligned_available_reader_mean",
        rendered_root=tmp_path,
    )
    strict = export_baseline_metadata(
        rendered,
        manifest,
        cohort_view="paper_aligned_strict_four_reader",
        rendered_root=tmp_path,
    )
    assert set(available["nodule_id"]) == {"eligible-two-reader", "eligible-four-reader"}
    assert strict["nodule_id"].tolist() == ["eligible-four-reader"]
    assert available["cohort_view"].nunique() == 1
    assert available["cohort_view"].iloc[0] == "paper_aligned_available_reader_mean"
    assert "boundary_zone" not in available["label"].tolist()


def test_unknown_patient_or_series_fails_primary_export(tmp_path, manifest):
    rendered = _rendered(manifest).iloc[[0]].copy()
    rendered.loc[rendered.index[0], "series_instance_uid"] = "unknown-series"
    with pytest.raises(ValueError, match="Unknown series"):
        export_baseline_metadata(
            rendered,
            manifest,
            cohort_view="paper_aligned_available_reader_mean",
            rendered_root=tmp_path,
        )
    rendered = _rendered(manifest).iloc[[0]].copy()
    rendered.loc[rendered.index[0], "patient_id"] = "LIDC-IDRI-9999"
    with pytest.raises(ValueError, match="Frozen identity mismatch"):
        export_baseline_metadata(
            rendered,
            manifest,
            cohort_view="paper_aligned_available_reader_mean",
            rendered_root=tmp_path,
        )


def test_export_is_phase1_compatible_and_deterministic(tmp_path, manifest):
    output_path = tmp_path / "baseline.csv"
    output = export_baseline_metadata(
        _rendered(manifest),
        manifest,
        cohort_view="paper_aligned_available_reader_mean",
        rendered_root=tmp_path,
        output_csv=output_path,
    )
    assert list(output.columns[:3]) == ["patient_id", "image_path", "label"]
    assert output["nodule_id"].tolist() == sorted(output["nodule_id"])
    assert output_path.read_text(encoding="utf-8").splitlines()[0].startswith(
        "patient_id,image_path,label"
    )
