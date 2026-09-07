from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.baseline_metadata import (
    CANONICAL_COLUMNS,
    CLASS_NAMES,
    MetadataAdapterError,
    adapt_metadata,
    write_adapted_metadata,
)
from consendxformer.manifest_guard import validate_frozen_manifest_bundle


MANIFEST_DIR = ROOT / "artifacts" / "phase0" / "manifests"


@pytest.fixture(scope="module")
def manifest_report():
    return validate_frozen_manifest_bundle(MANIFEST_DIR)


@pytest.fixture(scope="module")
def one_patient_per_split(manifest_report):
    patients = manifest_report.patient_split
    return {
        split: patients.loc[patients["split"] == split, "patient_id"].iloc[0]
        for split in ("train", "val", "test")
    }


def test_adapter_joins_many_samples_to_frozen_patient_split_without_touching_images(
    manifest_report, one_patient_per_split, tmp_path
):
    missing_image = tmp_path / "images-do-not-exist" / "roi-001.png"
    source = pd.DataFrame(
        {
            "patient_id": [
                one_patient_per_split["test"],
                one_patient_per_split["train"],
                one_patient_per_split["train"],
                one_patient_per_split["val"],
            ],
            "image_path": [str(missing_image), "train-b.png", "train-a.png", "val.png"],
            "label": ["benign", "1", 2, 0.0],
        }
    )

    adapted = adapt_metadata(source, manifest_report)
    assert tuple(adapted.columns) == CANONICAL_COLUMNS
    assert adapted["patient_id"].tolist() == source["patient_id"].tolist()
    assert adapted["split"].tolist() == ["test", "train", "train", "val"]
    assert adapted["label"].tolist() == ["benign", "indeterminate", "malignant", "benign"]
    assert adapted["label_id"].tolist() == [0, 1, 2, 0]
    assert not missing_image.exists()


def test_adapter_accepts_configured_equivalent_columns_and_csv_input(
    manifest_report, one_patient_per_split, tmp_path
):
    source = pd.DataFrame(
        {
            "Subject ID": [one_patient_per_split["val"]],
            "ROI file": ["future/roi.png"],
            "target": [2],
        }
    )
    csv_path = tmp_path / "future_phase2_metadata.csv"
    source.to_csv(csv_path, index=False)

    adapted = adapt_metadata(
        csv_path,
        manifest_report,
        columns={"patient_id": "Subject ID", "image_path": "ROI file", "label": "target"},
    )
    assert adapted.to_dict("records") == [
        {
            "patient_id": one_patient_per_split["val"],
            "image_path": "future/roi.png",
            "label": "malignant",
            "label_id": 2,
            "split": "val",
        }
    ]


def test_csv_preserves_leading_zero_patient_ids_and_normalizes_safe_labels(
    manifest_report, tmp_path
):
    patient_split = manifest_report.patient_split.copy()
    # The real IDs are prefixed, so exercise label normalization on a real row
    # and identifier preservation independently through the CSV reader.
    known = patient_split.iloc[0]["patient_id"]
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        f"patient_id,image_path,label\n{known},roi.png, Benign \n",
        encoding="utf-8",
    )
    adapted = adapt_metadata(csv_path, manifest_report)
    assert adapted.loc[0, "patient_id"] == known
    assert adapted.loc[0, "label"] == "benign"
    assert adapted.loc[0, "label_id"] == 0

    numeric_path = tmp_path / "numeric-identifiers.csv"
    numeric_path.write_text("patient_id,image_path,label\n0001,roi.png,1.0\n", encoding="utf-8")
    # The unknown-ID error must display 0001, proving CSV parsing did not coerce it to 1.
    with pytest.raises(MetadataAdapterError, match="0001"):
        adapt_metadata(numeric_path, manifest_report)


def test_unknown_patient_ids_error_by_default(manifest_report):
    source = pd.DataFrame(
        {"patient_id": ["UNKNOWN-0001"], "image_path": ["roi.png"], "label": ["benign"]}
    )
    with pytest.raises(MetadataAdapterError, match="absent from the frozen split.*UNKNOWN-0001"):
        adapt_metadata(source, manifest_report)


def test_unknown_patient_ids_are_dropped_only_when_explicit(
    manifest_report, one_patient_per_split
):
    known = one_patient_per_split["test"]
    source = pd.DataFrame(
        {
            "patient_id": ["UNKNOWN-0001", known, "UNKNOWN-0002"],
            "image_path": ["unknown-a.png", "known.png", "unknown-b.png"],
            "label": [0, 1, 2],
        }
    )
    adapted = adapt_metadata(source, manifest_report, unknown_patient_policy="drop")
    assert adapted["patient_id"].tolist() == [known]
    assert adapted["image_path"].tolist() == ["known.png"]
    assert adapted["split"].tolist() == ["test"]


@pytest.mark.parametrize(
    "invalid_label",
    ["uncertain", -1, 3, 1.5, True, None],
)
def test_invalid_or_noncanonical_labels_are_rejected(
    manifest_report, one_patient_per_split, invalid_label
):
    source = pd.DataFrame(
        {
            "patient_id": [one_patient_per_split["train"]],
            "image_path": ["roi.png"],
            "label": [invalid_label],
        }
    )
    with pytest.raises(MetadataAdapterError, match="label"):
        adapt_metadata(source, manifest_report)


def test_column_mapping_and_unknown_policy_are_strict(manifest_report):
    source = pd.DataFrame({"p": [], "path": [], "target": []})
    with pytest.raises(MetadataAdapterError, match="must define exactly"):
        adapt_metadata(source, manifest_report, columns={"patient_id": "p", "label": "target"})
    with pytest.raises(MetadataAdapterError, match="unknown_patient_policy"):
        adapt_metadata(source, manifest_report, unknown_patient_policy="ignore")


def test_empty_metadata_and_drop_to_empty_are_rejected(manifest_report):
    empty = pd.DataFrame(columns=["patient_id", "image_path", "label"])
    with pytest.raises(MetadataAdapterError, match="no sample rows"):
        adapt_metadata(empty, manifest_report)

    unknown = pd.DataFrame(
        {"patient_id": ["UNKNOWN"], "image_path": ["roi.png"], "label": [0]}
    )
    with pytest.raises(MetadataAdapterError, match="No metadata rows remain"):
        adapt_metadata(unknown, manifest_report, unknown_patient_policy="drop")


def test_writes_deterministic_combined_and_per_split_metadata(
    manifest_report, one_patient_per_split, tmp_path
):
    source = pd.DataFrame(
        {
            "patient_id": [
                one_patient_per_split["test"],
                one_patient_per_split["train"],
                one_patient_per_split["val"],
                one_patient_per_split["train"],
            ],
            "image_path": ["test.png", "train-2.png", "val.png", "train-1.png"],
            "label": [2, 1, 0, 2],
        }
    )
    adapted = adapt_metadata(source, manifest_report)
    first = write_adapted_metadata(adapted, tmp_path / "first")
    second = write_adapted_metadata(adapted, tmp_path / "second")

    assert first.combined_csv.name == "metadata.csv"
    assert first.combined_csv.read_bytes() == second.combined_csv.read_bytes()
    assert b"\r\n" not in first.combined_csv.read_bytes()
    assert first.row_counts == {"all": 4, "train": 2, "val": 1, "test": 1}
    assert set(first.split_csvs) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert first.split_csvs[split].read_bytes() == second.split_csvs[split].read_bytes()
        split_frame = pd.read_csv(first.split_csvs[split])
        assert split_frame["split"].eq(split).all()

    round_trip = pd.read_csv(first.combined_csv)
    pd.testing.assert_frame_equal(round_trip, adapted, check_dtype=False)
    assert first.to_dict()["combined_csv"] == str(first.combined_csv)


def test_writer_rejects_inconsistent_label_projection(
    manifest_report, one_patient_per_split, tmp_path
):
    adapted = adapt_metadata(
        pd.DataFrame(
            {
                "patient_id": [one_patient_per_split["train"]],
                "image_path": ["roi.png"],
                "label": ["benign"],
            }
        ),
        manifest_report,
    )
    adapted.loc[0, "label_id"] = 2
    with pytest.raises(MetadataAdapterError, match="disagree"):
        write_adapted_metadata(adapted, tmp_path / "bad")


def test_adapter_does_not_mutate_its_input(manifest_report, one_patient_per_split):
    source = pd.DataFrame(
        {
            "patient_id": [f" {one_patient_per_split['train']} "],
            "image_path": [" roi.png "],
            "label": [CLASS_NAMES[0]],
        }
    )
    before = source.copy(deep=True)
    adapted = adapt_metadata(source, manifest_report)
    pd.testing.assert_frame_equal(source, before)
    assert adapted.loc[0, "patient_id"] == one_patient_per_split["train"]
    assert adapted.loc[0, "image_path"] == "roi.png"


def test_adapter_revalidates_bundle_instead_of_trusting_mutated_report(tmp_path):
    copied_bundle = tmp_path / "manifests"
    shutil.copytree(MANIFEST_DIR, copied_bundle)
    report = validate_frozen_manifest_bundle(copied_bundle)

    train_index = report.patient_split.index[report.patient_split["split"] == "train"][0]
    val_index = report.patient_split.index[report.patient_split["split"] == "val"][0]
    train_patient = report.patient_split.loc[train_index, "patient_id"]

    # A count-preserving swap evades checks that only inspect total split sizes.
    report.patient_split.loc[[train_index, val_index], "split"] = ["val", "train"]
    assert report.split_counts == {"train": 707, "val": 152, "test": 151}

    adapted = adapt_metadata(
        pd.DataFrame(
            {
                "patient_id": [train_patient],
                "image_path": ["roi.png"],
                "label": ["benign"],
            }
        ),
        report,
    )

    assert adapted.loc[0, "split"] == "train"
