from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.hashing import sha256_json
from consendxformer.manifest_guard import (
    CHECKSUM_LEDGER,
    FROZEN_BUNDLE_SHA256,
    FROZEN_MANIFEST_FILES,
    ManifestIntegrityError,
    validate_frozen_manifest_bundle,
)


MANIFEST_DIR = ROOT / "artifacts" / "phase0" / "manifests"


@pytest.fixture(scope="module")
def report():
    return validate_frozen_manifest_bundle(MANIFEST_DIR)


def _copy_bundle(destination: Path) -> Path:
    destination.mkdir()
    for name in (*FROZEN_MANIFEST_FILES, CHECKSUM_LEDGER):
        shutil.copy2(MANIFEST_DIR / name, destination / name)
    return destination


def test_frozen_bundle_matches_external_source_pin_and_expected_counts(report):
    assert report.bundle_sha256 == FROZEN_BUNDLE_SHA256
    assert report.patient_split["patient_id"].nunique() == 1010
    assert report.series_split["series_instance_uid"].nunique() == 1018
    assert report.split_counts == {"train": 707, "val": 152, "test": 151}
    assert report.to_dict()["bundle_sha256"] == FROZEN_BUNDLE_SHA256


def test_patient_sets_are_disjoint_and_every_series_inherits_its_patient_split(report):
    patient = report.patient_split
    split_sets = {
        split: set(patient.loc[patient["split"] == split, "patient_id"])
        for split in ("train", "val", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])

    expected = patient.set_index("patient_id")["split"]
    inherited = report.series_split["patient_id"].map(expected)
    assert inherited.notna().all()
    assert inherited.equals(report.series_split["split"])


def test_text_files_and_summary_are_exact_semantic_projections(report):
    for split, expected_count in {"train": 707, "val": 152, "test": 151}.items():
        projected = (MANIFEST_DIR / f"{split}_patient_ids.txt").read_text(encoding="utf-8").splitlines()
        from_csv = sorted(
            report.patient_split.loc[report.patient_split["split"] == split, "patient_id"].tolist()
        )
        assert projected == from_csv
        assert len(projected) == expected_count
    assert report.summary["observed_unique_patients"] == 1010
    assert report.summary["observed_unique_ct_series"] == 1018
    assert report.summary["patients_with_multiple_ct_series"] == 8
    assert report.summary["extra_ct_series_beyond_one_per_patient"] == 8


def test_checkout_lf_and_crlf_payloads_have_the_same_frozen_hash(tmp_path):
    copied = _copy_bundle(tmp_path / "manifests")
    target = copied / "patient_split.csv"
    lf_payload = target.read_bytes().replace(b"\r\n", b"\n")
    target.write_bytes(lf_payload.replace(b"\n", b"\r\n"))
    crlf_report = validate_frozen_manifest_bundle(copied)
    assert crlf_report.bundle_sha256 == FROZEN_BUNDLE_SHA256

    target.write_bytes(lf_payload)
    lf_report = validate_frozen_manifest_bundle(copied)
    assert lf_report.bundle_sha256 == FROZEN_BUNDLE_SHA256


def test_modified_manifest_payload_is_rejected(tmp_path):
    copied = _copy_bundle(tmp_path / "manifests")
    target = copied / "train_patient_ids.txt"
    target.write_bytes(target.read_bytes() + b"LIDC-IDRI-9999\n")
    with pytest.raises(ManifestIntegrityError, match="checksum mismatch.*train_patient_ids.txt"):
        validate_frozen_manifest_bundle(copied)


def test_rewritten_ledger_cannot_bless_a_modified_bundle(tmp_path):
    copied = _copy_bundle(tmp_path / "manifests")
    ledger_path = copied / CHECKSUM_LEDGER
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["patient_split.csv"] = "0" * 64
    file_entries = {name: ledger[name] for name in FROZEN_MANIFEST_FILES}
    ledger["bundle_sha256"] = sha256_json(file_entries)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ManifestIntegrityError, match="checksum ledger was modified"):
        validate_frozen_manifest_bundle(copied)


def test_checksum_ledger_has_an_exact_six_file_allowlist(tmp_path):
    copied = _copy_bundle(tmp_path / "manifests")
    ledger_path = copied / CHECKSUM_LEDGER
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["invented.csv"] = "0" * 64
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ManifestIntegrityError, match="exactly the six allowed payload files"):
        validate_frozen_manifest_bundle(copied)


def test_missing_payload_hard_fails(tmp_path):
    copied = _copy_bundle(tmp_path / "manifests")
    (copied / "series_split.csv").unlink()
    with pytest.raises(FileNotFoundError, match="series_split.csv"):
        validate_frozen_manifest_bundle(copied)


def test_guard_never_dereferences_stale_inventory_provenance(monkeypatch):
    stale = str(json.loads((MANIFEST_DIR / "split_summary.json").read_text(encoding="utf-8"))["inventory_csv"])
    original_open = Path.open

    def protected_open(path: Path, *args, **kwargs):
        if str(path) == stale:
            raise AssertionError("stale Phase 0 inventory path was dereferenced")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", protected_open)
    assert validate_frozen_manifest_bundle(MANIFEST_DIR).summary["inventory_csv"] == stale
