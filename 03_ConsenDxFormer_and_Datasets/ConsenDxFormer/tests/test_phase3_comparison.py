from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase3_comparison import (
    FairComparisonError,
    build_comparison_descriptor,
    validate_fair_baseline_pair,
)


def _descriptor(variant: str) -> dict[str, object]:
    return {
        "baseline_variant": variant,
        "seed": 2026,
        "semantic_cohort_checksum": "a" * 64,
        "frozen_split_checksum": "b" * 64,
        "ordered_sample_checksum": "c" * 64,
        "class_order": ["benign", "indeterminate", "malignant"],
        "tta_enabled": False,
        "experiment_id": f"phase3-{variant.lower()}-seed2026",
    }


def test_fair_pair_ignores_expected_locked_hyperparameter_differences() -> None:
    b0 = _descriptor("B0")
    b1 = _descriptor("B1")
    b0.update(
        config_sha256="0" * 64,
        hyperparameters={"input_size": 96, "epochs": 6, "dropout": 0.2},
    )
    b1.update(
        config_sha256="1" * 64,
        hyperparameters={"input_size": 64, "epochs": 100, "dropout": 0.5},
    )

    report = validate_fair_baseline_pair(b0, b1)

    assert report["eligible"] is True
    assert report["baseline_variants"] == ["B0", "B1"]
    assert report["seed"] == 2026
    assert report["tta_enabled"] is False


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("semantic_cohort_checksum", "d" * 64, "semantic cohort checksum mismatch"),
        ("frozen_split_checksum", "d" * 64, "frozen split checksum mismatch"),
        ("seed", 2027, "seed mismatch"),
        ("ordered_sample_checksum", "d" * 64, "ordered sample checksum mismatch"),
        (
            "class_order",
            ["malignant", "indeterminate", "benign"],
            "class_order must exactly equal",
        ),
    ],
)
def test_unfair_pair_mismatches_are_rejected(
    field: str, replacement: object, message: str
) -> None:
    b0 = _descriptor("B0")
    b1 = _descriptor("B1")
    b1[field] = replacement

    with pytest.raises(FairComparisonError, match=message):
        validate_fair_baseline_pair(b0, b1)


def test_tta_must_be_equivalent_and_disabled() -> None:
    b0 = _descriptor("B0")
    b1 = _descriptor("B1")
    b1["tta_enabled"] = True
    with pytest.raises(FairComparisonError, match="TTA policy mismatch"):
        validate_fair_baseline_pair(b0, b1)

    b0["tta_enabled"] = True
    with pytest.raises(FairComparisonError, match="TTA must be disabled"):
        validate_fair_baseline_pair(b0, b1)


def test_pair_must_contain_exactly_one_b0_and_one_b1() -> None:
    with pytest.raises(FairComparisonError, match="exactly one B0 and one B1"):
        validate_fair_baseline_pair(_descriptor("B0"), _descriptor("B0"))


def test_descriptor_builder_canonicalizes_checksum_case_without_mutating_inputs() -> None:
    class_order = ["benign", "indeterminate", "malignant"]
    snapshot = deepcopy(class_order)

    descriptor = build_comparison_descriptor(
        baseline_variant="B0",
        seed=3,
        semantic_cohort_checksum="A" * 64,
        frozen_split_checksum="B" * 64,
        ordered_sample_checksum="C" * 64,
        class_order=class_order,
        tta_enabled=False,
    )

    assert descriptor["semantic_cohort_checksum"] == "a" * 64
    assert descriptor["frozen_split_checksum"] == "b" * 64
    assert descriptor["ordered_sample_checksum"] == "c" * 64
    assert class_order == snapshot
