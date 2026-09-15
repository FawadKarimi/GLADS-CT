"""Exact cohort matching across the actual Phase 3/4 text-dtype boundary."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.hashing import sha256_file
from consendxformer.phase3_predictions import (
    build_prediction_table, load_phase3_prediction_csv, write_prediction_csv,
)
from consendxformer.phase4_predictions import (
    build_phase4_prediction_table, load_phase4_prediction_csv,
    write_phase4_prediction_csv,
)
from consendxformer.phase4_evaluation_runner import (
    Phase4EvaluationError, validate_seeded_test_cohorts,
)

SEEDS = (42, 43, 44)
FIELDS = ("patient_id", "nodule_id", "series_instance_uid", "split", "true_class")


def _tables():
    phase4, b1 = {}, {}
    for seed in SEEDS:
        frame = pd.DataFrame({
            "patient_id": ["p1", "p1", "p2"],
            "nodule_id": ["n1", "n2", "n3"],
            "series_instance_uid": ["s1", "s1", "s2"],
            "split": ["test"] * 3,
            "true_class": ["benign", "indeterminate", "malignant"],
        }, dtype=object)
        frame["seed"] = seed
        phase4[seed] = frame.copy(deep=True)
        b1[seed] = frame.copy(deep=True)
        b1[seed]["true_class"] = b1[seed]["true_class"].astype("string")
    return phase4, b1


def _assert_unchanged(actual, before):
    for seed in SEEDS:
        pd.testing.assert_frame_equal(actual[seed], before[seed], check_exact=True)


@pytest.mark.parametrize("all_text_columns", [False, True])
def test_text_storage_and_row_order_do_not_change_exact_cohort(all_text_columns):
    phase4, b1 = _tables()
    for seed in SEEDS:
        if all_text_columns:
            b1[seed] = b1[seed].astype({field: "string" for field in FIELDS})
        b1[seed] = b1[seed].iloc[[2, 0, 1]]
    original_phase4 = {seed: frame.copy(deep=True) for seed, frame in phase4.items()}
    original_b1 = {seed: frame.copy(deep=True) for seed, frame in b1.items()}
    report = validate_seeded_test_cohorts(phase4, b1)
    assert report["matched"] is True
    assert report["seeds"] == list(SEEDS)
    assert report["sample_count_per_seed"] == 3
    _assert_unchanged(phase4, original_phase4)
    _assert_unchanged(b1, original_b1)


def test_actual_prediction_csv_loaders_match_without_dtype_mutation(tmp_path):
    phase4, b1 = {}, {}
    paths = []
    metadata = _tables()[0][42].drop(columns="seed").rename(columns={"true_class": "label"})
    logits = np.eye(3, dtype=np.float64) * 3.0
    for seed in SEEDS:
        b1_path = write_prediction_csv(build_prediction_table(
            metadata, logits, seed=seed, baseline_variant="B1",
            experiment_id=f"phase3-b1-s{seed}-fixture",
        ), tmp_path / f"b1-{seed}.csv")
        cdx_path = write_phase4_prediction_csv(build_phase4_prediction_table(
            metadata, logits, seed=seed, lambda_ordinal=0.5, lambda_attention=0.05,
            run_mode="final", experiment_id=f"phase4-final-s{seed}-fixture",
        ), tmp_path / f"cdx-{seed}.csv")
        b1[seed] = load_phase3_prediction_csv(b1_path)
        phase4[seed] = load_phase4_prediction_csv(cdx_path)
        paths.extend((b1_path, cdx_path))
        # No string conversion, trimming or set-only comparison in this proof.
        assert Counter(b1[seed].loc[:, FIELDS].itertuples(index=False, name=None)) == Counter(
            phase4[seed].loc[:, FIELDS].itertuples(index=False, name=None)
        )
    snapshots = {path: sha256_file(path) for path in paths}
    original_phase4 = {seed: frame.copy(deep=True) for seed, frame in phase4.items()}
    original_b1 = {seed: frame.copy(deep=True) for seed, frame in b1.items()}
    assert validate_seeded_test_cohorts(phase4, b1)["sample_count_per_seed"] == 3
    _assert_unchanged(phase4, original_phase4)
    _assert_unchanged(b1, original_b1)
    assert {path: sha256_file(path) for path in paths} == snapshots


@pytest.mark.parametrize("field,value", [
    ("true_class", "malignant"), ("patient_id", "other"),
    ("nodule_id", "other"), ("series_instance_uid", "other"),
    ("patient_id", " p1"), ("nodule_id", "N1"),
])
def test_genuine_text_differences_are_not_normalized_away(field, value):
    phase4, b1 = _tables()
    b1[43].loc[0, field] = value
    with pytest.raises(Phase4EvaluationError, match="cohort mismatch"):
        validate_seeded_test_cohorts(phase4, b1)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_row_counts_and_duplicate_identity_guard_remain_strict(change):
    phase4, b1 = _tables()
    if change == "missing":
        b1[44] = b1[44].iloc[:-1].copy()
    else:
        extra = b1[44].iloc[[0]].copy()
        if change == "extra":
            extra["nodule_id"] = "new-nodule"
        b1[44] = pd.concat([b1[44], extra], ignore_index=True)
        if change == "duplicate":
            # Identical duplicates on BOTH sides must not be hidden by sets.
            phase4[44] = pd.concat([phase4[44], phase4[44].iloc[[0]]], ignore_index=True)
    with pytest.raises(Phase4EvaluationError, match="duplicate|cohort mismatch"):
        validate_seeded_test_cohorts(phase4, b1)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [None, np.nan, pd.NA, "", " \t\n", 123, ["p1"]])
def test_invalid_matching_values_rejected_before_text_conversion(field, value):
    phase4, b1 = _tables()
    # Put the same invalid object on both sides: coercion must not bless it.
    for mapping in (phase4, b1):
        mapping[42][field] = mapping[42][field].astype(object)
        mapping[42].at[0, field] = value
    with pytest.raises(Phase4EvaluationError):
        validate_seeded_test_cohorts(phase4, b1)


@pytest.mark.parametrize("change", ["split", "missing_seed", "wrong_seed", "fractional_seed", "cross_seed", "label_alias", "missing_field", "duplicate_column"])
def test_remaining_cohort_guards_fail_closed(change):
    phase4, b1 = _tables()
    if change == "missing_seed":
        del b1[44]
    elif change == "wrong_seed":
        b1[43]["seed"] = 42
    elif change == "fractional_seed":
        b1[43]["seed"] = 43.5
    elif change == "missing_field":
        b1[43] = b1[43].drop(columns="patient_id")
    elif change == "duplicate_column":
        b1[43] = pd.concat([b1[43], b1[43][["patient_id"]]], axis=1)
    else:
        field, value = {
            "split": ("split", "val"),
            "cross_seed": ("nodule_id", "different-in-both-models"),
            "label_alias": ("true_class", "Benign"),
        }[change]
        for mapping in (phase4, b1):
            mapping[43].loc[0, field] = value
    with pytest.raises(Phase4EvaluationError):
        validate_seeded_test_cohorts(phase4, b1)
