"""Synthetic mathematical checks only; no research fixtures or models."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer.phase5_descriptive import (
    CLASS_ORDER, FLAGGED_NODULE, MODELS, SEEDS, accuracy_intervals,
    canonical_predictions, descriptive_protocol, load_protocol, pointing_records,
    pointing_tables, unresolved_hypotheses,
)


def tiny_frames():
    reference = pd.DataFrame({"patient_id": ["a", "a", "b"],
        "nodule_id": [FLAGGED_NODULE, "n2", "n3"], "series_instance_uid": ["sa", "sa", "sb"],
        "split": ["test"] * 3, "true_class": list(CLASS_ORDER),
        "predicted_class": list(CLASS_ORDER)})
    frames = {(m, s): reference.copy(deep=True) for m in MODELS for s in SEEDS}
    frames[("B1", 42)]["predicted_class"] = ["malignant", "benign", "malignant"]
    return reference, frames


def test_pointing_exact_ties_empty_targets_soft_mask_semantics_and_no_mutation():
    frame, _ = tiny_frames()
    maps = np.array([[[[.4, .4], [.1, .1]]], [[[.25, .25], [.25, .25]]], [[[.1, .1], [.1, .7]]]])
    soft = np.array([[[.1, .1], [.1, .1]], [[.5, 0], [0, 0]], [[0, 0], [0, .5]]])
    before = maps.copy(), soft.copy(), frame.copy(deep=True)
    rows = pointing_records(frame, maps, soft, model="B1", seed=42)
    assert [r["hit"] for r in rows] == [0, 1, 1]
    assert rows[0]["tie_count"] == 2 and rows[0]["maximizing_indices"] == "[0,1]"
    assert rows[0]["peak_flat_index"] == 0 and rows[0]["reason"] == "empty_native_binary_target"
    assert rows[1]["constant_map"] and rows[1]["tie_count"] == 4
    assert rows[2]["peak_row"] == rows[2]["peak_column"] == 1
    # Unit-mass normalization would turn a small positive mask into a hit:
    # reviewed pointing must instead use unnormalized soft-mask values.
    soft[0] = [[.1, 0], [0, 0]]
    assert pointing_records(frame, maps, soft, model="B1", seed=42)[0]["hit"] == 0
    np.testing.assert_array_equal(maps, before[0])
    pd.testing.assert_frame_equal(frame, before[2])
    np.testing.assert_array_equal(before[1][1:], soft[1:])


@pytest.mark.parametrize("kind", ["nan", "infinity", "negative", "range", "mass", "shape", "empty", "mask_nan"])
def test_pointing_rejects_invalid_native_inputs(kind):
    frame, _ = tiny_frames()
    maps, soft = np.full((3, 1, 2, 2), .25), np.ones((3, 2, 2))
    if kind in ("nan", "infinity", "negative", "range", "mass"):
        maps[0, 0, 0, 0] = {"nan": np.nan, "infinity": np.inf, "negative": -.1, "range": 2, "mass": 0}[kind]
    elif kind == "shape": maps = maps[:, 0]
    elif kind == "empty": maps = maps[:0]
    else: soft[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        pointing_records(frame, maps, soft, model="B1", seed=42)


def test_pointing_all_seeds_prior_once_and_nine_paired_contrasts():
    frame, _ = tiny_frames()
    soft = np.ones((3, 2, 2)); soft[0] = .1
    maps = {(m, s): np.full((3, 1, 2, 2), .25) for m in MODELS[:2] for s in SEEDS}
    tables = pointing_tables(frame, maps, soft, np.full((2, 2), .25))
    assert len(tables["pointing_per_nodule.csv"]) == 21
    assert len(tables["pointing_per_seed.csv"]) == 7
    assert len(tables["flagged_case.csv"]) == 7
    assert tables["flagged_case.csv"].hit.eq(0).all()
    assert tables["pointing_per_seed.csv"].n_nodules.eq(3).all()
    assert tables["pointing_per_seed.csv"].empty_targets.eq(1).all()
    assert len(tables["pointing_paired_descriptive.csv"]) == 9
    assert tables["pointing_paired_descriptive.csv"].mean_difference.eq(0).all()


def test_accuracy_common_draws_whole_clusters_ratios_quantiles_and_degenerate_cases():
    _, frames = tiny_frames()
    before = {k: v.copy(deep=True) for k, v in frames.items()}
    intervals, arrays, raw = accuracy_intervals(frames)
    assert len(intervals) == 33 and intervals.kind.value_counts().to_dict() == {"individual": 18, "paired_difference": 15}
    assert arrays["replicate_estimates"].shape == (10000, 33)
    assert intervals.patient_draw_sha256.nunique() == 1
    rng = np.random.Generator(np.random.PCG64(20260914))
    expected_draws = np.stack([rng.integers(0, 2, size=2) for _ in range(10000)])
    np.testing.assert_array_equal(arrays["draw_indices"], expected_draws)
    # a carries two wrong B1 nodules, b one correct: ratio is nodule weighted.
    denominators = np.where(expected_draws == 0, 2, 1).sum(axis=1)
    expected_b1 = (expected_draws == 1).sum(axis=1) / denominators
    ids = arrays["endpoint_ids"].tolist()
    b1 = ids.index("accuracy/B1/42")
    paired = ids.index("accuracy_difference/ConsenDxFormer_minus_B1/42")
    np.testing.assert_array_equal(arrays["nodule_denominators"], denominators)
    np.testing.assert_array_equal(arrays["replicate_estimates"][:, b1], expected_b1)
    np.testing.assert_array_equal(arrays["replicate_estimates"][:, paired], 1 - expected_b1)
    row = intervals.iloc[b1]
    assert row.point == pytest.approx(1/3)  # not the equal-patient mean 1/2
    assert (row.lower, row.upper) == tuple(np.quantile(expected_b1, [.025, .975], method="linear"))
    assert intervals.iloc[0][["point", "lower", "upper"]].tolist() == [1, 1, 1]
    assert len(raw) == 54
    for key in frames: pd.testing.assert_frame_equal(frames[key], before[key])
    # All-wrong and class-absent draws stay valid, with no dropped replicates.
    for f in frames.values(): f["predicted_class"] = ["malignant", "benign", "benign"]
    wrong, _, _ = accuracy_intervals(frames)
    assert wrong[["point", "lower", "upper"]].eq(0).all().all()


@pytest.mark.parametrize("field,value", [("patient_id", "changed"), ("series_instance_uid", "changed"),
    ("true_class", "malignant"), ("split", "train"), ("nodule_id", None), ("predicted_class", "unknown")])
def test_accuracy_rejects_pairing_or_invalid_class(field, value):
    _, frames = tiny_frames()
    frames[(MODELS[-1], 44)].loc[0, field] = value
    with pytest.raises(ValueError): canonical_predictions(frames)


def test_accuracy_reorders_exact_rows_but_never_drops_duplicates_or_missing_seeds():
    _, frames = tiny_frames()
    frames[("B1", 42)] = frames[("B1", 42)].iloc[::-1].astype({"true_class": "string"})
    assert canonical_predictions(frames)[("B1", 42)].nodule_id.tolist() == frames[(MODELS[0], 42)].nodule_id.tolist()
    frames[("B1", 42)] = pd.concat([frames[("B1", 42)], frames[("B1", 42)].iloc[:1]])
    with pytest.raises(ValueError): canonical_predictions(frames)
    del frames[("B1", 42)]
    with pytest.raises(ValueError): canonical_predictions(frames)


def test_versioned_protocol_keeps_hypotheses_unresolved_without_statistic_fields(tmp_path):
    import json
    records = unresolved_hypotheses()
    assert len(records) == len({r["hypothesis_id"] for r in records}) == 15
    assert records[0]["hypothesis_id"] == "C_B1_42"
    assert records[-1]["hypothesis_id"] == "C_fusion_off_44"
    assert not any({"Q", "p_raw", "p_adj", "p_value"} & r.keys() for r in records)
    path = tmp_path / "protocol.json"
    policy = descriptive_protocol(); path.write_text(json.dumps(policy))
    assert load_protocol(path) == policy
    policy["accuracy"]["resamples"] = 9999; path.write_text(json.dumps(policy))
    with pytest.raises(ValueError): load_protocol(path)
