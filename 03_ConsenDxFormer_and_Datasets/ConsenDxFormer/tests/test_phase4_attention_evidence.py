from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_attention_evidence import (
    FLAGGED_NODULE, IDENTITY, METRICS, array_record, build_location_prior,
    map_evidence, paired_alignment_bootstrap, select_panel_cases,
    supplementary_protocol, write_map_archive, write_overlay_panel,
)
from consendxformer.phase4_evaluation_metrics import ATTENTION_ALIGNMENT_DEFINITION, attention_alignment_metrics, _area_resize
from consendxformer.phase4_bootstrap import patient_cluster_resample_indices
from consendxformer.phase4_attention_comparison import compare_scalar_records


def _identities(split="test"):
    return pd.DataFrame({"patient_id": ["p1", "p1", "p2"], "nodule_id": [FLAGGED_NODULE, "n2", "n3"],
                         "series_instance_uid": ["s1", "s1", "s2"], "split": [split] * 3,
                         "true_class": ["benign", "indeterminate", "malignant"]})


def test_protocol_is_separate_explicitly_exploratory_and_definitions_unchanged():
    protocol = supplementary_protocol()
    assert json.loads((ROOT / "configs/phase4_attention_evidence_v1.json").read_text()) == protocol
    assert protocol["preregistered"] is False
    assert protocol["tuning_permitted"] is False
    assert protocol["alignment_definition"] == ATTENTION_ALIGNMENT_DEFINITION
    assert protocol["alignment_definition"]["dice_iou"]["attention_quantile"] == 0.9
    assert protocol["bootstrap"]["resamples"] == 10000
    assert protocol["source_final_model_commit"] == "e1b174490245c616e355ef7c0c6f7c3c239d3acd"


def test_prior_is_mean_of_train_distributions_not_mean_of_raw_masks():
    rows = _identities("train")
    masks = np.array([[[1., 0], [0, 0]], [[1., 1], [0, 0]], [[0., 0], [1, 1]]])
    prior, record = build_location_prior(rows, masks, (2, 2), expected_count=3)
    np.testing.assert_allclose(prior, [[0.5, 1/6], [1/6, 1/6]])
    other, other_record = build_location_prior(rows.iloc[::-1], masks[::-1], (2, 2), expected_count=3)
    np.testing.assert_array_equal(other, prior)
    assert other_record == record
    assert record["test_masks_used"] is False
    assert record["array"] == array_record(prior)
    assert not np.allclose(prior, masks.mean(axis=0) / masks.mean(axis=0).sum())
    resized, _ = build_location_prior(rows, masks, (1, 1), expected_count=3)
    np.testing.assert_array_equal(resized, [[1.]])


@pytest.mark.parametrize("change", ["test", "val", "duplicate", "missing"])
def test_prior_rejects_nontraining_or_wrong_membership(change):
    rows = _identities("train")
    masks = np.ones((3, 2, 2))
    if change in {"test", "val"}:
        rows.loc[0, "split"] = change
    elif change == "duplicate":
        rows.loc[1, "nodule_id"] = rows.loc[0, "nodule_id"]
    else:
        rows = rows.iloc[:2]
    with pytest.raises(ValueError):
        build_location_prior(rows, masks, (2, 2), expected_count=3)


def test_maps_reuse_metrics_and_distinguish_empty_target_from_disjoint_sets():
    ids = _identities().to_dict("records")
    attention = np.array([[[.25, .25], [.25, .25]], [[1., 0], [0, 0]], [[0., 0], [0, 1.]]], dtype=np.float32)
    masks = np.array([[[.1, .1], [.1, .1]], [[0., 0], [0, 1]], [[0., 0], [0, 1.]]])
    result = map_evidence(attention, masks, ids)
    assert result["metrics"] == attention_alignment_metrics(attention, masks, identifiers=ids)
    empty, disjoint, aligned = result["rows"]
    assert empty["attention_foreground_count"] == 4  # ties are NOT top-k
    assert empty["consensus_foreground_count"] == 0
    assert empty["dice"] == empty["iou"] == 0
    assert empty["zero_overlap_reason"] == "empty_target_foreground"
    assert disjoint["consensus_foreground_count"] == disjoint["attention_foreground_count"] == 1
    assert disjoint["intersection"] == 0 and disjoint["union"] == 2
    assert disjoint["zero_overlap_reason"] == "disjoint_nonempty_foregrounds"
    assert aligned["dice"] == aligned["iou"] == 1
    assert empty["attention_constant"] and empty["correlation_undefined"]
    np.testing.assert_allclose(result["target_distributions"].sum(axis=(1, 2)), 1)


def test_coarse_resizing_can_have_positive_mass_but_empty_foreground():
    masks = np.zeros((1, 4, 4)); masks[0, 0, 0] = 1
    result = map_evidence(np.full((1, 2, 2), .25), masks, _identities().iloc[:1].to_dict("records"))
    assert result["rows"][0]["zero_overlap_reason"] == "empty_target_foreground"
    np.testing.assert_array_equal(result["resized_soft_masks"][0], _area_resize(masks[0], (2, 2)))


def test_npz_roundtrip_preserves_raw_dtype_identifiers_hash_and_no_overwrite(tmp_path):
    arrays = {"attention": np.full((3, 1, 2, 2), .25, dtype=np.float32),
              "nodule_ids": np.asarray(_identities()["nodule_id"], dtype=np.str_)}
    path = tmp_path / "maps.npz"
    record = write_map_archive(path, arrays)
    with np.load(path, allow_pickle=False) as archive:
        for name, array in arrays.items():
            np.testing.assert_array_equal(archive[name], array)
            assert record["arrays"][name] == array_record(archive[name])
    second = tmp_path / "same.npz"
    write_map_archive(second, arrays)
    assert path.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        write_map_archive(path, arrays)
    with pytest.raises(ValueError, match="Object arrays"):
        write_map_archive(tmp_path / "bad.npz", {"bad": np.array([{}], dtype=object)})


def _metric_rows():
    rows = _identities()
    for metric in METRICS:
        rows[metric] = [1., 2., 4.]
    return rows


def test_paired_patient_bootstrap_matches_explicit_cluster_expansion():
    left = _metric_rows()
    right = _metric_rows()
    right.loc[:, list(METRICS)] = 0.
    result = paired_alignment_bootstrap(left, right.iloc[::-1])
    assert result == paired_alignment_bootstrap(left.iloc[::-1], right)
    canonical = left.sort_values(list(IDENTITY)).reset_index(drop=True)
    patients = sorted(canonical["patient_id"].unique())
    rng = np.random.Generator(np.random.PCG64(20260914))
    draws = rng.integers(0, len(patients), size=(10000, len(patients)))
    # Two patients means three possible cluster-multiplicity results; compare
    # the complete fixed RNG distribution, not independent nodule sampling.
    means = []
    for draw in np.unique(draws, axis=0):
        indices = patient_cluster_resample_indices(canonical["patient_id"], [patients[i] for i in draw])
        means.append((tuple(draw), float(canonical["dice"].iloc[indices].mean())))
    by_draw = dict(means)
    values = np.array([by_draw[tuple(draw)] for draw in draws])
    low, high = np.quantile(values, [.025, .975], method="linear")
    assert result["metrics"]["dice"]["lower"] == low
    assert result["metrics"]["dice"]["upper"] == high
    assert result["metrics"]["dice"]["mean_difference"] == pytest.approx(7/3)
    assert result["n_patients"] == 2


def test_missing_correlations_are_explicit_and_invalid_pairing_rejected():
    left, right = _metric_rows(), _metric_rows()
    left["spatial_correlation"] = [np.nan, np.nan, .4]
    result = paired_alignment_bootstrap(left, right)
    assert result["metrics"]["spatial_correlation"]["n_pairwise_defined"] == 1
    assert result["metrics"]["spatial_correlation"]["empty_defined_replicates"] > 0
    left["spatial_correlation"] = np.nan
    result = paired_alignment_bootstrap(left, right)
    assert result["metrics"]["spatial_correlation"]["lower"] is None
    assert result["metrics"]["spatial_correlation"]["empty_defined_replicates"] == 10000
    right.loc[0, "true_class"] = "malignant"
    with pytest.raises(ValueError, match="cohort mismatch"):
        paired_alignment_bootstrap(left, right)
    with pytest.raises(ValueError, match="unique test"):
        paired_alignment_bootstrap(pd.concat([left, left.iloc[:1]]), right)


def test_panel_selection_and_actual_array_overlay_are_deterministic(tmp_path):
    rows = pd.concat([_metric_rows().assign(seed=seed, dice=[0., .5, 1.]) for seed in (42, 43, 44)])
    cases = select_panel_cases(rows)
    assert [case["nodule_id"] for case in cases] == [FLAGGED_NODULE, "n2", "n3"]
    assert cases[0]["reasons"] == ["flagged_zero_alignment_case", "low"]
    assert cases == select_panel_cases(rows.sample(frac=1, random_state=4))
    # Synthetic inputs solely for unit testing the renderer; not research evidence.
    ct = np.arange(3 * 16, dtype=np.float32).reshape(3, 1, 4, 4) / 48
    maps = np.full((3, 2, 2), .25)
    path = write_overlay_panel(tmp_path / "synthetic_test_panel.png", cases, _identities(), ct, maps, maps, maps, maps[0])
    assert path.read_bytes().startswith(b"\x89PNG")
    second = write_overlay_panel(tmp_path / "same.png", cases, _identities(), ct, maps, maps, maps, maps[0])
    assert path.read_bytes() == second.read_bytes()


def test_scalar_replay_reports_without_adjusting_tolerance():
    rows = _metric_rows().to_dict("records")
    tolerance = supplementary_protocol()["previous_scalar_replay"]
    assert compare_scalar_records(rows, rows[::-1], tolerance)["consistent"]
    changed = [dict(row) for row in rows]
    changed[0]["dice"] += .1
    report = compare_scalar_records(changed, rows, tolerance)
    assert not report["consistent"]
    assert report["tolerance"] == tolerance
    assert rows[0]["dice"] == 1.
