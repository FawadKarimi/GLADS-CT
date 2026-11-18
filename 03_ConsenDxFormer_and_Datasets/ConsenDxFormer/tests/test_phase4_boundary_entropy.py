"""Synthetic float32-softmax rounding regressions; no actual inference."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_phase4_evidence_gaps import boundary_results, install_sources, supervision
from consendxformer import phase4_evidence_gaps as gaps
from consendxformer.phase4_boundary_entropy import boundary_entropy_policy, prepare_boundary_entropy_inputs
from consendxformer.phase4_evaluation_metrics import predictive_entropy
from consendxformer.hashing import sha256_file


def rounded_probabilities():
    return np.array([[.8, .1, .1], [.9, .05, .05], [.3, .4, .3]], dtype=np.float32).astype(np.float64)


def rounded_results():
    results = boundary_results()
    for result in results:
        probabilities = np.roll(rounded_probabilities(), result["seed"] - 42, axis=1)
        for row, values in zip(result["rows"], probabilities):
            for name, value in zip(gaps.CLASS_ORDER, values):
                row[f"prob_{name}"] = float(value)
            row["predicted_class"] = gaps.CLASS_ORDER[int(values.argmax())]
    return results


def test_adapter_normalizes_only_a_copy_and_shared_guard_remains_strict():
    raw = rounded_probabilities()
    original = raw.copy()
    deviation = np.abs(raw.sum(axis=1) - 1)
    assert ((deviation > 1e-8) & (deviation < 1e-6)).all()
    raw.setflags(write=False)
    with pytest.raises(ValueError, match="1e-8"):
        predictive_entropy(raw)
    prepared, diagnostics = prepare_boundary_entropy_inputs(raw)
    assert prepared.dtype == np.float64 and not np.shares_memory(raw, prepared)
    np.testing.assert_array_equal(raw, original)
    expected = original / original.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(prepared, expected)
    np.testing.assert_allclose(predictive_entropy(prepared), -(expected * np.log(expected)).sum(axis=1))
    np.testing.assert_array_equal(diagnostics["per_row"]["entropy_raw_probability_sum_deviation"], raw.sum(axis=1) - 1)
    assert diagnostics["summary"]["rows_with_nonunit_raw_mass"] == 3
    prepared[0, 0] = 0.0
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("raw", [
    [[.8, .1, .100002]], [[.8, .1, .099998]], [[0., 0., 0.]],
    [], np.empty((0, 3)), [.8, .1, .1], [[.5, .5]], [[.25] * 4],
    np.ones((1, 1, 3)), 1., [[1. + 1e-9, 0., 0.]],
    [[-1e-9, .5, .500000001]], [[float("nan"), .5, .5]],
    [[float("inf"), .5, .5]], [[float("-inf"), .5, .5]],
    [["not-numeric", .5, .5]], [[.8, .1], [.3, .3, .4]],
    [[.8 + .1j, .1, .1]],
])
def test_invalid_inputs_fail_before_division(raw):
    # A zero mass must be rejected, not divided into NaNs and then rejected by
    # the shared metric. All invalid inputs fail under strict FP error settings.
    with np.errstate(all="raise"), pytest.raises(ValueError):
        prepare_boundary_entropy_inputs(raw)


def test_zero_probability_is_preserved_without_clipping():
    prepared, diagnostics = prepare_boundary_entropy_inputs([[1., 0., 0.]])
    np.testing.assert_array_equal(prepared, [[1., 0., 0.]])
    np.testing.assert_array_equal(predictive_entropy(prepared), [0.])
    assert diagnostics["summary"]["max_abs_probability_adjustment"] == 0


def test_three_seed_summary_preserves_raw_rows_classes_confidence_and_means():
    results = rounded_results()
    before = copy.deepcopy(results)
    summary, tables = gaps.boundary_summary(results, supervision())
    assert results == before
    assert summary["boundary_entropy_preparation_policy"] == boundary_entropy_policy()
    raw = tables["boundary_predictions_per_seed.csv"]
    assert raw.seed.tolist() == [42] * 3 + [43] * 3 + [44] * 3
    for result in results:
        original = pd.DataFrame(result["rows"])
        actual = raw.loc[raw.seed == result["seed"]].reset_index(drop=True)
        # Every existing input column is retained, in original row order.
        pd.testing.assert_frame_equal(actual[original.columns], original)
        values = gaps._probabilities(original)
        np.testing.assert_array_equal(actual.confidence, values.max(axis=1))
        normalized = values / values.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(actual.predictive_entropy_nats, predictive_entropy(normalized), rtol=0, atol=0)
        for region in summary["per_seed_regions"]:
            if region["seed"] == result["seed"]:
                group = original.loc[original.boundary_stratum == region["boundary_stratum"]]
                for name in gaps.CLASS_ORDER:
                    assert region[f"mean_prob_{name}"] == group[f"prob_{name}"].mean()
    assert [record["row_count"] for record in summary["boundary_entropy_diagnostics_per_seed"]] == [3, 3, 3]
    pooled = tables["boundary_per_nodule_seed_mean.csv"]
    for row in pooled.to_dict("records"):
        original_rows = raw.loc[raw.nodule_id == row["nodule_id"]]
        for name in gaps.CLASS_ORDER:
            assert row[f"prob_{name}"] == original_rows[f"prob_{name}"].mean()


def test_boundary_export_records_policy_diagnostics_and_unchanged_raw_evidence(tmp_path, monkeypatch):
    full, _ = install_sources(tmp_path, monkeypatch)
    results = rounded_results()
    original = copy.deepcopy(results)
    # Synthetic counts only; production source-validity accounting is untouched.
    accounting = {"eligible_subset_count": 3, "full_boundary_membership_count": 4,
                  "source_invalid_exclusion_count": 1, "exclusion_records": []}
    boundary = SimpleNamespace(source_snapshot={}, provenance={"synthetic": True,
                              "source_validity_accounting": copy.deepcopy(accounting)})
    monkeypatch.setattr(gaps, "load_repository_phase4_boundary_source", lambda **kw: boundary)
    monkeypatch.setattr(gaps, "assert_boundary_source_snapshot_unchanged", lambda snapshot: None)
    monkeypatch.setattr(gaps, "load_frozen_final_best_model", lambda path, device:
        SimpleNamespace(source=next(s for s in full.values() if s.source_dir == path)))
    calls = []
    def saved_fixture(frozen, cohort):
        seed = frozen.source.descriptor["seed"]
        calls.append(seed)
        return results[seed - 42]
    monkeypatch.setattr(gaps, "predict_validated_boundary_cohort", saved_fixture)
    output = gaps.run_evidence_gap(analysis="boundary", full_source_dirs=[s.source_dir for s in full.values()])
    assert calls == [42, 43, 44] and results == original
    assert boundary.provenance["source_validity_accounting"] == accounting
    evaluation = json.loads((output / "evaluation.json").read_text())
    provenance = json.loads((output / "provenance/provenance.json").read_text())
    numerical = json.loads((output / "provenance/boundary_entropy_preparation.json").read_text())
    assert numerical["policy"] == provenance["boundary_entropy_preparation_policy"] == boundary_entropy_policy()
    assert numerical["policy"] == evaluation["summary"]["boundary_entropy_preparation_policy"]
    assert numerical["per_seed"] == evaluation["summary"]["boundary_entropy_diagnostics_per_seed"]
    assert evaluation["summary"]["source_validity_accounting"] == accounting
    evidence_path = output / numerical["per_row_evidence"]
    assert numerical["per_row_evidence_sha256"] == sha256_file(evidence_path)
    raw = pd.read_csv(evidence_path, float_precision="round_trip")
    expected = pd.DataFrame([row for result in original for row in result["rows"]])
    for name in gaps.CLASS_ORDER:
        np.testing.assert_array_equal(raw[f"prob_{name}"], expected[f"prob_{name}"])
    assert raw.predicted_class.tolist() == expected.predicted_class.tolist()
    np.testing.assert_array_equal(raw.entropy_raw_probability_sum, gaps._probabilities(expected).sum(axis=1))
    assert (raw.entropy_raw_probability_sum_abs_deviation > 1e-8).all()
    assert (raw.entropy_raw_probability_sum_abs_deviation < 1e-6).all()
    assert (raw.entropy_max_abs_probability_adjustment > 0).all()
    assert "Entropy alone uses a private float64 copy" in (output / "README.md").read_text()
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert manifest["files"]["provenance/boundary_entropy_preparation.json"] == sha256_file(output / "provenance/boundary_entropy_preparation.json")
    assert (output / "COMPLETED.json").exists()
