from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_ablations import (
    ABLATION_NAMES,
    FINAL_SEEDS,
    FROZEN_LAMBDA_ATTENTION,
    FROZEN_LAMBDA_ORDINAL,
    Phase4AblationError,
    phase4_ablation_protocol_record,
    phase4_ablation_specifications,
    validate_phase4_ablation_protocol,
)
from consendxformer.phase4_boundary import (
    BOUNDARY_RANGES,
    Phase4BoundaryError,
    isolate_phase2_boundary_cohort,
)


def test_ablation_specs_disable_exactly_one_component_with_frozen_controls():
    specs = phase4_ablation_specifications()

    assert tuple(spec.name for spec in specs) == ABLATION_NAMES
    assert len(specs) == 4
    for spec in specs:
        disabled = [name for name, enabled in spec.toggles.items() if not enabled]
        assert disabled == [spec.disabled_toggle]
        assert spec.seeds == FINAL_SEEDS == (42, 43, 44)
        assert spec.lambda_ordinal == FROZEN_LAMBDA_ORDINAL == 0.5
        assert spec.lambda_attention == FROZEN_LAMBDA_ATTENTION == 0.05
        assert spec.checkpoint_selection_split == "val"
        assert spec.checkpoint_selection_criterion == "validation_loss"
        assert spec.test_driven_tuning_forbidden is True
        assert spec.training_execution_authorized is False
        assert spec.all_toggles_off_is_b1 is False

    by_name = {spec.name: spec for spec in specs}
    assert "weighted hard cross-entropy only" in by_name[
        "disagreement_weighting"
    ].off_semantics
    assert "ordinal auxiliary loss contribution" in by_name[
        "ordinal_auxiliary"
    ].off_semantics
    assert "attention-alignment auxiliary loss contribution" in by_name[
        "attention_alignment"
    ].off_semantics
    assert "non-hierarchical execution path" in by_name[
        "hierarchical_fusion"
    ].off_semantics


def test_ablation_protocol_denies_b1_equivalence_and_test_driven_tuning():
    record = phase4_ablation_protocol_record()
    validated = validate_phase4_ablation_protocol(record)

    assert validated == record
    assert record["all_toggles_off_is_b1"] is False
    assert record["external_fair_baseline"] == "B1"
    assert record["test_set_used_for_selection"] is False
    assert record["selected_lambdas"] == {
        "lambda_ordinal": 0.5,
        "lambda_attention": 0.05,
    }

    changed = copy.deepcopy(record)
    changed["selected_lambdas"]["lambda_attention"] = 0.1
    with pytest.raises(Phase4AblationError, match="frozen"):
        validate_phase4_ablation_protocol(changed)

    changed = copy.deepcopy(record)
    changed["test_set_used_for_selection"] = True
    with pytest.raises(Phase4AblationError, match="test"):
        validate_phase4_ablation_protocol(changed)


def _phase2_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "p-primary",
                "nodule_id": "n-primary",
                "nodule_sha256": "a" * 64,
                "study_instance_uid": "study-primary",
                "series_instance_uid": "series-primary",
                "split": "test",
                "mean_malignancy": 3.0,
                "malignancy_score_sum": 6,
                "n_valid_malignancy_ratings": 2,
                "paper_label": "indeterminate",
                "paper_primary_eligible": True,
                "boundary_zone": False,
                "source_validity_status": "valid",
            },
            {
                "patient_id": "p-low",
                "nodule_id": "n-low",
                "nodule_sha256": "b" * 64,
                "study_instance_uid": "study-low",
                "series_instance_uid": "series-low",
                "split": "test",
                "mean_malignancy": 2.75,
                "malignancy_score_sum": 11,
                "n_valid_malignancy_ratings": 4,
                "paper_label": "boundary_zone",
                "paper_primary_eligible": False,
                "boundary_zone": True,
                "source_validity_status": "valid",
            },
            {
                "patient_id": "p-high",
                "nodule_id": "n-high",
                "nodule_sha256": "c" * 64,
                "study_instance_uid": "study-high",
                "series_instance_uid": "series-high",
                "split": "val",
                "mean_malignancy": 3.25,
                "malignancy_score_sum": 13,
                "n_valid_malignancy_ratings": 4,
                "paper_label": "boundary_zone",
                "paper_primary_eligible": "false",
                "boundary_zone": "true",
                "source_validity_status": "valid_with_missing_ratings",
            },
        ]
    )


def test_boundary_cohort_isolated_with_identifiers_counts_and_provenance():
    source = _phase2_rows()
    original = source.copy(deep=True)

    boundary, record = isolate_phase2_boundary_cohort(
        source,
        primary_nodule_ids={"n-primary"},
        source_provenance={
            "phase": 2,
            "canonical_ledger_sha256": "d" * 64,
        },
    )

    pd.testing.assert_frame_equal(source, original)
    assert boundary["nodule_id"].tolist() == ["n-low", "n-high"]
    assert boundary["boundary_stratum"].tolist() == [
        "lower_boundary",
        "upper_boundary",
    ]
    assert record["boundary_ranges"] == [
        {
            "lower": 2.5,
            "upper": 3.0,
            "lower_inclusive": True,
            "upper_inclusive": False,
        },
        {
            "lower": 3.0,
            "upper": 3.5,
            "lower_inclusive": False,
            "upper_inclusive": True,
        },
    ]
    assert record["sample_count"] == 2
    assert record["split_counts"] == {"train": 0, "val": 1, "test": 1}
    assert record["stratum_counts"] == {
        "lower_boundary": 1,
        "upper_boundary": 1,
    }
    assert [row["nodule_id"] for row in record["ordered_identities"]] == [
        "n-low",
        "n-high",
    ]
    assert record["source_provenance"]["phase"] == 2
    assert record["primary_cohort_excluded"] is True
    assert record["boundary_driven_tuning_forbidden"] is True
    assert "not assigned a primary three-class target" in record[
        "target_interpretation"
    ]
    assert record["limitations"]
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    ("mean", "score_sum", "reader_count"),
    [(2.5, 5, 2), (3.5, 7, 2)],
)
def test_boundary_phase2_endpoints_are_preserved(mean, score_sum, reader_count):
    source = _phase2_rows().iloc[[1]].copy()
    source.loc[:, "mean_malignancy"] = mean
    source.loc[:, "malignancy_score_sum"] = score_sum
    source.loc[:, "n_valid_malignancy_ratings"] = reader_count

    boundary, _ = isolate_phase2_boundary_cohort(
        source,
        primary_nodule_ids=set(),
        source_provenance={"phase": 2},
    )

    assert boundary["mean_malignancy"].tolist() == [mean]


def test_boundary_authoritative_marker_outside_phase2_rule_is_rejected():
    source = _phase2_rows().iloc[[1]].copy()
    source.loc[:, "mean_malignancy"] = 3.0
    source.loc[:, "malignancy_score_sum"] = 6
    source.loc[:, "n_valid_malignancy_ratings"] = 2

    with pytest.raises(Phase4BoundaryError, match="Phase 2 boundary rule"):
        isolate_phase2_boundary_cohort(
            source,
            primary_nodule_ids=set(),
            source_provenance={"phase": 2},
        )


def test_boundary_range_redefinition_and_primary_overlap_are_rejected():
    with pytest.raises(Phase4BoundaryError, match="must not be redefined"):
        isolate_phase2_boundary_cohort(
            _phase2_rows(),
            primary_nodule_ids={"n-primary"},
            source_provenance={"phase": 2},
            boundary_ranges=((2.6, 2.9), (3.1, 3.4)),
        )

    with pytest.raises(Phase4BoundaryError, match="overlaps the primary cohort"):
        isolate_phase2_boundary_cohort(
            _phase2_rows(),
            primary_nodule_ids={"n-low"},
            source_provenance={"phase": 2},
            boundary_ranges=BOUNDARY_RANGES,
        )


def test_boundary_phase2_boolean_and_label_disagreement_is_rejected():
    source = _phase2_rows()
    source.loc[source["nodule_id"] == "n-low", "paper_label"] = "benign"

    with pytest.raises(Phase4BoundaryError, match="disagree"):
        isolate_phase2_boundary_cohort(
            source,
            primary_nodule_ids={"n-primary"},
            source_provenance={"phase": 2},
        )
