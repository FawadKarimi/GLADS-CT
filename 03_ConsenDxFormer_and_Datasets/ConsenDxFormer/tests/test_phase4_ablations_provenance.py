from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.hashing import sha256_json
from consendxformer.phase4_ablations import (
    FROZEN_MANIFEST_BUNDLE_SHA256,
    FROZEN_ORDERED_SUPERVISION_SHA256,
    FROZEN_PHASE4_BASE_COMBINED_CONFIG_SHA256,
    FROZEN_PHASE4_INPUT_LOCK_SHA256,
    FROZEN_SOURCE_FINAL_MODEL_COMMIT,
    FROZEN_SUPERVISION_SHA256,
    Phase4AblationError,
    phase4_ablation_protocol_record,
    phase4_ablation_specifications,
    validate_phase4_ablation_protocol,
    validate_phase4_ablation_specification,
)
from consendxformer.phase4_config import load_phase4_config
from consendxformer.phase4_input_lock import load_phase4_input_lock


def test_ablation_protocol_is_bound_to_exact_frozen_source_and_cohort():
    record = phase4_ablation_protocol_record()

    assert record["source_binding"] == {
        "source_final_model_commit": (
            "e1b174490245c616e355ef7c0c6f7c3c239d3acd"
        ),
        "phase4_base_combined_config_sha256": (
            "246ea685c4fa8a9c2a73c937ca34b87ab9a7a8eddfce13fc60abc744dcaf90ae"
        ),
        "phase4_input_lock_sha256": (
            "5f9804163c17214c82af9fda3a5ce8ed21f0a63a55528af75ff8a004b0627ec5"
        ),
        "phase0_manifest_bundle_sha256": (
            "0c983f0ebbc6dd70623b570c8a657aa14ccc550ac590b577648b3163bc8a01e3"
        ),
        "supervision_sha256": (
            "cbe4e402825665428ee890887963437d2bc480f8616bd7ec7999c667759ccc25"
        ),
        "ordered_supervision_sha256": (
            "2bae364b180f75cf2332714cc486a517540fcff3b9e94c48d83f59d0d9d6789a"
        ),
    }
    assert record["cohort"] == {
        "cohort_view": "paper_aligned_multi_reader_2_to_4",
        "expected_resolution": 64,
        "sample_count": 1133,
        "split_counts": {"train": 807, "val": 143, "test": 183},
        "reader_count_counts": {"2": 307, "3": 305, "4": 521},
    }
    assert record["seeds"] == [42, 43, 44]
    assert record["selected_lambdas"] == {
        "lambda_ordinal": 0.5,
        "lambda_attention": 0.05,
    }


def test_ablation_provenance_digests_match_canonical_locked_inputs():
    lock = load_phase4_input_lock(ROOT / "configs" / "phase4_input_lock.json")
    config = load_phase4_config(ROOT / "configs" / "phase4_consensdxformer.yaml")

    assert sha256_json(lock.to_dict()) == FROZEN_PHASE4_INPUT_LOCK_SHA256
    assert config.combined_config_sha256 == FROZEN_PHASE4_BASE_COMBINED_CONFIG_SHA256
    assert lock.manifest_bundle_sha256 == FROZEN_MANIFEST_BUNDLE_SHA256
    assert lock.supervision_sha256 == FROZEN_SUPERVISION_SHA256
    assert lock.ordered_supervision_sha256 == FROZEN_ORDERED_SUPERVISION_SHA256
    assert FROZEN_SOURCE_FINAL_MODEL_COMMIT == (
        "e1b174490245c616e355ef7c0c6f7c3c239d3acd"
    )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("source_binding", "source_final_model_commit", "0" * 40),
        ("source_binding", "phase4_base_combined_config_sha256", "0" * 64),
        ("source_binding", "phase4_input_lock_sha256", "0" * 64),
        ("source_binding", "phase0_manifest_bundle_sha256", "0" * 64),
        ("source_binding", "supervision_sha256", "0" * 64),
        ("source_binding", "ordered_supervision_sha256", "0" * 64),
        ("cohort", "cohort_view", "different"),
        ("cohort", "expected_resolution", 128),
        ("cohort", "sample_count", 1132),
        ("cohort", "split_counts", {"train": 806, "val": 143, "test": 183}),
        ("cohort", "reader_count_counts", {"2": 306, "3": 305, "4": 521}),
    ],
)
def test_ablation_protocol_fails_closed_on_source_or_cohort_drift(
    section, field, replacement
):
    changed = copy.deepcopy(phase4_ablation_protocol_record())
    changed[section][field] = replacement

    with pytest.raises(Phase4AblationError, match="prespecified"):
        validate_phase4_ablation_protocol(changed)


@pytest.mark.parametrize(
    ("path", "integer_value"),
    [
        (("frozen_primary_cohort_required",), 1),
        (("frozen_split_required",), 1),
        (("frozen_preprocessing_required",), 1),
        (("post_test_architecture_changes_forbidden",), 1),
        (("specifications", 0, "toggles", "use_disagreement_weighting"), 0),
        (("specifications", 1, "toggles", "use_disagreement_weighting"), 1),
        (("specifications", 0, "frozen_split_required"), 1),
    ],
)
def test_ablation_protocol_rejects_integer_boolean_substitutions(path, integer_value):
    changed = copy.deepcopy(phase4_ablation_protocol_record())
    cursor = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = integer_value

    with pytest.raises(Phase4AblationError, match="prespecified"):
        validate_phase4_ablation_protocol(changed)


@pytest.mark.parametrize(
    ("field", "integer_value"),
    [
        ("use_disagreement_weighting", 1),
        ("use_ordinal_head", 0),
        ("frozen_split_required", 1),
        ("test_driven_tuning_forbidden", 1),
        ("training_execution_authorized", 0),
    ],
)
def test_typed_ablation_specification_rejects_integer_boolean_fields(
    field, integer_value
):
    specification = phase4_ablation_specifications()[1]
    changed = replace(specification, **{field: integer_value})

    with pytest.raises(Phase4AblationError, match="exact boolean"):
        validate_phase4_ablation_specification(changed)
