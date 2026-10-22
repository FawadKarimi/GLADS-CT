"""Frozen, non-executing specifications for the four Phase 4 ablations.

This module deliberately describes experiments; it does not launch training.
The records keep selection, cohort, loss-weight, and seed policy alongside the
toggle state so an ablation cannot quietly become a new tuning exercise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

FINAL_SEEDS: tuple[int, int, int] = (42, 43, 44)
FROZEN_LAMBDA_ORDINAL = 0.5
FROZEN_LAMBDA_ATTENTION = 0.05
FROZEN_SOURCE_FINAL_MODEL_COMMIT = (
    "e1b174490245c616e355ef7c0c6f7c3c239d3acd"
)
FROZEN_PHASE4_BASE_COMBINED_CONFIG_SHA256 = (
    "246ea685c4fa8a9c2a73c937ca34b87ab9a7a8eddfce13fc60abc744dcaf90ae"
)
FROZEN_PHASE4_INPUT_LOCK_SHA256 = (
    "5f9804163c17214c82af9fda3a5ce8ed21f0a63a55528af75ff8a004b0627ec5"
)
FROZEN_MANIFEST_BUNDLE_SHA256 = (
    "0c983f0ebbc6dd70623b570c8a657aa14ccc550ac590b577648b3163bc8a01e3"
)
FROZEN_SUPERVISION_SHA256 = (
    "cbe4e402825665428ee890887963437d2bc480f8616bd7ec7999c667759ccc25"
)
FROZEN_ORDERED_SUPERVISION_SHA256 = (
    "2bae364b180f75cf2332714cc486a517540fcff3b9e94c48d83f59d0d9d6789a"
)
FROZEN_COHORT_VIEW = "paper_aligned_multi_reader_2_to_4"
FROZEN_EXPECTED_RESOLUTION = 64
FROZEN_SAMPLE_COUNT = 1133
FROZEN_SPLIT_COUNTS = {"train": 807, "val": 143, "test": 183}
FROZEN_READER_COUNT_COUNTS = {"2": 307, "3": 305, "4": 521}
ABLATION_NAMES: tuple[str, str, str, str] = (
    "disagreement_weighting",
    "ordinal_auxiliary",
    "attention_alignment",
    "hierarchical_fusion",
)
ABLATION_TOGGLES: tuple[str, str, str, str] = (
    "use_disagreement_weighting",
    "use_ordinal_head",
    "use_attention_alignment",
    "use_hierarchical_fusion",
)

_OFF_SEMANTICS = {
    "disagreement_weighting": (
        "Replace the disagreement-aware diagnostic term with baseline-style "
        "weighted hard cross-entropy only; do not blend soft diagnostic targets "
        "or radiologist disagreement. Independently enabled auxiliary terms remain "
        "unchanged in this single-component ablation."
    ),
    "ordinal_auxiliary": (
        "Remove the ordinal auxiliary loss contribution; retain the frozen "
        "lambda_ordinal value as provenance and do not retune it."
    ),
    "attention_alignment": (
        "Remove the attention-alignment auxiliary loss contribution; this does "
        "not disable the model's DSA attention mechanism, and the frozen "
        "lambda_attention remains provenance rather than a retuned value."
    ),
    "hierarchical_fusion": (
        "Set use_hierarchical_fusion to false and use the model's already-defined "
        "non-hierarchical execution path; make no other architecture change."
    ),
}

_DISABLED_TOGGLE = dict(zip(ABLATION_NAMES, ABLATION_TOGGLES))


class Phase4AblationError(ValueError):
    """Raised when an ablation record departs from the frozen protocol."""


@dataclass(frozen=True)
class Phase4AblationSpecification:
    """One single-component OFF experiment under otherwise frozen controls."""

    name: str
    disabled_toggle: str
    use_disagreement_weighting: bool
    use_ordinal_head: bool
    use_attention_alignment: bool
    use_hierarchical_fusion: bool
    off_semantics: str
    lambda_ordinal: float = FROZEN_LAMBDA_ORDINAL
    lambda_attention: float = FROZEN_LAMBDA_ATTENTION
    seeds: tuple[int, int, int] = FINAL_SEEDS
    cohort_view: str = FROZEN_COHORT_VIEW
    frozen_split_required: bool = True
    frozen_preprocessing_required: bool = True
    checkpoint_selection_split: str = "val"
    checkpoint_selection_criterion: str = "validation_loss"
    test_driven_tuning_forbidden: bool = True
    post_test_architecture_changes_forbidden: bool = True
    training_execution_authorized: bool = False
    all_toggles_off_is_b1: bool = False
    external_fair_baseline: str = "B1"

    @property
    def toggles(self) -> dict[str, bool]:
        """Return a fresh execution mapping in the canonical toggle order."""

        return {
            "use_disagreement_weighting": self.use_disagreement_weighting,
            "use_ordinal_head": self.use_ordinal_head,
            "use_attention_alignment": self.use_attention_alignment,
            "use_hierarchical_fusion": self.use_hierarchical_fusion,
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ABLATION_TOGGLES:
            value.pop(field)
        value["toggles"] = self.toggles
        value["seeds"] = list(self.seeds)
        return value


def _make_specification(name: str) -> Phase4AblationSpecification:
    if name not in ABLATION_NAMES:
        raise Phase4AblationError(f"Unknown Phase 4 ablation: {name!r}")
    disabled = _DISABLED_TOGGLE[name]
    toggles = {field: field != disabled for field in ABLATION_TOGGLES}
    return Phase4AblationSpecification(
        name=name,
        disabled_toggle=disabled,
        off_semantics=_OFF_SEMANTICS[name],
        **toggles,
    )


def phase4_ablation_specifications() -> tuple[Phase4AblationSpecification, ...]:
    """Return exactly the four prespecified, single-toggle Phase 4 ablations."""

    return tuple(_make_specification(name) for name in ABLATION_NAMES)


def validate_phase4_ablation_specification(
    specification: Phase4AblationSpecification,
) -> Phase4AblationSpecification:
    """Validate one typed specification against its canonical frozen record."""

    if not isinstance(specification, Phase4AblationSpecification):
        raise TypeError("specification must be a Phase4AblationSpecification")
    boolean_fields = ABLATION_TOGGLES + (
        "frozen_split_required",
        "frozen_preprocessing_required",
        "test_driven_tuning_forbidden",
        "post_test_architecture_changes_forbidden",
        "training_execution_authorized",
        "all_toggles_off_is_b1",
    )
    for field in boolean_fields:
        if type(getattr(specification, field)) is not bool:
            raise Phase4AblationError(
                f"Ablation field {field!r} must be an exact boolean"
            )
    expected = _make_specification(specification.name)
    if specification != expected:
        raise Phase4AblationError(
            f"Ablation {specification.name!r} differs from its frozen specification"
        )
    disabled = [field for field, enabled in specification.toggles.items() if not enabled]
    if disabled != [specification.disabled_toggle]:
        raise Phase4AblationError(
            "Each Phase 4 ablation must disable exactly its one declared toggle"
        )
    return specification


def phase4_ablation_protocol_record() -> dict[str, Any]:
    """Return a JSON-serializable protocol record without executing training."""

    specs = phase4_ablation_specifications()
    return {
        "schema_version": 1,
        "milestone": "phase4_evaluation_and_ablation_infrastructure",
        "experiment_kind": "prespecified_single_component_ablation",
        "ablation_names": list(ABLATION_NAMES),
        "specifications": [spec.to_dict() for spec in specs],
        "source_binding": {
            "source_final_model_commit": FROZEN_SOURCE_FINAL_MODEL_COMMIT,
            "phase4_base_combined_config_sha256": (
                FROZEN_PHASE4_BASE_COMBINED_CONFIG_SHA256
            ),
            "phase4_input_lock_sha256": FROZEN_PHASE4_INPUT_LOCK_SHA256,
            "phase0_manifest_bundle_sha256": FROZEN_MANIFEST_BUNDLE_SHA256,
            "supervision_sha256": FROZEN_SUPERVISION_SHA256,
            "ordered_supervision_sha256": FROZEN_ORDERED_SUPERVISION_SHA256,
        },
        "cohort": {
            "cohort_view": FROZEN_COHORT_VIEW,
            "expected_resolution": FROZEN_EXPECTED_RESOLUTION,
            "sample_count": FROZEN_SAMPLE_COUNT,
            "split_counts": dict(FROZEN_SPLIT_COUNTS),
            "reader_count_counts": dict(FROZEN_READER_COUNT_COUNTS),
        },
        "cohort_view": FROZEN_COHORT_VIEW,
        "frozen_primary_cohort_required": True,
        "frozen_split_required": True,
        "frozen_preprocessing_required": True,
        "selected_lambdas": {
            "lambda_ordinal": FROZEN_LAMBDA_ORDINAL,
            "lambda_attention": FROZEN_LAMBDA_ATTENTION,
        },
        "seeds": list(FINAL_SEEDS),
        "checkpoint_selection": {
            "split": "val",
            "criterion": "validation_loss",
            "mode": "min",
        },
        "test_set_used_for_selection": False,
        "test_driven_tuning_forbidden": True,
        "post_test_architecture_changes_forbidden": True,
        "training_execution_authorized": False,
        "all_toggles_off_is_b1": False,
        "b1_equivalence_statement": (
            "All ConsenDxFormer toggles OFF is not claimed to be B1 because "
            "architectural identity has not been established."
        ),
        "external_fair_baseline": "B1",
    }


def validate_phase4_ablation_protocol(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed if a serialized protocol changes any frozen decision."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    value = dict(record)
    selected = value.get("selected_lambdas")
    if selected != {
        "lambda_ordinal": FROZEN_LAMBDA_ORDINAL,
        "lambda_attention": FROZEN_LAMBDA_ATTENTION,
    }:
        raise Phase4AblationError("Ablation selected lambdas differ from the frozen pair")
    if value.get("seeds") != list(FINAL_SEEDS):
        raise Phase4AblationError("Ablation seeds differ from the frozen final seeds")
    if value.get("test_set_used_for_selection") is not False or value.get(
        "test_driven_tuning_forbidden"
    ) is not True:
        raise Phase4AblationError("Ablation protocol permits test-driven tuning")
    if value.get("all_toggles_off_is_b1") is not False:
        raise Phase4AblationError("Ablation protocol makes an unsupported B1 equivalence claim")
    if value.get("training_execution_authorized") is not False:
        raise Phase4AblationError("This milestone must not authorize ablation training")
    expected = phase4_ablation_protocol_record()
    if not _values_are_exactly_equal(value, expected):
        changed = sorted(
            set(value).symmetric_difference(expected)
            | {key for key in set(value) & set(expected) if value[key] != expected[key]}
        )
        raise Phase4AblationError(
            f"Ablation protocol differs from the prespecified record: {changed}"
        )
    # Rebuild the value so callers cannot mistake their mutable input for authority.
    return phase4_ablation_protocol_record()


def _values_are_exactly_equal(actual: Any, expected: Any) -> bool:
    """Compare a protocol recursively without Python's bool/int equivalence."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _values_are_exactly_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_are_exactly_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return bool(actual == expected)


__all__ = [
    "ABLATION_NAMES",
    "ABLATION_TOGGLES",
    "FINAL_SEEDS",
    "FROZEN_LAMBDA_ATTENTION",
    "FROZEN_LAMBDA_ORDINAL",
    "FROZEN_MANIFEST_BUNDLE_SHA256",
    "FROZEN_ORDERED_SUPERVISION_SHA256",
    "FROZEN_PHASE4_BASE_COMBINED_CONFIG_SHA256",
    "FROZEN_PHASE4_INPUT_LOCK_SHA256",
    "FROZEN_SOURCE_FINAL_MODEL_COMMIT",
    "FROZEN_SUPERVISION_SHA256",
    "Phase4AblationError",
    "Phase4AblationSpecification",
    "phase4_ablation_protocol_record",
    "phase4_ablation_specifications",
    "validate_phase4_ablation_protocol",
    "validate_phase4_ablation_specification",
]
