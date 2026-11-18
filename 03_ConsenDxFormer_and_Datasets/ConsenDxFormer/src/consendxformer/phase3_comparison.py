"""Fair-pair guards for future Phase 3 B0/B1 comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .baseline_evaluation import CLASS_ORDER


_VARIANTS = ("B0", "B1")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_CHECKSUM_ALIASES = {
    "semantic_cohort_checksum": (
        "semantic_cohort_checksum",
        "cohort_checksum",
        "cohort_sha256",
    ),
    "frozen_split_checksum": (
        "frozen_split_checksum",
        "frozen_split_sha256",
    ),
    "ordered_sample_checksum": (
        "ordered_sample_checksum",
        "ordered_samples_sha256",
    ),
}


class FairComparisonError(ValueError):
    """Raised when two experiments are not eligible for paired comparison."""


def _required_field(record: Mapping[str, Any], field: str) -> Any:
    if field not in record:
        raise FairComparisonError(f"comparison descriptor is missing {field!r}")
    return record[field]


def _checksum(record: Mapping[str, Any], canonical_field: str) -> str:
    aliases = _CHECKSUM_ALIASES[canonical_field]
    supplied = [(alias, record[alias]) for alias in aliases if alias in record]
    if not supplied:
        raise FairComparisonError(
            f"comparison descriptor is missing {canonical_field!r} "
            f"(accepted names: {list(aliases)!r})"
        )
    normalized: list[tuple[str, str]] = []
    for alias, value in supplied:
        if not isinstance(value, str) or not _SHA256.fullmatch(value.strip()):
            raise FairComparisonError(
                f"{alias} must contain exactly 64 hexadecimal characters"
            )
        normalized.append((alias, value.strip().lower()))
    distinct = {value for _, value in normalized}
    if len(distinct) != 1:
        raise FairComparisonError(
            f"conflicting aliases were supplied for {canonical_field!r}: {normalized!r}"
        )
    return normalized[0][1]


def _class_order(value: Any) -> tuple[str, str, str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FairComparisonError("class_order must be a three-item sequence")
    supplied = tuple(value)
    if supplied != CLASS_ORDER:
        raise FairComparisonError(
            f"class_order must exactly equal {list(CLASS_ORDER)!r}; "
            f"got {list(supplied)!r}"
        )
    return CLASS_ORDER


def build_comparison_descriptor(
    *,
    baseline_variant: str,
    seed: int,
    semantic_cohort_checksum: str,
    frozen_split_checksum: str,
    ordered_sample_checksum: str,
    class_order: Sequence[str] = CLASS_ORDER,
    tta_enabled: bool = False,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Build the minimal semantic identity used by the pair guard."""

    record: dict[str, Any] = {
        "baseline_variant": baseline_variant,
        "seed": seed,
        "semantic_cohort_checksum": semantic_cohort_checksum,
        "frozen_split_checksum": frozen_split_checksum,
        "ordered_sample_checksum": ordered_sample_checksum,
        "class_order": list(class_order),
        "tta_enabled": tta_enabled,
    }
    if experiment_id is not None:
        record["experiment_id"] = experiment_id
    return validate_comparison_descriptor(record)


def validate_comparison_descriptor(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one experiment's comparison identity."""

    if not isinstance(record, Mapping):
        raise TypeError("comparison descriptor must be a mapping")
    variant = _required_field(record, "baseline_variant")
    if variant not in _VARIANTS:
        raise FairComparisonError(
            f"baseline_variant must be one of {list(_VARIANTS)!r}; got {variant!r}"
        )
    seed = _required_field(record, "seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FairComparisonError(
            f"seed must be a non-negative integer; got {seed!r}"
        )
    tta_enabled = _required_field(record, "tta_enabled")
    if not isinstance(tta_enabled, bool):
        raise FairComparisonError("tta_enabled must be boolean")

    canonical = {
        "baseline_variant": variant,
        "seed": int(seed),
        "semantic_cohort_checksum": _checksum(
            record, "semantic_cohort_checksum"
        ),
        "frozen_split_checksum": _checksum(record, "frozen_split_checksum"),
        "ordered_sample_checksum": _checksum(record, "ordered_sample_checksum"),
        "class_order": list(_class_order(_required_field(record, "class_order"))),
        "tta_enabled": tta_enabled,
    }
    if "experiment_id" in record:
        experiment_id = record["experiment_id"]
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise FairComparisonError("experiment_id must be a non-blank string")
        canonical["experiment_id"] = experiment_id.strip()
    return canonical


def validate_fair_baseline_pair(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the semantic invariants required for paired B0/B1 analysis.

    Configuration hashes and hyperparameter mappings are intentionally absent
    from the equality checks: locked B0 and B1 hyperparameter differences are
    expected and do not invalidate an otherwise paired experiment.
    """

    left = validate_comparison_descriptor(first)
    right = validate_comparison_descriptor(second)
    variants = {left["baseline_variant"], right["baseline_variant"]}
    if variants != set(_VARIANTS):
        raise FairComparisonError(
            "a fair baseline pair must contain exactly one B0 and one B1 "
            f"experiment; got {[left['baseline_variant'], right['baseline_variant']]!r}"
        )

    for field, description in (
        ("semantic_cohort_checksum", "semantic cohort checksum"),
        ("frozen_split_checksum", "frozen split checksum"),
        ("seed", "seed"),
        ("ordered_sample_checksum", "ordered sample checksum"),
        ("class_order", "class order"),
    ):
        if left[field] != right[field]:
            raise FairComparisonError(
                f"B0/B1 {description} mismatch: {left[field]!r} != {right[field]!r}"
            )

    if left["tta_enabled"] != right["tta_enabled"]:
        raise FairComparisonError(
            "B0/B1 TTA policy mismatch: both experiments must have equivalent disabled TTA"
        )
    if left["tta_enabled"]:
        raise FairComparisonError(
            "TTA must be disabled for Phase 3 paired baseline comparison"
        )

    by_variant = {
        descriptor["baseline_variant"]: descriptor for descriptor in (left, right)
    }
    report: dict[str, Any] = {
        "eligible": True,
        "baseline_variants": list(_VARIANTS),
        "seed": left["seed"],
        "semantic_cohort_checksum": left["semantic_cohort_checksum"],
        "frozen_split_checksum": left["frozen_split_checksum"],
        "ordered_sample_checksum": left["ordered_sample_checksum"],
        "class_order": list(CLASS_ORDER),
        "tta_enabled": False,
    }
    experiment_ids = {
        variant: descriptor["experiment_id"]
        for variant, descriptor in by_variant.items()
        if "experiment_id" in descriptor
    }
    if experiment_ids:
        report["experiment_ids"] = experiment_ids
    return report


__all__ = [
    "FairComparisonError",
    "build_comparison_descriptor",
    "validate_comparison_descriptor",
    "validate_fair_baseline_pair",
]
