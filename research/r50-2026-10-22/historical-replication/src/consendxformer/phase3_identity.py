"""Deterministic, content-addressed identities for Phase 3 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .baseline_config import BASELINE_MODES
from .baseline_source import PINNED_BASELINE_SHA
from .hashing import sha256_json


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


@dataclass(frozen=True)
class ExperimentIdentity:
    """Auditable identity record persisted with every experiment."""

    experiment_id: str
    digest: str
    baseline_variant: str
    seed: int
    baseline_source_commit: str
    project_commit: str
    combined_config_sha256: str
    metadata_sha256: str
    cohort_sha256: str
    ordered_sample_sha256: str
    frozen_split_bundle_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_seed(seed: Any) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be in the range 0..2**32-1")
    return seed


def _validate_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{role} must be an exact lowercase SHA-256 digest")
    return value


def _validate_git_commit(value: Any, role: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{role} must be an exact lowercase 40- or 64-hex Git commit")
    return value


def experiment_identity_payload(
    *,
    baseline_variant: str,
    seed: int,
    baseline_source_commit: str,
    project_commit: str,
    combined_config_sha256: str,
    metadata_sha256: str,
    cohort_sha256: str,
    ordered_sample_sha256: str,
    frozen_split_bundle_sha256: str,
) -> dict[str, Any]:
    """Return the validated canonical preimage used for the identity digest."""

    if not isinstance(baseline_variant, str):
        raise TypeError("baseline_variant must be a string")
    variant = baseline_variant.upper()
    if variant not in BASELINE_MODES:
        raise ValueError(f"baseline_variant must be one of {BASELINE_MODES}")
    seed = _validate_seed(seed)
    baseline_commit = _validate_git_commit(
        baseline_source_commit, "baseline_source_commit"
    )
    if baseline_commit != PINNED_BASELINE_SHA:
        raise ValueError(
            "baseline_source_commit does not match the immutable Phase 1 source pin"
        )
    return {
        "schema_version": 1,
        "baseline_variant": variant,
        "seed": seed,
        "baseline_source_commit": baseline_commit,
        "project_commit": _validate_git_commit(project_commit, "project_commit"),
        "combined_config_sha256": _validate_sha256(
            combined_config_sha256, "combined_config_sha256"
        ),
        "metadata_sha256": _validate_sha256(metadata_sha256, "metadata_sha256"),
        "cohort_sha256": _validate_sha256(cohort_sha256, "cohort_sha256"),
        "ordered_sample_sha256": _validate_sha256(
            ordered_sample_sha256, "ordered_sample_sha256"
        ),
        "frozen_split_bundle_sha256": _validate_sha256(
            frozen_split_bundle_sha256, "frozen_split_bundle_sha256"
        ),
    }


def build_experiment_identity(
    *,
    baseline_variant: str,
    seed: int,
    baseline_source_commit: str,
    project_commit: str,
    combined_config_sha256: str,
    metadata_sha256: str,
    cohort_sha256: str,
    ordered_sample_sha256: str,
    frozen_split_bundle_sha256: str,
) -> ExperimentIdentity:
    """Derive a stable human-readable ID backed by a full SHA-256 digest."""

    payload = experiment_identity_payload(
        baseline_variant=baseline_variant,
        seed=seed,
        baseline_source_commit=baseline_source_commit,
        project_commit=project_commit,
        combined_config_sha256=combined_config_sha256,
        metadata_sha256=metadata_sha256,
        cohort_sha256=cohort_sha256,
        ordered_sample_sha256=ordered_sample_sha256,
        frozen_split_bundle_sha256=frozen_split_bundle_sha256,
    )
    digest = sha256_json(payload)
    variant = str(payload["baseline_variant"])
    normalized_seed = int(payload["seed"])
    experiment_id = f"phase3-{variant.lower()}-s{normalized_seed}-{digest[:16]}"
    return ExperimentIdentity(
        experiment_id=experiment_id,
        digest=digest,
        baseline_variant=variant,
        seed=normalized_seed,
        baseline_source_commit=str(payload["baseline_source_commit"]),
        project_commit=str(payload["project_commit"]),
        combined_config_sha256=str(payload["combined_config_sha256"]),
        metadata_sha256=str(payload["metadata_sha256"]),
        cohort_sha256=str(payload["cohort_sha256"]),
        ordered_sample_sha256=str(payload["ordered_sample_sha256"]),
        frozen_split_bundle_sha256=str(payload["frozen_split_bundle_sha256"]),
    )


# The verb "derive" reads naturally in runners; keep one canonical implementation.
derive_experiment_identity = build_experiment_identity
