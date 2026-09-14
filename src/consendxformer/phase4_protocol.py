"""Frozen single-run controls and future Phase 4 tuning protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .hashing import sha256_json
from .reproducibility import validate_seed


ORDINAL_CANDIDATES: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
ATTENTION_CANDIDATES: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25)
TUNING_SEEDS: tuple[int, ...] = (42, 43, 44)
RUN_MODES: tuple[str, str] = ("tuning", "final")


class Phase4ProtocolError(ValueError):
    """Raised when a run would violate the validation-only tuning protocol."""


@dataclass(frozen=True)
class Phase4RunSpec:
    mode: str
    seed: int
    lambda_ordinal: float
    lambda_attention: float
    evaluate_test: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tuning_protocol_record() -> dict[str, Any]:
    """Return the complete locked selection protocol without executing a matrix."""

    return {
        "schema_version": 1,
        "ordinal_candidates": list(ORDINAL_CANDIDATES),
        "attention_candidates": list(ATTENTION_CANDIDATES),
        "stage1": {
            "seed": 42,
            "candidate_count": 16,
            "evaluate_split": "val",
            "test_evaluation_forbidden": True,
            "ranking_metric": "validation_macro_f1",
            "advance_top_k": 4,
        },
        "stage2": {
            "seeds": [43, 44],
            "candidate_count": 4,
            "evaluate_split": "val",
            "test_evaluation_forbidden": True,
        },
        "selection_seed_set": [42, 43, 44],
        "selection_aggregation": {
            "scope": "candidate_across_seeds_42_43_44",
            "primary": "mean_validation_macro_f1",
            "direction": "max",
            "test_metrics_permitted": False,
        },
        "final_tie_break_order": [
            "mean_validation_macro_f1",
            "mean_validation_indeterminate_f1",
            "mean_validation_indeterminate_recall",
            "smaller_lambda_sum",
            "smaller_lambda_ordinal",
            "smaller_lambda_attention",
        ],
        "test_set_used_for_selection": False,
        "selected_pair_destination": (
            "losses.selected_lambda_ordinal_and_selected_lambda_attention"
        ),
        "final_runs_require_frozen_selected_pair": True,
        "matrix_execution_in_this_milestone": False,
    }


def tuning_protocol_sha256() -> str:
    return sha256_json(tuning_protocol_record())


def _numeric_candidate(value: Any, candidates: tuple[float, ...], role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4ProtocolError(f"{role} must be one of the locked numeric candidates")
    number = float(value)
    if number not in candidates:
        raise Phase4ProtocolError(
            f"{role} must be one of the locked candidates {list(candidates)!r}; got {value!r}"
        )
    return number


def _validate_declared_grid(config: Mapping[str, Any]) -> Mapping[str, Any]:
    losses = config.get("losses")
    if not isinstance(losses, Mapping):
        raise Phase4ProtocolError("Phase 4 configuration lacks losses")
    if tuple(float(value) for value in losses.get("lambda_ordinal_candidates", ())) != (
        ORDINAL_CANDIDATES
    ):
        raise Phase4ProtocolError("Configured ordinal candidate grid differs from protocol")
    if tuple(float(value) for value in losses.get("lambda_attention_candidates", ())) != (
        ATTENTION_CANDIDATES
    ):
        raise Phase4ProtocolError("Configured attention candidate grid differs from protocol")
    return losses


def resolve_phase4_run_spec(
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int | None = None,
    lambda_ordinal: float | None = None,
    lambda_attention: float | None = None,
) -> Phase4RunSpec:
    """Resolve one run while making tuning/final authority mutually exclusive."""

    if mode not in RUN_MODES:
        raise Phase4ProtocolError(f"mode must be one of {list(RUN_MODES)!r}")
    losses = _validate_declared_grid(config)
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise Phase4ProtocolError("Phase 4 configuration lacks experiment settings")
    resolved_seed = validate_seed(experiment.get("seed") if seed is None else seed)
    if resolved_seed not in TUNING_SEEDS:
        raise Phase4ProtocolError(
            f"Phase 4 run seed must be one of the locked seeds {list(TUNING_SEEDS)!r}"
        )

    if mode == "tuning":
        if lambda_ordinal is None or lambda_attention is None:
            raise Phase4ProtocolError(
                "Tuning mode requires both lambda values from the locked candidate grid"
            )
        ordinal = _numeric_candidate(
            lambda_ordinal, ORDINAL_CANDIDATES, "lambda_ordinal"
        )
        attention = _numeric_candidate(
            lambda_attention, ATTENTION_CANDIDATES, "lambda_attention"
        )
        return Phase4RunSpec(
            mode=mode,
            seed=resolved_seed,
            lambda_ordinal=ordinal,
            lambda_attention=attention,
            evaluate_test=False,
        )

    if lambda_ordinal is not None or lambda_attention is not None:
        raise Phase4ProtocolError(
            "Final mode forbids lambda overrides; values must come from the frozen config"
        )
    selected_ordinal = losses.get("selected_lambda_ordinal")
    selected_attention = losses.get("selected_lambda_attention")
    if selected_ordinal is None or selected_attention is None:
        raise Phase4ProtocolError(
            "Final mode is locked while selected Phase 4 lambdas are null"
        )
    return Phase4RunSpec(
        mode=mode,
        seed=resolved_seed,
        lambda_ordinal=_numeric_candidate(
            selected_ordinal, ORDINAL_CANDIDATES, "selected_lambda_ordinal"
        ),
        lambda_attention=_numeric_candidate(
            selected_attention, ATTENTION_CANDIDATES, "selected_lambda_attention"
        ),
        evaluate_test=True,
    )


__all__ = [
    "ATTENTION_CANDIDATES",
    "ORDINAL_CANDIDATES",
    "RUN_MODES",
    "TUNING_SEEDS",
    "Phase4ProtocolError",
    "Phase4RunSpec",
    "resolve_phase4_run_spec",
    "tuning_protocol_record",
    "tuning_protocol_sha256",
]
