"""Validated Phase 4 ConsenDxFormer configuration.

Phase 4 is anchored to the validated B1 protocol. The proposed model may add
only explicitly declared ConsenDxFormer components; data resolution, training
budget, augmentation policy, class order, and core Transformer/CNN capacity
remain locked to B1 for fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .baseline_config import load_baseline_config
from .config import load_config
from .hashing import sha256_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_COHORT_VIEW = "paper_aligned_multi_reader_2_to_4"

_LOCKED_DATASET_KEYS = (
    "image_size",
    "in_channels",
    "num_classes",
    "batch_size",
    "num_workers",
    "class_names",
)

_LOCKED_MODEL_KEYS = (
    "cnn_channels",
    "embed_dim",
    "transformer_heads",
    "transformer_layers",
    "transformer_mlp_ratio",
    "dropout",
    "classifier_hidden_dim",
    "use_positional_encoding",
    "use_spatial_attention",
)

_LOCKED_TRAINING_KEYS = (
    "epochs",
    "lr",
    "weight_decay",
    "label_smoothing",
    "class_weights",
    "early_stopping_patience",
    "scheduler",
    "scheduler_patience",
    "scheduler_factor",
    "amp",
    "grad_clip_norm",
)


@dataclass(frozen=True)
class Phase4ConfigBundle:
    config_path: Path
    b1_anchor_path: Path
    phase4_config: dict[str, Any]
    b1_anchor_config: dict[str, Any]
    combined_config_sha256: str


def _mapping(
    config: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Phase 4 configuration section {key!r} must be a mapping"
        )
    return value


def _nonempty_text(
    value: Any,
    role: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a non-empty string")
    return value.strip()


def _validate_locked_section(
    *,
    phase4: Mapping[str, Any],
    anchor: Mapping[str, Any],
    section_name: str,
    keys: tuple[str, ...],
) -> None:
    proposed = _mapping(phase4, section_name)
    baseline = _mapping(anchor, section_name)

    for key in keys:
        if key not in proposed:
            raise ValueError(
                f"Phase 4 {section_name}.{key} is required by the B1 fairness anchor"
            )
        if proposed[key] != baseline.get(key):
            raise ValueError(
                "Phase 4 fairness anchor mismatch for "
                f"{section_name}.{key}: "
                f"{proposed[key]!r} != {baseline.get(key)!r}"
            )


def _validate_candidate_grid(
    value: Any,
    role: str,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{role} must be a non-empty list")

    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{role} must contain only numeric values")
        number = float(item)
        if number < 0.0:
            raise ValueError(f"{role} values must be non-negative")
        result.append(number)

    if len(result) != len(set(result)):
        raise ValueError(f"{role} cannot contain duplicate values")

    if 0.0 not in result:
        raise ValueError(
            f"{role} must include 0.0 so the corresponding auxiliary term can be ablated"
        )

    return tuple(result)


def validate_phase4_config(
    config: Mapping[str, Any],
    *,
    b1_anchor_config: Mapping[str, Any],
) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("Phase 4 configuration must be a mapping")
    if not isinstance(b1_anchor_config, Mapping):
        raise TypeError("b1_anchor_config must be a mapping")

    project = _mapping(config, "project")
    if project.get("phase") != 4:
        raise ValueError("project.phase must be exactly 4")

    experiment = _mapping(config, "experiment")
    if _nonempty_text(
        experiment.get("model_name"),
        "experiment.model_name",
    ) != "ConsenDxFormer":
        raise ValueError(
            "experiment.model_name must be exactly 'ConsenDxFormer'"
        )

    seed = experiment.get("seed")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError(
            "experiment.seed must be an integer in the range 0..2**32-1"
        )

    if seed != b1_anchor_config.get("seed"):
        raise ValueError(
            "Phase 4 fairness anchor requires the initial configuration seed "
            "to match B1"
        )

    device = config.get("device")
    if device != b1_anchor_config.get("device"):
        raise ValueError(
            "Phase 4 fairness anchor requires the B1 device-selection policy"
        )

    fairness = _mapping(config, "fairness_anchor")
    if fairness.get("config") != "configs/b1.yaml":
        raise ValueError(
            "fairness_anchor.config must be exactly 'configs/b1.yaml'"
        )

    if fairness.get("baseline_variant") != "B1":
        raise ValueError(
            "fairness_anchor.baseline_variant must be exactly 'B1'"
        )

    _validate_locked_section(
        phase4=config,
        anchor=b1_anchor_config,
        section_name="dataset",
        keys=_LOCKED_DATASET_KEYS,
    )
    _validate_locked_section(
        phase4=config,
        anchor=b1_anchor_config,
        section_name="model",
        keys=_LOCKED_MODEL_KEYS,
    )
    _validate_locked_section(
        phase4=config,
        anchor=b1_anchor_config,
        section_name="training",
        keys=_LOCKED_TRAINING_KEYS,
    )

    proposed_augmentation = _mapping(config, "augmentation")
    anchor_augmentation = _mapping(
        b1_anchor_config,
        "augmentation",
    )
    if dict(proposed_augmentation) != dict(anchor_augmentation):
        raise ValueError(
            "Phase 4 fairness anchor requires the exact B1 augmentation policy"
        )

    data = _mapping(config, "data")

    if data.get("mode") != "real":
        raise ValueError(
            "Phase 4 real-data contract requires data.mode='real'"
        )

    if (
        data.get("source_readiness_report")
        != "artifacts/phase2/runtime/source_readiness.json"
    ):
        raise ValueError(
            "Phase 4 real-data lineage requires the locked Phase 2 "
            "source-readiness report"
        )

    if data.get("synthetic_marker") is not None:
        raise ValueError(
            "Phase 4 real-data contract forbids a synthetic marker"
        )

    for key in (
        "metadata_path",
        "rendered_dir",
        "canonical_dir",
        "frozen_split_dir",
        "source_readiness_report",
        "cohort_view",
    ):
        _nonempty_text(
            data.get(key),
            f"data.{key}",
        )

    if data["cohort_view"] != PRIMARY_COHORT_VIEW:
        raise ValueError(
            f"data.cohort_view must be exactly {PRIMARY_COHORT_VIEW!r}"
        )

    anchor_manifest = _mapping(
        b1_anchor_config,
        "frozen_manifest",
    )
    if str(data["frozen_split_dir"]) != str(
        anchor_manifest.get("directory")
    ):
        raise ValueError(
            "Phase 4 fairness anchor requires the same frozen Phase 0 split"
        )

    if int(_mapping(config, "dataset")["image_size"]) != 64:
        raise ValueError(
            "Phase 4 primary comparison is locked to the 64x64 B1 input"
        )

    if "rendered_64" not in str(data["rendered_dir"]):
        raise ValueError(
            "Phase 4 primary rendered_dir must reference the 64x64 artifacts"
        )

    expected_determinism = {
        "deterministic_algorithms": True,
        "warn_only": True,
        "cudnn_benchmark": False,
        "dataloader_worker_seeding": True,
        "seeded_dataloader_generator": True,
    }
    if dict(_mapping(config, "determinism")) != expected_determinism:
        raise ValueError(
            "Phase 4 determinism policy must match the validated Phase 3 contract"
        )

    expected_checkpoint = {
        "selection_criterion": "validation_loss",
        "selection_mode": "min",
        "save_best": True,
        "save_last": True,
    }
    if dict(_mapping(config, "checkpoint")) != expected_checkpoint:
        raise ValueError(
            "Phase 4 checkpoint policy must match the validated Phase 3 contract"
        )

    expected_predictions = {
        "format": "csv",
        "splits": ["val", "test"],
        "class_order": ["benign", "indeterminate", "malignant"],
        "export_logits": True,
        "export_probabilities": True,
    }
    if dict(_mapping(config, "predictions")) != expected_predictions:
        raise ValueError(
            "Phase 4 prediction-export policy must match the validated Phase 3 contract"
        )

    expected_inference = {
        "tta": False,
        "ema": False,
        "pretrained_weights": False,
    }
    if dict(_mapping(config, "inference")) != expected_inference:
        raise ValueError(
            "Phase 4 inference policy must preserve the validated no-TTA, "
            "no-EMA, no-pretraining contract"
        )

    losses = _mapping(config, "losses")

    if losses.get("selection_split") != "val":
        raise ValueError(
            "Phase 4 loss hyperparameters must be selected on validation data only"
        )

    if losses.get("test_tuning_forbidden") is not True:
        raise ValueError(
            "Phase 4 must explicitly forbid test-based hyperparameter tuning"
        )

    ordinal_candidates = _validate_candidate_grid(
        losses.get("lambda_ordinal_candidates"),
        "losses.lambda_ordinal_candidates",
    )
    attention_candidates = _validate_candidate_grid(
        losses.get("lambda_attention_candidates"),
        "losses.lambda_attention_candidates",
    )

    selected_ordinal = losses.get(
        "selected_lambda_ordinal"
    )
    selected_attention = losses.get(
        "selected_lambda_attention"
    )

    for selected, candidates, role in (
        (
            selected_ordinal,
            ordinal_candidates,
            "selected_lambda_ordinal",
        ),
        (
            selected_attention,
            attention_candidates,
            "selected_lambda_attention",
        ),
    ):
        if selected is not None:
            if (
                isinstance(selected, bool)
                or not isinstance(selected, (int, float))
            ):
                raise ValueError(
                    f"losses.{role} must be null or numeric"
                )
            if float(selected) not in candidates:
                raise ValueError(
                    f"losses.{role} must be chosen from its declared validation candidate grid"
                )

    if (selected_ordinal is None) != (
        selected_attention is None
    ):
        raise ValueError(
            "selected Phase 4 auxiliary loss weights must either both be null "
            "or both be explicitly selected"
        )

    ablation = _mapping(config, "ablation")
    expected_ablation = {
        "use_disagreement_weighting": True,
        "use_ordinal_head": True,
        "use_attention_alignment": True,
        "use_hierarchical_fusion": True,
    }

    if dict(ablation) != expected_ablation:
        raise ValueError(
            "Phase 4 ablation defaults must explicitly enable all proposed components"
        )

    model = _mapping(config, "model")
    if model.get("use_hierarchical_fusion") is not True:
        raise ValueError(
            "model.use_hierarchical_fusion must explicitly enable the locked Phase 4 design"
        )
    if model["use_hierarchical_fusion"] != ablation["use_hierarchical_fusion"]:
        raise ValueError(
            "model and ablation hierarchical-fusion controls must agree"
        )


def _resolve_anchor_path(
    config_path: Path,
    reference: str,
) -> Path:
    path = Path(reference)

    if path.is_absolute():
        return path.resolve()

    project_candidate = (
        PROJECT_ROOT / path
    ).resolve()

    if project_candidate.is_file():
        return project_candidate

    local_candidate = (
        config_path.parent / path
    ).resolve()

    return local_candidate


def load_phase4_config(
    path: str | Path,
) -> Phase4ConfigBundle:
    config_path = Path(path).resolve()
    phase4 = load_config(config_path)

    fairness = _mapping(
        phase4,
        "fairness_anchor",
    )
    anchor_reference = _nonempty_text(
        fairness.get("config"),
        "fairness_anchor.config",
    )

    anchor_path = _resolve_anchor_path(
        config_path,
        anchor_reference,
    )

    expected_anchor = (
        PROJECT_ROOT / "configs" / "b1.yaml"
    ).resolve()

    if anchor_path != expected_anchor:
        raise ValueError(
            "Phase 4 fairness anchor must resolve to the repository's locked configs/b1.yaml"
        )

    b1_anchor = load_baseline_config(
        anchor_path,
        expected_mode="B1",
    )

    validate_phase4_config(
        phase4,
        b1_anchor_config=b1_anchor,
    )

    combined_hash = sha256_json(
        {
            "schema_version": 1,
            "phase4": phase4,
            "b1_anchor": b1_anchor,
        }
    )

    return Phase4ConfigBundle(
        config_path=config_path,
        b1_anchor_path=anchor_path,
        phase4_config=dict(phase4),
        b1_anchor_config=dict(b1_anchor),
        combined_config_sha256=combined_hash,
    )
