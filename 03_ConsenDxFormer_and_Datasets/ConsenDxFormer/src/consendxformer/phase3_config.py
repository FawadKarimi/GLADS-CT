"""Validated Phase 3 baseline-experiment configuration.

Phase 3 deliberately keeps the LungDxFormer protocol in the locked Phase 1
configuration files.  This module adds only experiment orchestration policy and
returns the exact Phase 1 configuration after applying the Phase 3 seed/device.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .baseline_config import (
    BASELINE_MODES,
    CLASS_NAMES,
    apply_safe_overrides,
    load_baseline_config,
)
from .baseline_metadata_export import COHORT_VIEWS
from .config import load_config
from .hashing import sha256_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_MODES = ("real", "synthetic")
SYNTHETIC_MARKER = "SYNTHETIC_NON_RESEARCH_ONLY"
SYNTHETIC_COHORT_VIEW = "synthetic_tiny_fixture"
PREDICTION_SPLITS = ("val", "test")


@dataclass(frozen=True)
class Phase3ConfigBundle:
    """A Phase 3 config plus its validated, effective Phase 1 dependency."""

    config_path: Path
    phase1_config_path: Path
    phase3_config: dict[str, Any]
    effective_phase1_config: dict[str, Any]
    combined_config_sha256: str

    @property
    def config(self) -> dict[str, Any]:
        """Compatibility shorthand for orchestration callers."""

        return self.phase3_config

    @property
    def phase1_config(self) -> dict[str, Any]:
        return self.effective_phase1_config

    @property
    def config_sha256(self) -> str:
        return self.combined_config_sha256

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["config_path"] = str(self.config_path)
        record["phase1_config_path"] = str(self.phase1_config_path)
        return record


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase 3 configuration section {key!r} must be a mapping")
    return value


def _nonempty_text(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a non-empty string")
    return value.strip()


def _optional_path_text(value: Any, role: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, role)


def _require_boolean(section: Mapping[str, Any], key: str, expected: bool | None = None) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    if expected is not None and value is not expected:
        raise ValueError(f"{key} is locked to {expected!r} for Phase 3")
    return value


def validate_phase3_config(
    config: Mapping[str, Any],
    *,
    effective_phase1_config: Mapping[str, Any] | None = None,
    expected_variant: str | None = None,
) -> None:
    """Validate orchestration policy without requiring unavailable real data.

    Paths are intentionally checked for shape, not existence.  The separate
    readiness guard owns evidence validation and must fail closed before a real
    run creates research results.
    """

    if not isinstance(config, Mapping):
        raise TypeError("Phase 3 configuration must be a mapping")

    project = _mapping(config, "project")
    if project.get("phase") != 3:
        raise ValueError("project.phase must be exactly 3")

    experiment = _mapping(config, "experiment")
    variant = _nonempty_text(
        experiment.get("baseline_variant"), "experiment.baseline_variant"
    ).upper()
    if variant not in BASELINE_MODES:
        raise ValueError(
            f"experiment.baseline_variant must be one of {BASELINE_MODES}; got {variant!r}"
        )
    if expected_variant is not None and variant != str(expected_variant).upper():
        raise ValueError(
            f"Expected Phase 3 variant {str(expected_variant).upper()}, got {variant}"
        )
    seed = experiment.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("experiment.seed must be an integer in the range 0..2**32-1")
    _nonempty_text(experiment.get("output_root"), "experiment.output_root")

    baseline = _mapping(config, "baseline")
    _nonempty_text(baseline.get("phase1_config"), "baseline.phase1_config")

    device = config.get("device")
    if device not in {"auto", "cpu"}:
        raise ValueError(
            "device must be 'auto' or 'cpu', matching the locked released selection rule"
        )

    data = _mapping(config, "data")
    required_data_keys = {
        "mode",
        "metadata_path",
        "frozen_split_dir",
        "source_readiness_report",
        "canonical_dir",
        "rendered_dir",
        "cohort_view",
        "synthetic_marker",
    }
    missing_data_keys = sorted(required_data_keys - set(data))
    if missing_data_keys:
        raise ValueError(f"data section lacks required keys: {missing_data_keys}")
    mode = _nonempty_text(data.get("mode"), "data.mode")
    if mode not in DATA_MODES:
        raise ValueError(f"data.mode must be one of {DATA_MODES}; got {mode!r}")
    _nonempty_text(data.get("metadata_path"), "data.metadata_path")
    _nonempty_text(data.get("frozen_split_dir"), "data.frozen_split_dir")
    cohort_view = _nonempty_text(data.get("cohort_view"), "data.cohort_view")
    readiness = _optional_path_text(
        data.get("source_readiness_report"), "data.source_readiness_report"
    )
    canonical = _optional_path_text(data.get("canonical_dir"), "data.canonical_dir")
    rendered = _optional_path_text(data.get("rendered_dir"), "data.rendered_dir")
    marker = data.get("synthetic_marker")
    if mode == "real":
        if marker is not None:
            raise ValueError("Real-data mode requires data.synthetic_marker to be null")
        missing_paths = [
            role
            for role, value in (
                ("source_readiness_report", readiness),
                ("canonical_dir", canonical),
                ("rendered_dir", rendered),
            )
            if value is None
        ]
        if missing_paths:
            raise ValueError(
                "Real-data mode requires Phase 2 evidence paths: " + ", ".join(missing_paths)
            )
        if cohort_view not in COHORT_VIEWS:
            raise ValueError(f"Real data.cohort_view must be one of {COHORT_VIEWS}")
    else:
        if marker != SYNTHETIC_MARKER:
            raise ValueError(
                f"Synthetic mode requires data.synthetic_marker={SYNTHETIC_MARKER!r}"
            )
        if any(value is not None for value in (readiness, canonical, rendered)):
            raise ValueError(
                "Synthetic mode cannot reference real source-readiness, canonical, "
                "or rendered evidence"
            )
        if cohort_view != SYNTHETIC_COHORT_VIEW:
            raise ValueError(
                f"Synthetic mode requires data.cohort_view={SYNTHETIC_COHORT_VIEW!r}"
            )

    determinism = _mapping(config, "determinism")
    _require_boolean(determinism, "deterministic_algorithms", True)
    _require_boolean(determinism, "warn_only")
    _require_boolean(determinism, "cudnn_benchmark", False)
    _require_boolean(determinism, "dataloader_worker_seeding", True)
    _require_boolean(determinism, "seeded_dataloader_generator", True)

    checkpoint = _mapping(config, "checkpoint")
    if checkpoint.get("selection_criterion") != "validation_loss":
        raise ValueError(
            "checkpoint.selection_criterion is locked to 'validation_loss'"
        )
    if checkpoint.get("selection_mode") != "min":
        raise ValueError("checkpoint.selection_mode is locked to 'min'")
    _require_boolean(checkpoint, "save_best", True)
    _require_boolean(checkpoint, "save_last", True)

    predictions = _mapping(config, "predictions")
    if predictions.get("format") != "csv":
        raise ValueError("predictions.format is locked to 'csv'")
    splits = predictions.get("splits")
    if not isinstance(splits, (list, tuple)) or tuple(splits) != PREDICTION_SPLITS:
        raise ValueError(
            f"predictions.splits must be ordered as {list(PREDICTION_SPLITS)!r}"
        )
    class_order = predictions.get("class_order")
    if not isinstance(class_order, (list, tuple)) or tuple(class_order) != CLASS_NAMES:
        raise ValueError(f"predictions.class_order must be ordered as {list(CLASS_NAMES)!r}")
    _require_boolean(predictions, "export_logits", True)
    _require_boolean(predictions, "export_probabilities", True)

    inference = _mapping(config, "inference")
    _require_boolean(inference, "tta", False)
    _require_boolean(inference, "ema", False)
    _require_boolean(inference, "pretrained_weights", False)

    if effective_phase1_config is not None:
        phase1_variant = str(
            _mapping(effective_phase1_config, "experiment").get("baseline_mode", "")
        ).upper()
        if phase1_variant != variant:
            raise ValueError(
                f"Phase 3 {variant} references a Phase 1 {phase1_variant or '<missing>'} config"
            )
        if effective_phase1_config.get("seed") != seed:
            raise ValueError("Effective Phase 1 seed disagrees with experiment.seed")
        if effective_phase1_config.get("device") != device:
            raise ValueError("Effective Phase 1 device disagrees with Phase 3 device")
        phase1_manifest = _mapping(effective_phase1_config, "frozen_manifest")
        if str(phase1_manifest.get("directory")) != str(data.get("frozen_split_dir")):
            raise ValueError(
                "data.frozen_split_dir must reference the same immutable split as Phase 1"
            )
        phase1_classes = tuple(
            str(value)
            for value in _mapping(effective_phase1_config, "dataset").get(
                "class_names", ()
            )
        )
        if phase1_classes != CLASS_NAMES:
            raise ValueError("Effective Phase 1 class order disagrees with Phase 3")


def _resolve_phase1_config_path(config_path: Path, value: str) -> Path:
    reference = Path(value)
    if reference.is_absolute():
        return reference.resolve()
    project_candidate = (PROJECT_ROOT / reference).resolve()
    local_candidate = (config_path.parent / reference).resolve()
    if project_candidate.is_file() or not local_candidate.is_file():
        return project_candidate
    return local_candidate


def load_phase3_config(
    path: str | Path,
    *,
    expected_variant: str | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> Phase3ConfigBundle:
    """Load Phase 3 and its locked Phase 1 configuration deterministically.

    ``seed`` and ``device`` are the only runtime overrides.  A baseline variant
    is selected by choosing the corresponding explicit Phase 3 config file.
    """

    config_path = Path(path).resolve()
    phase3 = deepcopy(load_config(config_path))
    if seed is not None:
        phase3.setdefault("experiment", {})["seed"] = seed
    if device is not None:
        phase3["device"] = device
    validate_phase3_config(phase3, expected_variant=expected_variant)

    variant = str(phase3["experiment"]["baseline_variant"]).upper()
    phase1_path = _resolve_phase1_config_path(
        config_path, str(phase3["baseline"]["phase1_config"])
    )
    expected_phase1_path = (PROJECT_ROOT / "configs" / f"{variant.lower()}.yaml").resolve()
    if phase1_path != expected_phase1_path:
        raise ValueError(
            "baseline.phase1_config must reference the repository's locked Phase 1 "
            f"{variant} config: {expected_phase1_path}"
        )
    phase1 = load_baseline_config(phase1_path, expected_mode=variant)
    phase1 = apply_safe_overrides(
        phase1,
        [
            f"seed={phase3['experiment']['seed']}",
            f"device={phase3['device']}",
        ],
    )
    validate_phase3_config(
        phase3,
        effective_phase1_config=phase1,
        expected_variant=expected_variant,
    )
    combined_hash = sha256_json(
        {
            "schema_version": 1,
            "phase3": phase3,
            "effective_phase1": phase1,
        }
    )
    return Phase3ConfigBundle(
        config_path=config_path,
        phase1_config_path=phase1_path,
        phase3_config=phase3,
        effective_phase1_config=phase1,
        combined_config_sha256=combined_hash,
    )
