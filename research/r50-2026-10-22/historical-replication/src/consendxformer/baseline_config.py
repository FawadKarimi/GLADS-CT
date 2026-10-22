from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import load_config

CLASS_NAMES = ("benign", "indeterminate", "malignant")
BASELINE_MODES = ("B0", "B1")
PINNED_MANIFEST_BUNDLE_SHA256 = "0c983f0ebbc6dd70623b570c8a657aa14ccc550ac590b577648b3163bc8a01e3"

_ARCHITECTURE_PROFILE: dict[str, Any] = {
    "cnn_channels": [32, 64, 128],
    "embed_dim": 128,
    "transformer_heads": 4,
    "transformer_layers": 2,
    "transformer_mlp_ratio": 4.0,
    "attention_dropout": 0.1,
    "classifier_hidden_dim": 128,
    "use_transformer": True,
    "use_positional_encoding": True,
    "use_spatial_attention": True,
    "fusion_type": "concat",
    "use_raw_transformer_in_fusion": False,
}

_SHARED_RELEASED_PROFILE: dict[tuple[str, ...], Any] = {
    ("dataset", "split_mode"): "csv_or_random",
    ("dataset", "train_ratio"): 0.70,
    ("dataset", "val_ratio"): 0.15,
    ("dataset", "test_ratio"): 0.15,
    ("dataset", "in_channels"): 1,
    ("augmentation", "train", "translation_frac"): 0.08,
    ("augmentation", "train", "intensity_jitter"): 0.08,
    ("augmentation", "train", "gaussian_noise_std"): 0.01,
    ("augmentation", "eval", "enabled"): False,
    ("training", "weight_decay"): 0.00001,
    ("training", "label_smoothing"): 0.0,
    ("training", "class_weights"): "auto",
    ("training", "scheduler"): "reduce_on_plateau",
    ("training", "scheduler_patience"): 3,
    ("training", "scheduler_factor"): 0.5,
    ("training", "amp"): False,
    ("training", "grad_clip_norm"): 1.0,
    ("evaluation", "save_confusion_matrix"): True,
    ("evaluation", "save_roc_curves"): True,
    ("evaluation", "save_predictions_csv"): True,
}

_MODE_PROFILE: dict[str, dict[tuple[str, ...], Any]] = {
    "B0": {
        ("dataset", "image_size"): 96,
        ("dataset", "batch_size"): 16,
        ("model", "dropout"): 0.2,
        ("training", "epochs"): 6,
        ("training", "lr"): 0.0001,
        ("training", "early_stopping_patience"): 8,
        ("augmentation", "train", "hflip"): True,
        ("augmentation", "train", "vflip"): False,
        ("augmentation", "train", "rotation_deg"): 15,
    },
    "B1": {
        ("dataset", "image_size"): 64,
        ("dataset", "batch_size"): 32,
        ("model", "dropout"): 0.5,
        ("training", "epochs"): 100,
        ("training", "lr"): 0.0001,
        ("training", "early_stopping_patience"): 10,
        ("augmentation", "train", "hflip"): True,
        ("augmentation", "train", "vflip"): True,
        ("augmentation", "train", "rotation_deg"): 10,
    },
}


def _at(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Baseline config is missing {'.'.join(path)}")
        value = value[key]
    return value


def _require_equal(config: dict[str, Any], path: tuple[str, ...], expected: Any) -> None:
    actual = _at(config, path)
    if actual != expected:
        raise ValueError(
            f"Invalid {'.'.join(path)} for locked baseline profile: "
            f"expected {expected!r}, got {actual!r}"
        )


def validate_baseline_config(config: dict[str, Any], *, expected_mode: str | None = None) -> None:
    """Validate the immutable three-class B0/B1 protocol surface."""
    mode = str(_at(config, ("experiment", "baseline_mode"))).upper()
    if mode not in BASELINE_MODES:
        raise ValueError(f"experiment.baseline_mode must be one of {BASELINE_MODES}; got {mode!r}")
    if expected_mode is not None and mode != expected_mode.upper():
        raise ValueError(f"Expected {expected_mode.upper()} config, got {mode}")

    name = _at(config, ("experiment", "name"))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("experiment.name must be a non-empty string")
    _require_equal(config, ("project", "phase"), 1)
    _require_equal(config, ("baseline", "name"), "LungDxFormer")
    _require_equal(config, ("baseline", "source_lock_file"), "locks/baseline_sources.yaml")
    _require_equal(config, ("baseline", "repository_dir"), "external/LungDxFormer")
    _require_equal(config, ("baseline", "project_subdir"), "LungDxFormer")
    _require_equal(config, ("baseline", "released_config"), "LungDxFormer/configs/config.yaml")
    _require_equal(config, ("baseline", "training_entrypoint"), "LungDxFormer/src/train.py")
    _require_equal(config, ("frozen_manifest", "directory"), "artifacts/phase0/manifests")
    _require_equal(
        config,
        ("frozen_manifest", "bundle_sha256"),
        PINNED_MANIFEST_BUNDLE_SHA256,
    )
    _require_equal(config, ("frozen_manifest", "unique_patients"), 1010)
    _require_equal(config, ("frozen_manifest", "unique_ct_series"), 1018)
    for split, count in {"train": 707, "val": 152, "test": 151}.items():
        _require_equal(config, ("frozen_manifest", "split_counts", split), count)
    _require_equal(config, ("dataset", "num_classes"), 3)
    class_names = tuple(str(x).strip().lower() for x in _at(config, ("dataset", "class_names")))
    if class_names != CLASS_NAMES:
        raise ValueError(f"dataset.class_names must be ordered as {list(CLASS_NAMES)!r}")

    if (
        not isinstance(config.get("seed"), int)
        or isinstance(config.get("seed"), bool)
        or not 0 <= config["seed"] <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer in the range 0..2**32-1")
    if config.get("device") not in {"auto", "cpu"}:
        raise ValueError("device must be 'auto' or 'cpu' (the released code uses 'auto' for CUDA selection)")
    if _at(config, ("metadata", "unknown_patient_policy")) not in {"error", "drop"}:
        raise ValueError("metadata.unknown_patient_policy must be 'error' or 'drop'")
    metadata_columns = _at(config, ("metadata", "columns"))
    if not isinstance(metadata_columns, dict):
        raise ValueError("metadata.columns must be a mapping")
    mapped_columns = [metadata_columns.get(name) for name in ("patient_id", "image_path", "label")]
    if any(not isinstance(name, str) or not name.strip() for name in mapped_columns):
        raise ValueError("metadata column names must be non-empty strings")
    if len(set(mapped_columns)) != 3:
        raise ValueError("metadata patient_id/image_path/label columns must be distinct")

    workers = _at(config, ("dataset", "num_workers"))
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError("dataset.num_workers must be a non-negative integer")

    for key, expected in _ARCHITECTURE_PROFILE.items():
        _require_equal(config, ("model", key), expected)
    for path, expected in _SHARED_RELEASED_PROFILE.items():
        _require_equal(config, path, expected)
    for path, expected in _MODE_PROFILE[mode].items():
        _require_equal(config, path, expected)

    for path in (
        ("dataset", "image_size"),
        ("dataset", "batch_size"),
        ("training", "epochs"),
        ("training", "early_stopping_patience"),
    ):
        value = _at(config, path)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{'.'.join(path)} must be a positive integer")
    if float(_at(config, ("training", "lr"))) <= 0:
        raise ValueError("training.lr must be positive")


def load_baseline_config(path: str | Path, *, expected_mode: str | None = None) -> dict[str, Any]:
    config = load_config(path)
    validate_baseline_config(config, expected_mode=expected_mode)
    return config


def set_dotted_value(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """Return a copy with one dotted key updated."""
    if not dotted_key or any(not part for part in dotted_key.split(".")):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    out = deepcopy(config)
    target: dict[str, Any] = out
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override non-mapping key at {part!r}")
        target = child
    if parts[-1] not in target:
        raise ValueError(f"Unknown override key: {dotted_key}")
    target[parts[-1]] = value
    return out


def parse_scalar(raw: str) -> Any:
    """Parse one CLI scalar using YAML while rejecting containers."""
    value = yaml.safe_load(raw)
    if isinstance(value, (dict, list)):
        raise ValueError("CLI overrides accept scalar values only")
    return value


def apply_safe_overrides(config: dict[str, Any], items: Iterable[str]) -> dict[str, Any]:
    """Apply the small runtime-only override surface exposed by the runners."""
    allowed = {
        "seed",
        "device",
        "dataset.num_workers",
        "metadata.columns.patient_id",
        "metadata.columns.image_path",
        "metadata.columns.label",
        "metadata.unknown_patient_policy",
    }
    out = deepcopy(config)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must have KEY=VALUE form: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in allowed:
            raise ValueError(
                f"Override {key!r} is not runtime-safe. Allowed keys: {sorted(allowed)}"
            )
        out = set_dotted_value(out, key, parse_scalar(raw))
    validate_baseline_config(out, expected_mode=_at(config, ("experiment", "baseline_mode")))
    return out


def project_official_config(
    config: dict[str, Any],
    released_config: dict[str, Any],
    *,
    metadata_csv: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create the external LungDxFormer config without changing its checkout."""
    validate_baseline_config(config)
    official = deepcopy(released_config)
    required_sections = {
        "seed", "device", "dataset", "augmentation", "model", "training", "evaluation", "paths"
    }
    missing = required_sections.difference(official)
    if missing:
        raise ValueError(f"Pinned released config is missing sections: {sorted(missing)}")

    # The architecture is always the released architecture. Protocol fields are
    # supplied externally for B1; B0 values are identical to released defaults.
    official["seed"] = config["seed"]
    official["device"] = config["device"]
    for section in ("dataset", "augmentation", "model", "training", "evaluation"):
        for key, value in config[section].items():
            if key in official[section]:
                official[section][key] = deepcopy(value)

    official["dataset"]["metadata_csv"] = str(Path(metadata_csv).resolve())
    official["dataset"]["image_root"] = str(Path(image_root).resolve())

    root = Path(output_dir).resolve()
    official["paths"] = {
        "checkpoint_dir": str(root / "checkpoints"),
        "metrics_dir": str(root / "metrics"),
        "plots_dir": str(root / "plots"),
        "predictions_dir": str(root / "predictions"),
    }
    return official
