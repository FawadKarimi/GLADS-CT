from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.baseline_config import (
    CLASS_NAMES,
    apply_safe_overrides,
    load_baseline_config,
    project_official_config,
)


def test_b0_config_loads_as_released_three_class_baseline():
    cfg = load_baseline_config(ROOT / "configs" / "b0.yaml", expected_mode="B0")
    assert cfg["experiment"]["baseline_mode"] == "B0"
    assert cfg["experiment"]["protocol"] == "released-code"
    assert cfg["dataset"]["num_classes"] == 3
    assert tuple(cfg["dataset"]["class_names"]) == CLASS_NAMES
    assert cfg["dataset"]["image_size"] == 96
    assert cfg["dataset"]["batch_size"] == 16
    assert cfg["training"]["epochs"] == 6
    assert cfg["model"]["dropout"] == 0.2
    assert cfg["training"]["early_stopping_patience"] == 8


def test_b1_config_loads_as_paper_aligned_three_class_baseline():
    cfg = load_baseline_config(ROOT / "configs" / "b1.yaml", expected_mode="B1")
    assert cfg["experiment"]["baseline_mode"] == "B1"
    assert cfg["experiment"]["protocol"] == "paper-aligned-released-architecture"
    assert cfg["dataset"]["num_classes"] == 3
    assert tuple(cfg["dataset"]["class_names"]) == CLASS_NAMES
    assert cfg["dataset"]["image_size"] == 64
    assert cfg["dataset"]["batch_size"] == 32
    assert cfg["training"]["epochs"] == 100
    assert cfg["training"]["lr"] == pytest.approx(1e-4)
    assert cfg["model"]["dropout"] == 0.5
    assert cfg["training"]["early_stopping_patience"] == 10


def test_b0_and_b1_are_explicit_but_share_the_locked_source_and_architecture():
    b0 = load_baseline_config(ROOT / "configs" / "b0.yaml")
    b1 = load_baseline_config(ROOT / "configs" / "b1.yaml")
    assert b0["experiment"] != b1["experiment"]
    assert b0["baseline"] == b1["baseline"]
    for key in (
        "cnn_channels",
        "embed_dim",
        "transformer_heads",
        "transformer_layers",
        "transformer_mlp_ratio",
        "classifier_hidden_dim",
        "use_transformer",
        "use_positional_encoding",
        "use_spatial_attention",
        "fusion_type",
        "use_raw_transformer_in_fusion",
    ):
        assert b0["model"][key] == b1["model"][key]


def test_only_runtime_safe_cli_overrides_are_accepted():
    b0 = load_baseline_config(ROOT / "configs" / "b0.yaml")
    updated = apply_safe_overrides(
        b0,
        ["seed=7", "device=cpu", "dataset.num_workers=2", "metadata.columns.label=target"],
    )
    assert updated["seed"] == 7
    assert updated["device"] == "cpu"
    assert updated["dataset"]["num_workers"] == 2
    assert updated["metadata"]["columns"]["label"] == "target"

    with pytest.raises(ValueError, match="not runtime-safe"):
        apply_safe_overrides(b0, ["model.embed_dim=256"])


def test_official_projection_keeps_source_shape_and_uses_external_paths(tmp_path):
    b0 = load_baseline_config(ROOT / "configs" / "b0.yaml")
    released = {
        "seed": 42,
        "device": "auto",
        "dataset": dict(b0["dataset"], metadata_csv="old.csv", image_root="old"),
        "augmentation": b0["augmentation"],
        "model": b0["model"],
        "training": b0["training"],
        "evaluation": b0["evaluation"],
        "paths": {"checkpoint_dir": "old", "metrics_dir": "old", "plots_dir": "old", "predictions_dir": "old"},
    }
    metadata = tmp_path / "prepared" / "metadata.csv"
    image_root = tmp_path / "images"
    output = tmp_path / "run"
    projected = project_official_config(
        b0,
        released,
        metadata_csv=metadata,
        image_root=image_root,
        output_dir=output,
    )
    assert projected["dataset"]["metadata_csv"] == str(metadata.resolve())
    assert projected["dataset"]["image_root"] == str(image_root.resolve())
    assert projected["paths"]["checkpoint_dir"] == str((output / "checkpoints").resolve())
    assert projected["model"] == b0["model"]
