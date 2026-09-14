from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_config import (
    load_phase4_config,
    validate_phase4_config,
)


def test_phase4_config_uses_locked_b1_fairness_anchor():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )

    cfg = bundle.phase4_config
    b1 = bundle.b1_anchor_config

    assert cfg["project"]["phase"] == 4
    assert cfg["experiment"]["model_name"] == "ConsenDxFormer"
    assert cfg["fairness_anchor"]["config"] == "configs/b1.yaml"

    assert cfg["dataset"]["image_size"] == b1["dataset"]["image_size"] == 64
    assert cfg["dataset"]["in_channels"] == b1["dataset"]["in_channels"] == 1
    assert cfg["dataset"]["num_classes"] == b1["dataset"]["num_classes"] == 3
    assert cfg["dataset"]["batch_size"] == b1["dataset"]["batch_size"] == 32
    assert cfg["dataset"]["num_workers"] == b1["dataset"]["num_workers"] == 0

    assert cfg["training"]["epochs"] == b1["training"]["epochs"] == 100
    assert cfg["training"]["lr"] == b1["training"]["lr"] == pytest.approx(1e-4)
    assert cfg["training"]["weight_decay"] == b1["training"]["weight_decay"]
    assert cfg["training"]["early_stopping_patience"] == 10

    assert cfg["model"]["cnn_channels"] == [32, 64, 128]
    assert cfg["model"]["embed_dim"] == 128
    assert cfg["model"]["transformer_heads"] == 4
    assert cfg["model"]["transformer_layers"] == 2
    assert cfg["model"]["dropout"] == pytest.approx(0.5)


def test_phase4_config_locks_primary_multirater_cohort():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )
    data = bundle.phase4_config["data"]

    assert data["cohort_view"] == "paper_aligned_multi_reader_2_to_4"
    assert data["metadata_path"].endswith(
        "rendered_64/baseline_metadata_2_to_4.csv"
    )
    assert data["rendered_dir"].endswith(
        "consendxformer_phase2/rendered_64"
    )
    assert data["canonical_dir"].endswith(
        "consendxformer_phase2"
    )


def test_phase4_loss_weights_are_validation_candidates_not_test_tuned():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )
    losses = bundle.phase4_config["losses"]

    assert losses["selection_split"] == "val"
    assert losses["test_tuning_forbidden"] is True

    ordinal = losses["lambda_ordinal_candidates"]
    attention = losses["lambda_attention_candidates"]

    assert 0.0 in ordinal
    assert 0.0 in attention
    assert all(value >= 0.0 for value in ordinal)
    assert all(value >= 0.0 for value in attention)

    assert losses["selected_lambda_ordinal"] == 0.5
    assert losses["selected_lambda_attention"] == 0.05
    assert losses["selected_lambda_ordinal"] in ordinal
    assert losses["selected_lambda_attention"] in attention


def test_phase4_ablation_toggles_are_explicit():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )
    ablation = bundle.phase4_config["ablation"]

    assert ablation == {
        "use_disagreement_weighting": True,
        "use_ordinal_head": True,
        "use_attention_alignment": True,
        "use_hierarchical_fusion": True,
    }


def test_validator_rejects_changed_b1_training_budget():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )

    bad = dict(bundle.phase4_config)
    bad["training"] = dict(bad["training"])
    bad["training"]["epochs"] = 99

    with pytest.raises(ValueError, match="fairness anchor"):
        validate_phase4_config(
            bad,
            b1_anchor_config=bundle.b1_anchor_config,
        )


def test_validator_rejects_test_based_loss_selection():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )

    bad = dict(bundle.phase4_config)
    bad["losses"] = dict(bad["losses"])
    bad["losses"]["selection_split"] = "test"

    with pytest.raises(ValueError, match="validation"):
        validate_phase4_config(
            bad,
            b1_anchor_config=bundle.b1_anchor_config,
        )


def test_phase4_config_locks_real_data_lineage_contract():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )
    data = bundle.phase4_config["data"]

    assert data["mode"] == "real"
    assert (
        data["source_readiness_report"]
        == "artifacts/phase2/runtime/source_readiness.json"
    )
    assert data["canonical_dir"] == "data/processed/consendxformer_phase2"
    assert (
        data["rendered_dir"]
        == "data/processed/consendxformer_phase2/rendered_64"
    )
    assert data["frozen_split_dir"] == "artifacts/phase0/manifests"
    assert data["synthetic_marker"] is None


def test_phase4_config_preserves_phase3_determinism_and_inference_guards():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )
    cfg = bundle.phase4_config

    assert cfg["determinism"] == {
        "deterministic_algorithms": True,
        "warn_only": True,
        "cudnn_benchmark": False,
        "dataloader_worker_seeding": True,
        "seeded_dataloader_generator": True,
    }

    assert cfg["checkpoint"] == {
        "selection_criterion": "validation_loss",
        "selection_mode": "min",
        "save_best": True,
        "save_last": True,
    }

    assert cfg["predictions"] == {
        "format": "csv",
        "splits": ["val", "test"],
        "class_order": ["benign", "indeterminate", "malignant"],
        "export_logits": True,
        "export_probabilities": True,
    }

    assert cfg["inference"] == {
        "tta": False,
        "ema": False,
        "pretrained_weights": False,
    }


def test_validator_rejects_weakened_real_data_lineage():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )

    bad = dict(bundle.phase4_config)
    bad["data"] = dict(bad["data"])
    bad["data"]["mode"] = "synthetic"

    with pytest.raises(ValueError, match="real-data"):
        validate_phase4_config(
            bad,
            b1_anchor_config=bundle.b1_anchor_config,
        )


def test_validator_rejects_weakened_determinism_policy():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )

    bad = dict(bundle.phase4_config)
    bad["determinism"] = dict(bad["determinism"])
    bad["determinism"]["cudnn_benchmark"] = True

    with pytest.raises(ValueError, match="determinism"):
        validate_phase4_config(
            bad,
            b1_anchor_config=bundle.b1_anchor_config,
        )


def test_validator_rejects_changed_checkpoint_selection():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )

    bad = dict(bundle.phase4_config)
    bad["checkpoint"] = dict(bad["checkpoint"])
    bad["checkpoint"]["selection_criterion"] = "validation_macro_f1"

    with pytest.raises(ValueError, match="checkpoint"):
        validate_phase4_config(
            bad,
            b1_anchor_config=bundle.b1_anchor_config,
        )


def test_validator_rejects_test_time_augmentation():
    bundle = load_phase4_config(
        ROOT / "configs" / "phase4_consensdxformer.yaml"
    )

    bad = dict(bundle.phase4_config)
    bad["inference"] = dict(bad["inference"])
    bad["inference"]["tta"] = True

    with pytest.raises(ValueError, match="inference"):
        validate_phase4_config(
            bad,
            b1_anchor_config=bundle.b1_anchor_config,
        )
