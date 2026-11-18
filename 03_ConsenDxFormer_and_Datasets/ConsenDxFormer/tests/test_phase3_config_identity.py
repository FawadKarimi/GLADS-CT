from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.baseline_source import PINNED_BASELINE_SHA
from consendxformer.manifest_guard import FROZEN_BUNDLE_SHA256
from consendxformer.phase3_config import (
    SYNTHETIC_MARKER,
    load_phase3_config,
    validate_phase3_config,
)
from consendxformer.phase3_identity import (
    build_experiment_identity,
    experiment_identity_payload,
)


PROJECT_COMMIT = "7" * 40


def _identity(**overrides):
    values = {
        "baseline_variant": "B0",
        "seed": 2026,
        "baseline_source_commit": PINNED_BASELINE_SHA,
        "project_commit": PROJECT_COMMIT,
        "combined_config_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "cohort_sha256": "c" * 64,
        "ordered_sample_sha256": "d" * 64,
        "frozen_split_bundle_sha256": FROZEN_BUNDLE_SHA256,
    }
    values.update(overrides)
    return build_experiment_identity(**values)


@pytest.mark.parametrize(
    ("name", "variant", "image_size", "batch_size", "epochs", "dropout"),
    [
        ("phase3_b0.yaml", "B0", 96, 16, 6, 0.2),
        ("phase3_b1.yaml", "B1", 64, 32, 100, 0.5),
    ],
)
def test_phase3_config_selects_exact_locked_phase1_profile(
    name, variant, image_size, batch_size, epochs, dropout
):
    bundle = load_phase3_config(ROOT / "configs" / name, expected_variant=variant)

    assert bundle.phase3_config["experiment"]["baseline_variant"] == variant
    assert bundle.phase1_config_path == ROOT / "configs" / f"{variant.lower()}.yaml"
    assert bundle.effective_phase1_config["dataset"]["image_size"] == image_size
    assert bundle.effective_phase1_config["dataset"]["batch_size"] == batch_size
    assert bundle.effective_phase1_config["training"]["epochs"] == epochs
    assert bundle.effective_phase1_config["model"]["dropout"] == dropout
    assert bundle.phase3_config["predictions"]["class_order"] == [
        "benign",
        "indeterminate",
        "malignant",
    ]


def test_phase3_does_not_duplicate_locked_baseline_hyperparameters():
    for name in ("phase3_common.yaml", "phase3_b0.yaml", "phase3_b1.yaml"):
        raw = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert "dataset" not in raw
        assert "model" not in raw
        assert "training" not in raw
        assert "augmentation" not in raw


def test_invalid_baseline_and_synthetic_masquerade_are_rejected():
    bundle = load_phase3_config(ROOT / "configs" / "phase3_synthetic_b0.yaml")
    invalid = deepcopy(bundle.phase3_config)
    invalid["experiment"]["baseline_variant"] = "B2"
    with pytest.raises(ValueError, match="baseline_variant"):
        validate_phase3_config(invalid)

    masquerade = deepcopy(bundle.phase3_config)
    masquerade["data"]["mode"] = "real"
    with pytest.raises(ValueError, match="synthetic_marker"):
        validate_phase3_config(masquerade)

    missing_marker = deepcopy(bundle.phase3_config)
    missing_marker["data"]["synthetic_marker"] = None
    with pytest.raises(ValueError, match=SYNTHETIC_MARKER):
        validate_phase3_config(missing_marker)


def test_phase3_runtime_seed_override_updates_dependency_and_hash():
    path = ROOT / "configs" / "phase3_b0.yaml"
    first = load_phase3_config(path, seed=11)
    second = load_phase3_config(path, seed=12)

    assert first.phase3_config["experiment"]["seed"] == 11
    assert first.effective_phase1_config["seed"] == 11
    assert second.phase3_config["experiment"]["seed"] == 12
    assert first.combined_config_sha256 != second.combined_config_sha256


def test_experiment_identity_is_deterministic_and_content_addressed():
    first = _identity()
    second = _identity()

    assert first == second
    assert first.experiment_id == second.experiment_id
    assert first.experiment_id.startswith("phase3-b0-s2026-")
    assert len(first.digest) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 2027),
        ("combined_config_sha256", "e" * 64),
        ("metadata_sha256", "f" * 64),
        ("cohort_sha256", "1" * 64),
        ("ordered_sample_sha256", "2" * 64),
        ("frozen_split_bundle_sha256", "3" * 64),
    ],
)
def test_each_required_input_changes_experiment_identity(field, value):
    assert _identity(**{field: value}).experiment_id != _identity().experiment_id


def test_identity_payload_explicitly_contains_cohort_and_frozen_split_checksums():
    identity = _identity()
    payload = experiment_identity_payload(
        baseline_variant=identity.baseline_variant,
        seed=identity.seed,
        baseline_source_commit=identity.baseline_source_commit,
        project_commit=identity.project_commit,
        combined_config_sha256=identity.combined_config_sha256,
        metadata_sha256=identity.metadata_sha256,
        cohort_sha256=identity.cohort_sha256,
        ordered_sample_sha256=identity.ordered_sample_sha256,
        frozen_split_bundle_sha256=identity.frozen_split_bundle_sha256,
    )

    assert payload["cohort_sha256"] == "c" * 64
    assert payload["frozen_split_bundle_sha256"] == FROZEN_BUNDLE_SHA256
    assert payload["baseline_source_commit"] == PINNED_BASELINE_SHA


def test_identity_rejects_any_unlocked_baseline_commit():
    with pytest.raises(ValueError, match="immutable Phase 1 source pin"):
        _identity(baseline_source_commit="0" * 40)
