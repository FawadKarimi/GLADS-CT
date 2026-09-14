from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_checkpoints import (
    Phase4CheckpointContext,
    build_phase4_checkpoint,
    load_and_validate_phase4_checkpoints,
    save_phase4_checkpoint,
    select_phase4_best_epoch,
)
from consendxformer.phase4_config import load_phase4_config, validate_phase4_config
from consendxformer.phase4_input_lock import (
    EXPECTED_PHASE4_INPUT_LOCK,
    Phase4InputLockError,
    load_phase4_input_lock,
    validate_phase4_input_lock,
)
from consendxformer.phase4_predictions import (
    PHASE4_PREDICTION_COLUMNS,
    build_phase4_prediction_table,
    evaluate_phase4_predictions,
)
from consendxformer.phase4_protocol import (
    ATTENTION_CANDIDATES,
    ORDINAL_CANDIDATES,
    Phase4ProtocolError,
    resolve_phase4_run_spec,
    tuning_protocol_record,
)
from consendxformer.phase4_runner import phase4_cli
import consendxformer.phase4_training as phase4_training
from consendxformer.phase4_training import (
    compute_phase4_loss,
    compute_training_class_weights,
)


def _phase4_config() -> dict:
    return deepcopy(
        load_phase4_config(ROOT / "configs" / "phase4_consensdxformer.yaml").phase4_config
    )


def test_versioned_phase4_input_lock_accepts_only_locked_readiness():
    lock = load_phase4_input_lock(ROOT / "configs" / "phase4_input_lock.json")
    assert lock.to_dict() == EXPECTED_PHASE4_INPUT_LOCK

    readiness = {
        "cohort_view": lock.cohort_view,
        "expected_resolution": lock.expected_resolution,
        "sample_count": lock.sample_count,
        "split_counts": dict(lock.split_counts),
        "reader_count_counts": dict(lock.reader_count_counts),
        "manifest_bundle_sha256": lock.manifest_bundle_sha256,
        "supervision_sha256": lock.supervision_sha256,
        "ordered_supervision_sha256": lock.ordered_supervision_sha256,
    }
    validate_phase4_input_lock(lock, readiness)

    changed = deepcopy(readiness)
    changed["ordered_supervision_sha256"] = "0" * 64
    with pytest.raises(Phase4InputLockError, match="ordered_supervision_sha256"):
        validate_phase4_input_lock(lock, changed)


def test_phase4_tuning_protocol_is_frozen_and_resolves_only_grid_values():
    record = tuning_protocol_record()
    assert record["stage1"]["candidate_count"] == 16
    assert record["stage1"]["seed"] == 42
    assert record["stage2"]["seeds"] == [43, 44]
    assert record["test_set_used_for_selection"] is False
    assert record["final_tie_break_order"] == [
        "mean_validation_macro_f1",
        "mean_validation_indeterminate_f1",
        "mean_validation_indeterminate_recall",
        "smaller_lambda_sum",
        "smaller_lambda_ordinal",
        "smaller_lambda_attention",
    ]

    config = _phase4_config()
    run = resolve_phase4_run_spec(
        config,
        mode="tuning",
        seed=43,
        lambda_ordinal=ORDINAL_CANDIDATES[2],
        lambda_attention=ATTENTION_CANDIDATES[1],
    )
    assert run.mode == "tuning"
    assert run.seed == 43
    assert run.lambda_ordinal == 0.25
    assert run.lambda_attention == 0.05
    assert run.evaluate_test is False

    with pytest.raises(Phase4ProtocolError, match="candidate"):
        resolve_phase4_run_spec(
            config,
            mode="tuning",
            lambda_ordinal=0.2,
            lambda_attention=0.1,
        )


def test_hierarchical_fusion_config_cannot_disagree_with_executed_ablation():
    bundle = load_phase4_config(ROOT / "configs" / "phase4_consensdxformer.yaml")
    changed = deepcopy(bundle.phase4_config)
    changed["model"]["use_hierarchical_fusion"] = False
    with pytest.raises(ValueError, match="hierarchical_fusion"):
        validate_phase4_config(changed, b1_anchor_config=bundle.b1_anchor_config)


def test_final_mode_requires_frozen_selected_lambdas_and_forbids_overrides():
    config = _phase4_config()

    unfrozen = deepcopy(config)
    unfrozen["losses"]["selected_lambda_ordinal"] = None
    unfrozen["losses"]["selected_lambda_attention"] = None
    with pytest.raises(Phase4ProtocolError, match="selected.*null"):
        resolve_phase4_run_spec(unfrozen, mode="final")

    final = resolve_phase4_run_spec(config, mode="final")
    assert final.evaluate_test is True
    assert (final.lambda_ordinal, final.lambda_attention) == (0.5, 0.05)

    with pytest.raises(Phase4ProtocolError, match="override"):
        resolve_phase4_run_spec(
            config,
            mode="final",
            lambda_ordinal=0.5,
            lambda_attention=None,
        )


def test_phase4_cli_rejects_final_lambda_override_before_any_execution():
    with pytest.raises(Phase4ProtocolError, match="override"):
        phase4_cli(
            [
                "--config",
                str(ROOT / "configs" / "phase4_consensdxformer.yaml"),
                "--mode",
                "final",
                "--lambda-ordinal",
                "0.25",
            ]
        )


def test_class_weights_use_train_hard_labels_only_and_match_b1_formula():
    supervision = pd.DataFrame(
        {
            "split": ["train", "train", "train", "train", "val", "test"],
            "label": [
                "benign",
                "benign",
                "indeterminate",
                "malignant",
                "malignant",
                "malignant",
            ],
        }
    )
    weights = compute_training_class_weights(supervision, num_classes=3)
    expected_b1 = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(3),
        y=np.asarray([0, 0, 1, 2]),
    )
    np.testing.assert_array_equal(weights, expected_b1)

    changed_eval = supervision.copy()
    changed_eval.loc[changed_eval["split"] != "train", "label"] = "benign"
    np.testing.assert_array_equal(
        compute_training_class_weights(changed_eval, num_classes=3),
        weights,
    )


def _loss_inputs():
    outputs = {
        "logits": torch.tensor(
            [[0.4, -0.2, 0.1], [-0.3, 0.7, 0.2]], requires_grad=True
        ),
        "ordinal_logits": torch.tensor(
            [[0.3, 0.2, -0.1, -0.4], [0.9, 0.4, 0.2, -0.2]],
            requires_grad=True,
        ),
        "attention_map": torch.tensor(
            [
                [[[0.2, 0.8], [0.4, 0.6]]],
                [[[0.7, 0.3], [0.5, 0.5]]],
            ],
            requires_grad=True,
        ),
    }
    batch = {
        "label": torch.tensor([0, 1]),
        "soft_label": torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.7, 0.2]]),
        "disagreement": torch.tensor([0.2, 0.6]),
        "ordinal_target": torch.tensor(
            [[1.0, 0.5, 0.0, 0.0], [1.0, 1.0, 0.5, 0.0]]
        ),
        "soft_consensus_mask": torch.tensor(
            [
                [[[0.0, 1.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [0.0, 0.0]]],
            ]
        ),
    }
    return outputs, batch


def test_phase4_total_loss_arithmetic_and_zero_lambda_contributions():
    outputs, batch = _loss_inputs()
    ablation = {
        "use_disagreement_weighting": True,
        "use_ordinal_head": True,
        "use_attention_alignment": True,
        "use_hierarchical_fusion": True,
    }
    result = compute_phase4_loss(
        outputs,
        batch,
        lambda_ordinal=0.25,
        lambda_attention=0.1,
        class_weights=torch.tensor([1.0, 1.5, 2.0]),
        ablation=ablation,
    )
    torch.testing.assert_close(
        result.total,
        result.diagnostic + 0.25 * result.ordinal + 0.1 * result.attention,
    )

    zero = compute_phase4_loss(
        outputs,
        batch,
        lambda_ordinal=0.0,
        lambda_attention=0.0,
        class_weights=None,
        ablation=ablation,
    )
    torch.testing.assert_close(zero.total, zero.diagnostic)
    assert zero.ordinal.item() == 0.0
    assert zero.attention.item() == 0.0


def test_disabled_auxiliary_ablations_do_not_compute_terms(monkeypatch):
    outputs, batch = _loss_inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled auxiliary loss was computed")

    monkeypatch.setattr(phase4_training, "ordinal_distribution_loss", forbidden)
    monkeypatch.setattr(phase4_training, "attention_alignment_loss", forbidden)
    monkeypatch.setattr(phase4_training, "diagnostic_disagreement_loss", forbidden)
    result = compute_phase4_loss(
        outputs,
        batch,
        lambda_ordinal=0.5,
        lambda_attention=0.25,
        class_weights=None,
        ablation={
            "use_disagreement_weighting": False,
            "use_ordinal_head": False,
            "use_attention_alignment": False,
            "use_hierarchical_fusion": False,
        },
    )
    assert result.ordinal.item() == 0.0
    assert result.attention.item() == 0.0
    torch.testing.assert_close(result.total, result.diagnostic)


def _checkpoint_context() -> Phase4CheckpointContext:
    return Phase4CheckpointContext(
        experiment_id="phase4-tuning-s42-0123456789abcdef",
        run_mode="tuning",
        seed=42,
        lambda_ordinal=0.1,
        lambda_attention=0.05,
        project_commit="a" * 40,
        config_sha256="b" * 64,
        manifest_bundle_sha256="c" * 64,
        phase3_input_fingerprint={"ordered_samples_sha256": "d" * 64},
        supervision_sha256="e" * 64,
        ordered_supervision_sha256="f" * 64,
        input_lock_sha256="1" * 64,
        cohort_view="paper_aligned_multi_reader_2_to_4",
        expected_resolution=64,
        sample_count=3,
        split_counts={"train": 1, "val": 1, "test": 1},
    )


def test_best_epoch_selection_uses_validation_total_loss_not_macro_f1():
    history = [
        {
            "epoch": 1,
            "train": {"loss": 0.5, "total_loss": 0.5},
            "validation": {"loss": 0.4, "total_loss": 0.4, "macro_f1": 0.99},
        },
        {
            "epoch": 2,
            "train": {"loss": 0.4, "total_loss": 0.4},
            "validation": {"loss": 0.2, "total_loss": 0.2, "macro_f1": 0.10},
        },
    ]
    assert select_phase4_best_epoch(history)["epoch"] == 2


def test_persisted_checkpoint_snapshots_are_authoritative_and_distinguishable(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    context = _checkpoint_context()
    history = []

    with torch.no_grad():
        model.weight.fill_(1.0)
    best_metrics = {"loss": 0.2, "total_loss": 0.2, "macro_f1": 0.7}
    best = build_phase4_checkpoint(
        checkpoint_kind="best",
        context=context,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        epoch=1,
        best_epoch=1,
        best_validation_loss=0.2,
        validation_metrics=best_metrics,
    )
    best_path = save_phase4_checkpoint(best, tmp_path / "best.pt")
    history.append({"epoch": 1, "train": {"loss": 0.5}, "validation": best_metrics})

    # PyTorch state_dict values can alias live module storage. Change that source
    # after best.pt has been serialized, then persist a genuinely different last.pt.
    with torch.no_grad():
        model.weight.fill_(9.0)
    last_metrics = {"loss": 0.4, "total_loss": 0.4, "macro_f1": 0.6}
    last = build_phase4_checkpoint(
        checkpoint_kind="last",
        context=context,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        epoch=2,
        best_epoch=1,
        best_validation_loss=0.2,
        validation_metrics=last_metrics,
    )
    last_path = save_phase4_checkpoint(last, tmp_path / "last.pt")
    history.append({"epoch": 2, "train": {"loss": 0.4}, "validation": last_metrics})

    persisted_best, persisted_last = load_and_validate_phase4_checkpoints(
        best_path,
        last_path,
        context=context,
        history=history,
        map_location="cpu",
    )
    assert persisted_best["model_state"]["weight"].item() == 1.0
    assert persisted_last["model_state"]["weight"].item() == 9.0
    assert persisted_best["supervision_sha256"] == "e" * 64
    assert persisted_best["ordered_supervision_sha256"] == "f" * 64
    assert persisted_last["supervision_sha256"] == "e" * 64
    assert persisted_last["ordered_supervision_sha256"] == "f" * 64

    model.load_state_dict(persisted_best["model_state"])
    assert model.weight.item() == 1.0
    # Loading best for evaluation must not mutate the authoritative last record.
    assert persisted_last["model_state"]["weight"].item() == 9.0


def test_phase4_prediction_contract_and_metrics_include_indeterminate_statistics():
    metadata = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "nodule_id": ["n1", "n2", "n3"],
            "series_instance_uid": ["s1", "s2", "s3"],
            "split": ["val", "val", "val"],
            "label": ["benign", "indeterminate", "malignant"],
        }
    )
    logits = np.asarray([[3.0, 0.0, -1.0], [0.0, 2.0, 0.0], [-1.0, 0.0, 3.0]])
    table = build_phase4_prediction_table(
        metadata,
        logits,
        seed=42,
        lambda_ordinal=0.25,
        lambda_attention=0.1,
        run_mode="tuning",
        experiment_id="phase4-tuning-s42-0123456789abcdef",
    )
    assert tuple(table.columns) == PHASE4_PREDICTION_COLUMNS
    assert table["nodule_id"].tolist() == ["n1", "n2", "n3"]
    metrics = evaluate_phase4_predictions(table)
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["indeterminate"]["f1"] == 1.0
    assert metrics["per_class"]["indeterminate"]["recall"] == 1.0
