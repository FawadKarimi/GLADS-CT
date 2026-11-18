from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_losses import (
    attention_alignment_loss,
    diagnostic_disagreement_loss,
    ordinal_distribution_loss,
    total_consensus_loss,
)


def test_diagnostic_loss_reduces_to_hard_ce_when_disagreement_is_zero():
    logits = torch.tensor(
        [[2.0, 0.5, -1.0], [0.2, 1.5, -0.3]],
        dtype=torch.float32,
    )
    hard_target = torch.tensor([0, 1], dtype=torch.long)
    soft_target = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, 0.7, 0.1]],
        dtype=torch.float32,
    )
    disagreement = torch.zeros(2, dtype=torch.float32)

    observed = diagnostic_disagreement_loss(
        logits,
        hard_target,
        soft_target,
        disagreement,
    )
    expected = F.cross_entropy(logits, hard_target)

    assert observed == pytest.approx(float(expected), rel=1e-6)


def test_diagnostic_loss_reduces_to_soft_ce_when_disagreement_is_one():
    logits = torch.tensor(
        [[1.2, 0.1, -0.6]],
        dtype=torch.float32,
    )
    hard_target = torch.tensor([0], dtype=torch.long)
    soft_target = torch.tensor(
        [[0.25, 0.50, 0.25]],
        dtype=torch.float32,
    )
    disagreement = torch.ones(1, dtype=torch.float32)

    observed = diagnostic_disagreement_loss(
        logits,
        hard_target,
        soft_target,
        disagreement,
    )

    log_probs = F.log_softmax(logits, dim=1)
    expected = -(soft_target * log_probs).sum(dim=1).mean()

    assert observed == pytest.approx(float(expected), rel=1e-6)


def test_ordinal_distribution_loss_accepts_soft_cumulative_targets():
    ordinal_logits = torch.tensor(
        [[2.0, 1.0, -0.5, -1.5]],
        dtype=torch.float32,
    )
    ordinal_target = torch.tensor(
        [[1.0, 0.75, 0.25, 0.0]],
        dtype=torch.float32,
    )

    loss = ordinal_distribution_loss(
        ordinal_logits,
        ordinal_target,
    )

    expected = F.binary_cross_entropy_with_logits(
        ordinal_logits,
        ordinal_target,
    )

    assert torch.isfinite(loss)
    assert loss == pytest.approx(float(expected), rel=1e-6)


def test_attention_alignment_prefers_matching_spatial_distribution():
    soft_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    soft_mask[:, :, :2, :2] = 1.0

    matching = torch.tensor(
        [[[[0.25, 0.25], [0.25, 0.25]]]],
        dtype=torch.float32,
    )
    mismatching = torch.tensor(
        [[[[0.01, 0.01], [0.01, 0.97]]]],
        dtype=torch.float32,
    )

    matching_loss = attention_alignment_loss(
        matching,
        soft_mask,
    )
    mismatching_loss = attention_alignment_loss(
        mismatching,
        soft_mask,
    )

    assert torch.isfinite(matching_loss)
    assert torch.isfinite(mismatching_loss)
    assert matching_loss < mismatching_loss


def test_total_loss_auxiliary_terms_are_independently_toggleable():
    logits = torch.tensor(
        [[0.8, 0.3, -0.2]],
        dtype=torch.float32,
    )
    hard_target = torch.tensor([0], dtype=torch.long)
    soft_target = torch.tensor(
        [[0.5, 0.5, 0.0]],
        dtype=torch.float32,
    )
    disagreement = torch.tensor([0.5], dtype=torch.float32)

    ordinal_logits = torch.tensor(
        [[1.0, 0.5, -0.5, -1.0]],
        dtype=torch.float32,
    )
    ordinal_target = torch.tensor(
        [[1.0, 0.5, 0.25, 0.0]],
        dtype=torch.float32,
    )

    attention_map = torch.full(
        (1, 1, 2, 2),
        0.25,
        dtype=torch.float32,
    )
    soft_mask = torch.ones(
        (1, 1, 4, 4),
        dtype=torch.float32,
    )

    diagnostic_only = total_consensus_loss(
        logits=logits,
        hard_target=hard_target,
        soft_target=soft_target,
        disagreement=disagreement,
        ordinal_logits=ordinal_logits,
        ordinal_target=ordinal_target,
        attention_map=attention_map,
        soft_consensus_mask=soft_mask,
        lambda_ordinal=0.0,
        lambda_attention=0.0,
    )

    with_auxiliary = total_consensus_loss(
        logits=logits,
        hard_target=hard_target,
        soft_target=soft_target,
        disagreement=disagreement,
        ordinal_logits=ordinal_logits,
        ordinal_target=ordinal_target,
        attention_map=attention_map,
        soft_consensus_mask=soft_mask,
        lambda_ordinal=0.5,
        lambda_attention=0.25,
    )

    assert diagnostic_only["total"] == pytest.approx(
        float(diagnostic_only["diagnostic"]),
        rel=1e-6,
    )
    assert with_auxiliary["total"] > diagnostic_only["total"]
    assert set(with_auxiliary) == {
        "total",
        "diagnostic",
        "ordinal",
        "attention",
    }


def test_diagnostic_loss_matches_b1_weighted_ce_when_disagreement_is_zero():
    """At u=0, Phase 4 must reproduce B1's weighted CE reduction exactly."""
    logits = torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [0.2, 1.5, -0.3],
            [-0.4, 0.1, 1.8],
        ],
        dtype=torch.float32,
    )
    hard_target = torch.tensor([0, 1, 2], dtype=torch.long)
    soft_target = F.one_hot(
        hard_target,
        num_classes=3,
    ).to(dtype=torch.float32)
    disagreement = torch.zeros(3, dtype=torch.float32)
    class_weights = torch.tensor(
        [1.0, 3.0, 2.0],
        dtype=torch.float32,
    )

    observed = diagnostic_disagreement_loss(
        logits,
        hard_target,
        soft_target,
        disagreement,
        class_weights=class_weights,
    )

    expected = F.cross_entropy(
        logits,
        hard_target,
        weight=class_weights,
    )

    assert observed == pytest.approx(
        float(expected),
        rel=1e-6,
        abs=1e-7,
    )


def test_attention_alignment_is_zero_for_exact_distribution_match():
    """Exact consensus-attention agreement should have zero KL alignment loss."""
    soft_mask = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]],
        dtype=torch.float32,
    )

    attention = soft_mask / soft_mask.sum(dim=(2, 3), keepdim=True)

    observed = attention_alignment_loss(
        attention,
        soft_mask,
    )

    assert observed == pytest.approx(
        0.0,
        abs=1e-6,
    )
