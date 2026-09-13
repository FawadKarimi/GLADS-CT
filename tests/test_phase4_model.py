from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_model import ConsenDxFormer


def _model() -> ConsenDxFormer:
    return ConsenDxFormer(
        image_size=64,
        in_channels=1,
        num_classes=3,
        cnn_channels=(32, 64, 128),
        embed_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        transformer_mlp_ratio=4.0,
        dropout=0.5,
        classifier_hidden_dim=128,
        use_positional_encoding=True,
        use_spatial_attention=True,
        use_hierarchical_fusion=True,
    )


def test_forward_returns_expected_multitask_outputs():
    model = _model()
    model.eval()

    x = torch.randn(2, 1, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert output["logits"].shape == (2, 3)
    assert output["probs"].shape == (2, 3)
    assert output["ordinal_logits"].shape == (2, 4)
    assert output["pooled_features"].shape == (2, 128)

    assert output["attention_map"].shape == (2, 1, 8, 8)

    spatial_mass = output["attention_map"].flatten(1).sum(dim=1)
    assert torch.allclose(
        spatial_mass,
        torch.ones_like(spatial_mass),
        atol=1e-6,
    )


def test_encoder_exposes_expected_three_scales():
    model = _model()
    model.eval()

    x = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert output["c1_features"].shape == (1, 32, 32, 32)
    assert output["c2_features"].shape == (1, 64, 16, 16)
    assert output["c3_features"].shape == (1, 128, 8, 8)

    assert output["transformer_features"].shape == (1, 128, 8, 8)
    assert output["deep_fused_features"].shape == (1, 128, 8, 8)
    assert output["fused_features"].shape == (1, 128, 32, 32)


def test_hierarchical_fusion_can_be_disabled_for_ablation():
    model = ConsenDxFormer(
        image_size=64,
        in_channels=1,
        num_classes=3,
        cnn_channels=(32, 64, 128),
        embed_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        transformer_mlp_ratio=4.0,
        dropout=0.5,
        classifier_hidden_dim=128,
        use_positional_encoding=True,
        use_spatial_attention=True,
        use_hierarchical_fusion=False,
    )
    model.eval()

    x = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert output["deep_fused_features"].shape == (1, 128, 8, 8)
    assert output["fused_features"].shape == (1, 128, 8, 8)
    assert output["logits"].shape == (1, 3)
    assert output["ordinal_logits"].shape == (1, 4)


def test_attention_disabled_returns_uniform_spatial_distribution():
    model = ConsenDxFormer(
        image_size=64,
        in_channels=1,
        num_classes=3,
        cnn_channels=(32, 64, 128),
        embed_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        transformer_mlp_ratio=4.0,
        dropout=0.5,
        classifier_hidden_dim=128,
        use_positional_encoding=True,
        use_spatial_attention=False,
        use_hierarchical_fusion=True,
    )
    model.eval()

    x = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        attention = model(x)["attention_map"]

    expected = torch.full_like(attention, 1.0 / 64.0)
    assert torch.allclose(attention, expected, atol=1e-6)


def test_primary_and_ordinal_heads_both_receive_gradients():
    model = _model()
    model.train()

    x = torch.randn(2, 1, 64, 64)
    output = model(x)

    loss = output["logits"].sum() + output["ordinal_logits"].sum()
    loss.backward()

    assert model.classifier[-1].weight.grad is not None
    assert model.ordinal_head[-1].weight.grad is not None

    assert torch.isfinite(model.classifier[-1].weight.grad).all()
    assert torch.isfinite(model.ordinal_head[-1].weight.grad).all()
