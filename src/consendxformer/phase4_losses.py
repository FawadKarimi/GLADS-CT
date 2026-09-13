"""Losses for disagreement-aware ConsenDxFormer training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


_EPS = 1e-8


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


def diagnostic_disagreement_loss(
    logits: torch.Tensor,
    hard_target: torch.Tensor,
    soft_target: torch.Tensor,
    disagreement: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend hard and soft diagnostic supervision using reader disagreement.

    For sample i:

        L_i = (1 - u_i) * L_hard_i + u_i * L_soft_i

    where ``u_i`` is the normalized multi-rater disagreement entropy in [0, 1].

    Class weights, when supplied, are applied consistently to both the hard
    and soft class contributions.
    """

    _require_finite("logits", logits)
    _require_finite("soft_target", soft_target)
    _require_finite("disagreement", disagreement)

    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, C]")
    if hard_target.ndim != 1 or hard_target.shape[0] != logits.shape[0]:
        raise ValueError("hard_target must have shape [B]")
    if soft_target.shape != logits.shape:
        raise ValueError("soft_target must have the same shape as logits")
    if disagreement.ndim != 1 or disagreement.shape[0] != logits.shape[0]:
        raise ValueError("disagreement must have shape [B]")

    if hard_target.dtype != torch.long:
        raise ValueError("hard_target must use torch.long dtype")

    if torch.any(soft_target < 0.0) or torch.any(soft_target > 1.0):
        raise ValueError("soft_target probabilities must lie in [0, 1]")
    if not torch.allclose(
        soft_target.sum(dim=1),
        torch.ones(
            logits.shape[0],
            device=soft_target.device,
            dtype=soft_target.dtype,
        ),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("soft_target probabilities must sum to 1")

    if torch.any(disagreement < 0.0) or torch.any(disagreement > 1.0):
        raise ValueError("disagreement values must lie in [0, 1]")

    weights = None
    if class_weights is not None:
        _require_finite("class_weights", class_weights)
        if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
            raise ValueError("class_weights must have shape [C]")
        if torch.any(class_weights <= 0):
            raise ValueError("class_weights must be strictly positive")
        weights = class_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        )

    log_probs = F.log_softmax(logits, dim=1)

    disagreement = disagreement.to(
        device=logits.device,
        dtype=logits.dtype,
    )
    soft_target = soft_target.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    if weights is None:
        hard_numerator = F.cross_entropy(
            logits,
            hard_target,
            reduction="none",
        )
        hard_mass = torch.ones_like(hard_numerator)

        soft_numerator = -(soft_target * log_probs).sum(dim=1)
        soft_mass = torch.ones_like(soft_numerator)
    else:
        hard_mass = weights[hard_target]
        hard_numerator = (
            F.cross_entropy(
                logits,
                hard_target,
                reduction="none",
            )
            * hard_mass
        )

        weighted_target = soft_target * weights.unsqueeze(0)
        soft_mass = weighted_target.sum(dim=1)
        soft_numerator = -(weighted_target * log_probs).sum(dim=1)

    blended_numerator = (
        (1.0 - disagreement) * hard_numerator
        + disagreement * soft_numerator
    )
    blended_mass = (
        (1.0 - disagreement) * hard_mass
        + disagreement * soft_mass
    )

    return blended_numerator.sum() / blended_mass.sum().clamp_min(_EPS)


def ordinal_distribution_loss(
    ordinal_logits: torch.Tensor,
    ordinal_target: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy for four cumulative ordinal thresholds."""

    _require_finite("ordinal_logits", ordinal_logits)
    _require_finite("ordinal_target", ordinal_target)

    if ordinal_logits.ndim != 2 or ordinal_logits.shape[1] != 4:
        raise ValueError("ordinal_logits must have shape [B, 4]")
    if ordinal_target.shape != ordinal_logits.shape:
        raise ValueError("ordinal_target must have shape [B, 4]")

    if torch.any(ordinal_target < 0.0) or torch.any(ordinal_target > 1.0):
        raise ValueError("ordinal_target values must lie in [0, 1]")

    if torch.any(ordinal_target[:, 1:] > ordinal_target[:, :-1] + 1e-7):
        raise ValueError("ordinal_target values must be non-increasing")

    return F.binary_cross_entropy_with_logits(
        ordinal_logits,
        ordinal_target.to(
            device=ordinal_logits.device,
            dtype=ordinal_logits.dtype,
        ),
    )


def _normalized_spatial_target(
    soft_consensus_mask: torch.Tensor,
    *,
    spatial_size: tuple[int, int],
) -> torch.Tensor:
    _require_finite("soft_consensus_mask", soft_consensus_mask)

    if (
        soft_consensus_mask.ndim != 4
        or soft_consensus_mask.shape[1] != 1
    ):
        raise ValueError(
            "soft_consensus_mask must have shape [B, 1, H, W]"
        )

    if torch.any(soft_consensus_mask < 0.0):
        raise ValueError("soft_consensus_mask cannot contain negative values")

    resized = F.interpolate(
        soft_consensus_mask.float(),
        size=spatial_size,
        mode="area",
    )

    flat = resized.flatten(1)
    mass = flat.sum(dim=1, keepdim=True)

    # A valid nodule mask should contain positive spatial mass. Failing here is
    # preferable to silently inventing an attention target.
    if torch.any(mass <= 0.0):
        raise ValueError(
            "soft_consensus_mask must contain positive spatial mass"
        )

    normalized = flat / mass
    return normalized


def attention_alignment_loss(
    attention_map: torch.Tensor,
    soft_consensus_mask: torch.Tensor,
) -> torch.Tensor:
    """KL divergence from consensus spatial distribution to native DSA attention.

    LungDxFormer's DSA is a spatial softmax distribution, not a sigmoid mask.
    The rendered soft consensus mask is therefore resized to the DSA grid and
    normalized to unit spatial mass before alignment.
    """

    _require_finite("attention_map", attention_map)

    if attention_map.ndim != 4 or attention_map.shape[1] != 1:
        raise ValueError("attention_map must have shape [B, 1, H, W]")

    if attention_map.shape[0] != soft_consensus_mask.shape[0]:
        raise ValueError(
            "attention_map and soft_consensus_mask batch sizes must match"
        )

    if torch.any(attention_map < 0.0):
        raise ValueError("attention_map cannot contain negative values")

    attention = attention_map.flatten(1)
    attention_mass = attention.sum(dim=1, keepdim=True)

    if torch.any(attention_mass <= 0.0):
        raise ValueError("attention_map must contain positive spatial mass")

    # Re-normalize defensively. The native DSA already sums to one, but this
    # makes the loss contract explicit and robust to small numerical drift.
    attention = attention / attention_mass

    target = _normalized_spatial_target(
        soft_consensus_mask.to(
            device=attention_map.device,
            dtype=attention_map.dtype,
        ),
        spatial_size=(
            attention_map.shape[2],
            attention_map.shape[3],
        ),
    )

    target_safe = target.clamp_min(_EPS)
    attention_safe = attention.clamp_min(_EPS)

    return (
        target
        * (
            torch.log(target_safe)
            - torch.log(attention_safe)
        )
    ).sum(dim=1).mean()


def total_consensus_loss(
    *,
    logits: torch.Tensor,
    hard_target: torch.Tensor,
    soft_target: torch.Tensor,
    disagreement: torch.Tensor,
    ordinal_logits: torch.Tensor,
    ordinal_target: torch.Tensor,
    attention_map: torch.Tensor,
    soft_consensus_mask: torch.Tensor,
    lambda_ordinal: float,
    lambda_attention: float,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compose Phase 4 losses while keeping each auxiliary term toggleable."""

    lambda_ordinal = float(lambda_ordinal)
    lambda_attention = float(lambda_attention)

    if lambda_ordinal < 0.0:
        raise ValueError("lambda_ordinal must be non-negative")
    if lambda_attention < 0.0:
        raise ValueError("lambda_attention must be non-negative")

    diagnostic = diagnostic_disagreement_loss(
        logits,
        hard_target,
        soft_target,
        disagreement,
        class_weights=class_weights,
    )

    ordinal = ordinal_distribution_loss(
        ordinal_logits,
        ordinal_target,
    )

    attention = attention_alignment_loss(
        attention_map,
        soft_consensus_mask,
    )

    total = (
        diagnostic
        + lambda_ordinal * ordinal
        + lambda_attention * attention
    )

    if not torch.isfinite(total):
        raise ValueError("total consensus loss is non-finite")

    return {
        "total": total,
        "diagnostic": diagnostic,
        "ordinal": ordinal,
        "attention": attention,
    }
