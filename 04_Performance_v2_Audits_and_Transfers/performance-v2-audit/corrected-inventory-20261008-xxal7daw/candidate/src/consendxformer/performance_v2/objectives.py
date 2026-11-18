"""Exact scalar reductions, without a data loader or an optimization step."""
import torch
from torch.nn import functional as F

OBJECTIVES = ("HARD", "LEGACY", "SHARED-010", "SEPARATE-010", "CAPACITY-010")


def class_weights_from_training_counts(counts):
    """Counts must come from the frozen TRAIN partition, never validation/test."""
    if len(counts) != 3 or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("three positive training class counts required")
    return torch.tensor([sum(counts) / (3 * n) for n in counts], dtype=torch.float32)


def objective_loss(objective, diagnostic, labels, weights, *, reader_targets=None,
                   disagreement=None, reader_logits=None, capacity_control=False):
    if objective not in OBJECTIVES or (objective == "CAPACITY-010" and not capacity_control):
        raise ValueError("unsupported objective or inactive capacity control")
    if (diagnostic.ndim != 2 or diagnostic.shape[1] != 3 or diagnostic.shape[0] == 0
            or diagnostic.dtype != torch.float32 or not torch.isfinite(diagnostic).all()):
        raise ValueError("finite float32 [B,3] diagnostic logits required")
    batch = diagnostic.shape[0]
    if (labels.shape != (batch,) or labels.dtype != torch.int64
            or labels.device != diagnostic.device or not ((labels >= 0) & (labels < 3)).all()):
        raise ValueError("exact int64 three-class labels required")
    if (weights.shape != (3,) or weights.dtype != torch.float32
            or weights.device != diagnostic.device or weights.requires_grad or not torch.isfinite(weights).all()
            or not (weights > 0).all()):
        raise ValueError("finite positive float32 class weights required")
    needs_reader = objective in ("SEPARATE-010", "CAPACITY-010")
    if needs_reader:
        if (reader_logits is None or reader_logits.shape != diagnostic.shape
                or reader_logits.dtype != diagnostic.dtype or reader_logits.device != diagnostic.device
                or not torch.isfinite(reader_logits).all()):
            raise ValueError("separate reader logits required")
    elif reader_logits is not None:
        raise ValueError("unexpected reader output for this objective")
    if objective in ("LEGACY", "SHARED-010", "SEPARATE-010"):
        q = reader_targets
        if (q is None or q.shape != diagnostic.shape or q.dtype != torch.float32
                or q.device != diagnostic.device or q.requires_grad or not torch.isfinite(q).all()
                or not ((q >= 0) & (q <= 1)).all()
                or not torch.allclose(q.sum(dim=1), torch.ones_like(q[:, 0]), atol=1e-6, rtol=0)):
            raise ValueError("normalized finite reader targets required; no repair")
    log_d = diagnostic.log_softmax(dim=1)
    per_sample = -log_d.gather(1, labels[:, None]).squeeze(1)
    hard_numerator = (weights[labels] * per_sample).sum()
    hard_denominator = weights[labels].sum()
    hard = hard_numerator / hard_denominator
    zero = diagnostic.new_zeros(())
    reader = legacy = zero
    legacy_numerator = legacy_denominator = zero
    if objective == "LEGACY":
        u = disagreement
        if (u is None or u.shape != (batch,) or u.dtype != torch.float32
                or u.device != diagnostic.device or u.requires_grad or not torch.isfinite(u).all()
                or not ((u >= 0) & (u <= 1)).all()):
            raise ValueError("finite disagreement in [0,1] required")
        target = (1-u[:, None]) * F.one_hot(labels, 3) + u[:, None] * q
        mass = target * weights[None, :]
        legacy_numerator = -(mass * log_d).sum()
        legacy_denominator = mass.sum().clamp_min(1e-8)
        legacy = legacy_numerator / legacy_denominator
        total = legacy
    else:
        if objective == "SHARED-010":
            reader = -(q * log_d).sum(dim=1).mean()
        elif objective == "SEPARATE-010":
            reader = -(q * reader_logits.log_softmax(dim=1)).sum(dim=1).mean()
        elif objective == "CAPACITY-010":
            reader = F.cross_entropy(reader_logits, labels, reduction="mean")
        total = hard + 0.1 * reader
    if not torch.isfinite(total):
        raise ValueError("nonfinite objective; no automatic numeric fallback")
    return {"total": total, "hard": hard, "reader": reader, "legacy": legacy,
            "hard_numerator": hard_numerator, "hard_denominator": hard_denominator,
            "legacy_numerator": legacy_numerator, "legacy_denominator": legacy_denominator}
