from __future__ import annotations
import torch
import torch.nn as nn

def build_loss(class_weights=None, label_smoothing: float = 0.0, device: str = "cpu"):
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
