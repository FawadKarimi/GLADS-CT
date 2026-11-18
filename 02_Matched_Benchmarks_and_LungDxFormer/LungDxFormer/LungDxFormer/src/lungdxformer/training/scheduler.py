from __future__ import annotations
import torch

def build_scheduler(optimizer, config: dict):
    if config.get("scheduler", "reduce_on_plateau") == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("scheduler_factor", 0.5),
            patience=config.get("scheduler_patience", 3),
        )
    return None
