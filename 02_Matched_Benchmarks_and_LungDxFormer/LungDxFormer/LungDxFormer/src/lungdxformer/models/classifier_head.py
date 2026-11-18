from __future__ import annotations
import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(x)
        logits = self.fc(pooled)
        return logits, pooled.flatten(1)
