from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSpatialAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_2d: [B, C, H, W]
        returns:
          weighted_features: [B, C, H, W]
          attention_map: [B, 1, H, W]
        """
        logits = self.conv(x_2d)
        b, _, h, w = logits.shape
        attn = F.softmax(logits.view(b, 1, h*w), dim=-1).view(b, 1, h, w)
        weighted = x_2d * attn
        return weighted, attn
