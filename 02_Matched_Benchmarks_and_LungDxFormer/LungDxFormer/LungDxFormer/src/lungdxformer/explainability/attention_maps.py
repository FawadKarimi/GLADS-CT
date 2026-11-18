from __future__ import annotations
import torch
import torch.nn.functional as F

def upscale_attention_map(attn_map: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """
    attn_map: [B,1,H,W]
    """
    return F.interpolate(attn_map, size=target_hw, mode="bilinear", align_corners=False)
