from __future__ import annotations
import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, cnn_channels: int, trans_channels: int, fusion_type: str = "concat", use_raw_transformer: bool = False):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_raw_transformer = use_raw_transformer

        if fusion_type not in {"concat", "sum", "gated"}:
            raise ValueError("fusion_type must be one of: concat, sum, gated")

        if fusion_type == "concat":
            in_ch = cnn_channels + trans_channels + (trans_channels if use_raw_transformer else 0)
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, trans_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(trans_channels),
                nn.ReLU(inplace=True),
            )
            self.out_channels = trans_channels

        elif fusion_type == "sum":
            self.align_cnn = nn.Conv2d(cnn_channels, trans_channels, kernel_size=1)
            self.out_channels = trans_channels

        else:  # gated
            self.align_cnn = nn.Conv2d(cnn_channels, trans_channels, kernel_size=1)
            gate_in = trans_channels * (2 + int(use_raw_transformer))
            self.gate = nn.Sequential(
                nn.Conv2d(gate_in, trans_channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.out_channels = trans_channels

    def forward(self, cnn_feat: torch.Tensor, attn_feat: torch.Tensor, raw_transformer_feat: torch.Tensor | None = None) -> torch.Tensor:
        if self.fusion_type == "concat":
            feats = [cnn_feat, attn_feat]
            if self.use_raw_transformer and raw_transformer_feat is not None:
                feats.append(raw_transformer_feat)
            x = torch.cat(feats, dim=1)
            return self.proj(x)

        if self.fusion_type == "sum":
            return self.align_cnn(cnn_feat) + attn_feat

        cnn_aligned = self.align_cnn(cnn_feat)
        parts = [cnn_aligned, attn_feat]
        if self.use_raw_transformer and raw_transformer_feat is not None:
            parts.append(raw_transformer_feat)
        cat = torch.cat(parts, dim=1)
        gate = self.gate(cat)
        return gate * cnn_aligned + (1 - gate) * attn_feat
