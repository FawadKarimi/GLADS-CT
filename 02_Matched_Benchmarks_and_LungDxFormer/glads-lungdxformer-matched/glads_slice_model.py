
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from lungdxformer.models.cnn_encoder import CNNEncoder
from lungdxformer.models.positional_encoding import LearnablePositionalEncoding
from lungdxformer.models.transformer_encoder import TransformerEncoder
from lungdxformer.models.spatial_attention import DynamicSpatialAttention
from lungdxformer.models.classifier_head import ClassificationHead

class GatedFusionBlock(nn.Module):
    """
    GLADS-CT Coarse-to-Fine Lateral Gate:
    Dynamically arbitrates local CNN spatial features and global Transformer contextual features.
    """
    def __init__(self, cnn_channels: int, trans_channels: int, out_channels: int = 128):
        super().__init__()
        # Align projections if needed
        self.align_cnn = nn.Sequential(
            nn.Conv2d(cnn_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.align_trans = nn.Sequential(
            nn.Conv2d(trans_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Lateral Gate: Sigmoid arbitration map in [0, 1]
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # Refinement convolution
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_channels

    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_cnn = self.align_cnn(cnn_feat)
        f_trans = self.align_trans(trans_feat)
        # Compute spatial gate
        cat_feat = torch.cat([f_cnn, f_trans], dim=1)
        g = self.gate(cat_feat)
        # Gated fusion
        f_gated = g * f_cnn + (1.0 - g) * f_trans
        f_out = self.refine(f_gated)
        return f_out, g

class GLADS_CT_Slice(nn.Module):
    """
    GLADS-CT Slice-Level Architecture with Lateral Gating and Dual Supervised Heads.
    """
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 1,
        num_classes: int = 3,
        cnn_channels=(32, 64, 128),
        embed_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        classifier_dropout: float = 0.5,
        classifier_hidden_dim: int = 128,
    ):
        super().__init__()
        self.image_size = image_size
        self.cnn_encoder = CNNEncoder(in_channels=in_channels, channels=cnn_channels)
        cnn_out = self.cnn_encoder.out_channels
        self.token_proj = nn.Conv2d(cnn_out, embed_dim, kernel_size=1)

        reduced_h = image_size // (2 ** len(cnn_channels))
        reduced_w = image_size // (2 ** len(cnn_channels))
        self.num_tokens = reduced_h * reduced_w

        self.positional_encoding = LearnablePositionalEncoding(max_tokens=self.num_tokens, embed_dim=embed_dim)
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            mlp_ratio=transformer_mlp_ratio,
            dropout=dropout,
        )
        self.dynamic_attention = DynamicSpatialAttention(embed_dim=embed_dim)
        
        # GLADS-CT Gated Fusion
        self.fusion = GatedFusionBlock(
            cnn_channels=embed_dim,
            trans_channels=embed_dim,
            out_channels=embed_dim
        )
        
        # Primary Diagnostic Head
        self.classifier = ClassificationHead(
            in_channels=self.fusion.out_channels,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

    def _to_tokens(self, x_2d: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        b, c, h, w = x_2d.shape
        tokens = x_2d.flatten(2).transpose(1, 2)
        return tokens, (h, w)

    def _to_2d(self, tokens: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        b, n, c = tokens.shape
        h, w = spatial_hw
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cnn_local = self.cnn_encoder(x)
        cnn_proj = self.token_proj(cnn_local)
        tokens, spatial_hw = self._to_tokens(cnn_proj)

        tokens = self.positional_encoding(tokens)
        trans_tokens = self.transformer(tokens)
        trans_2d = self._to_2d(trans_tokens, spatial_hw)

        attn_feat, attn_map = self.dynamic_attention(trans_2d)
        
        # GLADS-CT Gated Fusion
        fused, gate_map = self.fusion(cnn_proj, attn_feat)
        
        logits, pooled = self.classifier(fused)
        probs = F.softmax(logits, dim=1)

        return {
            "logits": logits,
            "probs": probs,
            "attention_map": attn_map,
            "gate_map": gate_map,
            "cnn_features": cnn_proj,
            "transformer_features": trans_2d,
            "fused_features": fused,
            "pooled_features": pooled,
        }
