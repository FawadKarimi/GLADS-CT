from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lungdxformer.models.cnn_encoder import CNNEncoder
from lungdxformer.models.positional_encoding import LearnablePositionalEncoding
from lungdxformer.models.transformer_encoder import TransformerEncoder
from lungdxformer.models.spatial_attention import DynamicSpatialAttention
from lungdxformer.models.fusion import FeatureFusion
from lungdxformer.models.classifier_head import ClassificationHead

class LungDxFormer(nn.Module):
    def __init__(
        self,
        image_size: int = 96,
        in_channels: int = 1,
        num_classes: int = 3,
        cnn_channels=(32, 64, 128),
        embed_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        use_transformer: bool = True,
        use_positional_encoding: bool = True,
        use_spatial_attention: bool = True,
        fusion_type: str = "concat",
        use_raw_transformer_in_fusion: bool = False,
        classifier_hidden_dim: int = 128,
    ):
        super().__init__()
        self.image_size = image_size
        self.use_transformer = use_transformer
        self.use_positional_encoding = use_positional_encoding
        self.use_spatial_attention = use_spatial_attention

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
        self.fusion = FeatureFusion(
            cnn_channels=embed_dim,
            trans_channels=embed_dim,
            fusion_type=fusion_type,
            use_raw_transformer=use_raw_transformer_in_fusion,
        )
        self.classifier = ClassificationHead(
            in_channels=self.fusion.out_channels,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self._last_feature_map = None

    def _to_tokens(self, x_2d: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        b, c, h, w = x_2d.shape
        tokens = x_2d.flatten(2).transpose(1, 2)  # [B, N, C]
        return tokens, (h, w)

    def _to_2d(self, tokens: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        b, n, c = tokens.shape
        h, w = spatial_hw
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cnn_local = self.cnn_encoder(x)
        cnn_proj = self.token_proj(cnn_local)
        tokens, spatial_hw = self._to_tokens(cnn_proj)

        if self.use_positional_encoding:
            tokens = self.positional_encoding(tokens)

        if self.use_transformer:
            trans_tokens = self.transformer(tokens)
        else:
            trans_tokens = tokens

        trans_2d = self._to_2d(trans_tokens, spatial_hw)

        if self.use_spatial_attention:
            attn_feat, attn_map = self.dynamic_attention(trans_2d)
        else:
            attn_feat = trans_2d
            attn_map = torch.ones(trans_2d.shape[0], 1, trans_2d.shape[2], trans_2d.shape[3], device=trans_2d.device)

        fused = self.fusion(cnn_proj, attn_feat, trans_2d)
        logits, pooled = self.classifier(fused)
        probs = F.softmax(logits, dim=1)

        self._last_feature_map = fused

        return {
            "logits": logits,
            "probs": probs,
            "attention_map": attn_map,
            "cnn_features": cnn_proj,
            "transformer_features": trans_2d,
            "attention_features": attn_feat,
            "fused_features": fused,
            "pooled_features": pooled,
        }

    @property
    def last_feature_map(self):
        return self._last_feature_map
