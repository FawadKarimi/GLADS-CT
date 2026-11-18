"""ConsenDxFormer Phase 4 model.

The model preserves the validated 2D LungDxFormer backbone scale and
Transformer capacity while adding:
- explicit C1/C2/C3 feature outputs,
- native spatial-softmax attention,
- hierarchical coarse-to-fine gated residual fusion,
- a primary 3-class head,
- a four-threshold ordinal auxiliary head.

No code in the locked external LungDxFormer checkout is modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStage(nn.Module):
    """Two convolutions followed by 2x downsampling, matching the baseline style."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleCNNEncoder(nn.Module):
    """Return all three CNN scales required for hierarchical fusion."""

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, int, int],
    ) -> None:
        super().__init__()

        if len(channels) != 3:
            raise ValueError("cnn_channels must contain exactly three stages")

        c1, c2, c3 = channels

        self.stage1 = ConvStage(in_channels, c1)
        self.stage2 = ConvStage(c1, c2)
        self.stage3 = ConvStage(c2, c3)

        self.channels = channels

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        return c1, c2, c3


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_tokens: int, embed_dim: int) -> None:
        super().__init__()
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.position = nn.Parameter(
            torch.zeros(1, max_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, N, C]")
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError(
                "Input contains more tokens than positional encoding supports"
            )
        return tokens + self.position[:, : tokens.shape[1]]


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.norm1(x)

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        x = residual + self.dropout1(attended)
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("transformer_layers must be positive")

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DynamicSpatialAttention(nn.Module):
    """Native LungDxFormer-style spatial-softmax attention."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            embed_dim,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.projection(feature_map)

        batch, _, height, width = logits.shape

        attention = F.softmax(
            logits.reshape(batch, 1, height * width),
            dim=-1,
        ).reshape(
            batch,
            1,
            height,
            width,
        )

        weighted = feature_map * attention

        return weighted, attention


class DeepFusion(nn.Module):
    """Baseline-like deepest-scale CNN/attention fusion."""

    def __init__(
        self,
        cnn_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        self.cnn_projection = nn.Conv2d(
            cnn_channels,
            embed_dim,
            kernel_size=1,
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                embed_dim * 2,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        c3: torch.Tensor,
        attention_features: torch.Tensor,
    ) -> torch.Tensor:
        projected_c3 = self.cnn_projection(c3)
        return self.fusion(
            torch.cat(
                [projected_c3, attention_features],
                dim=1,
            )
        )


class GatedResidualFusion(nn.Module):
    """Inject a finer CNN scale into an upsampled deeper representation."""

    def __init__(
        self,
        lateral_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        self.lateral_projection = nn.Sequential(
            nn.Conv2d(
                lateral_channels,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                embed_dim * 2,
                embed_dim,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        deep_feature: torch.Tensor,
        lateral_feature: torch.Tensor,
    ) -> torch.Tensor:
        upsampled = F.interpolate(
            deep_feature,
            size=lateral_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        lateral = self.lateral_projection(
            lateral_feature
        )

        gate = self.gate(
            torch.cat(
                [upsampled, lateral],
                dim=1,
            )
        )

        # Residual preservation of the deeper semantic representation,
        # with gated injection of the finer spatial feature.
        fused = upsampled + gate * lateral

        return self.refine(fused)


class ConsenDxFormer(nn.Module):
    """Disagreement-aware 2D CNN-Transformer for pulmonary nodules."""

    def __init__(
        self,
        *,
        image_size: int = 64,
        in_channels: int = 1,
        num_classes: int = 3,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        embed_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_mlp_ratio: float = 4.0,
        dropout: float = 0.5,
        classifier_hidden_dim: int = 128,
        use_positional_encoding: bool = True,
        use_spatial_attention: bool = True,
        use_hierarchical_fusion: bool = True,
    ) -> None:
        super().__init__()

        if image_size <= 0 or image_size % 8 != 0:
            raise ValueError(
                "image_size must be positive and divisible by 8"
            )

        if num_classes != 3:
            raise ValueError(
                "ConsenDxFormer primary head is locked to three classes"
            )

        if len(cnn_channels) != 3:
            raise ValueError(
                "cnn_channels must contain exactly three values"
            )

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        if embed_dim % transformer_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by transformer_heads"
            )

        self.image_size = int(image_size)
        self.embed_dim = int(embed_dim)

        self.use_positional_encoding = bool(
            use_positional_encoding
        )
        self.use_spatial_attention = bool(
            use_spatial_attention
        )
        self.use_hierarchical_fusion = bool(
            use_hierarchical_fusion
        )

        self.encoder = MultiScaleCNNEncoder(
            in_channels=in_channels,
            channels=cnn_channels,
        )

        deepest_channels = cnn_channels[-1]

        self.token_projection = nn.Conv2d(
            deepest_channels,
            embed_dim,
            kernel_size=1,
        )

        deep_size = image_size // 8
        number_of_tokens = deep_size * deep_size

        self.positional_encoding = LearnablePositionalEncoding(
            max_tokens=number_of_tokens,
            embed_dim=embed_dim,
        )

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            mlp_ratio=transformer_mlp_ratio,
            dropout=dropout,
        )

        self.dynamic_attention = DynamicSpatialAttention(
            embed_dim=embed_dim,
        )

        self.deep_fusion = DeepFusion(
            cnn_channels=deepest_channels,
            embed_dim=embed_dim,
        )

        self.c2_fusion = GatedResidualFusion(
            lateral_channels=cnn_channels[1],
            embed_dim=embed_dim,
        )

        self.c1_fusion = GatedResidualFusion(
            lateral_channels=cnn_channels[0],
            embed_dim=embed_dim,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(
                embed_dim,
                classifier_hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                classifier_hidden_dim,
                num_classes,
            ),
        )

        self.ordinal_head = nn.Sequential(
            nn.Linear(
                embed_dim,
                classifier_hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                classifier_hidden_dim,
                4,
            ),
        )

    @staticmethod
    def _to_tokens(
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        batch, channels, height, width = feature_map.shape

        tokens = feature_map.flatten(2).transpose(1, 2)

        return tokens, (height, width)

    @staticmethod
    def _to_feature_map(
        tokens: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, token_count, channels = tokens.shape
        height, width = spatial_size

        if token_count != height * width:
            raise ValueError(
                "Token count disagrees with requested spatial shape"
            )

        return (
            tokens.transpose(1, 2)
            .reshape(
                batch,
                channels,
                height,
                width,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                "Input must have shape [B, C, H, W]"
            )

        c1, c2, c3 = self.encoder(x)

        projected_c3 = self.token_projection(c3)

        tokens, spatial_size = self._to_tokens(
            projected_c3
        )

        if self.use_positional_encoding:
            tokens = self.positional_encoding(tokens)

        transformed_tokens = self.transformer(tokens)

        transformer_features = self._to_feature_map(
            transformed_tokens,
            spatial_size,
        )

        if self.use_spatial_attention:
            attention_features, attention_map = (
                self.dynamic_attention(
                    transformer_features
                )
            )
        else:
            batch, _, height, width = (
                transformer_features.shape
            )

            attention_map = torch.full(
                (
                    batch,
                    1,
                    height,
                    width,
                ),
                fill_value=1.0 / float(height * width),
                device=transformer_features.device,
                dtype=transformer_features.dtype,
            )

            attention_features = (
                transformer_features * attention_map
            )

        deep_fused = self.deep_fusion(
            c3,
            attention_features,
        )

        if self.use_hierarchical_fusion:
            fused_c2 = self.c2_fusion(
                deep_fused,
                c2,
            )
            fused = self.c1_fusion(
                fused_c2,
                c1,
            )
        else:
            fused = deep_fused

        pooled = self.pool(fused).flatten(1)

        logits = self.classifier(pooled)
        ordinal_logits = self.ordinal_head(pooled)

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        return {
            "logits": logits,
            "probs": probabilities,
            "ordinal_logits": ordinal_logits,
            "attention_map": attention_map,
            "c1_features": c1,
            "c2_features": c2,
            "c3_features": c3,
            "transformer_features": transformer_features,
            "attention_features": attention_features,
            "deep_fused_features": deep_fused,
            "fused_features": fused,
            "pooled_features": pooled,
        }
