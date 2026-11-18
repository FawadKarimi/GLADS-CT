from __future__ import annotations
import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

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
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + self.drop1(x)
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x)
        return x
