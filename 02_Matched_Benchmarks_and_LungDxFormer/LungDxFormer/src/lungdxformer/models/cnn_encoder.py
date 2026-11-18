from __future__ import annotations
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)

class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, channels=(32, 64, 128)):
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers.append(ConvBlock(prev, ch))
            prev = ch
        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
