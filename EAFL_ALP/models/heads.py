"""PersonalHead —— 完整客户端个性化头部实现"""
import torch
import torch.nn as nn
from .fractal import FractalBlock


class PersonalHead(nn.Module):

    def __init__(self, n_cls: int, w: float, depth: int, seed: int):
        super().__init__()
        self.frac = FractalBlock(32, 32, depth, w, seed)

        enc = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.tr   = nn.TransformerEncoder(enc, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cls  = nn.Linear(32, n_cls)

    def forward(self, x):
        z = self.frac(x).unsqueeze(1)     # [B, 1, 32]
        z = self.tr(z)                    # [B, 1, 32]
        z = self.pool(z.transpose(1, 2)).squeeze(-1)  # [B, 32]
        return self.cls(z)
