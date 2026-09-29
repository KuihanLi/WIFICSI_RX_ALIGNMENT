"""Public B2_MEAN frozen model snapshot.

This module preserves the algorithmic semantics of the B2_MEAN condition used in
WIFICSI_RX_ALIGNMENT: a shared ViewEncoder is applied independently to three
synchronized receiver views, the three 64-D view latents are averaged equally,
and a dropout + linear classifier predicts the task label.

Expected input shape: [batch, 3, channels, time_bins, subcarrier_bins].
The frozen paper configuration uses channels=2 (amplitude + observation mask),
time_bins=128, subcarrier_bins=64, latent_dim=64, dropout=0.15.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ViewEncoder(nn.Module):
    """Exact encoder architecture used by the frozen B2_MEAN condition."""

    def __init__(self, in_channels: int = 2, latent_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class B2Mean(nn.Module):
    """Frozen B2_MEAN: shared encoder -> equal receiver mean -> linear head."""

    def __init__(
        self,
        n_classes: int,
        *,
        in_channels: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.15,
        n_receivers: int = 3,
    ):
        super().__init__()
        self.n_receivers = int(n_receivers)
        self.encoder = ViewEncoder(in_channels=in_channels, latent_dim=latent_dim, dropout=dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(latent_dim, n_classes))

    def encode_views(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"expected [B,V,C,T,S], got {tuple(x.shape)}")
        batch, views, channels, time_bins, subcarrier_bins = x.shape
        if views != self.n_receivers:
            raise ValueError(f"expected {self.n_receivers} receiver views, got {views}")
        z = self.encoder(x.reshape(batch * views, channels, time_bins, subcarrier_bins)).reshape(batch, views, -1)
        fused = z.mean(dim=1)
        weights = torch.full((batch, views), 1.0 / views, dtype=z.dtype, device=z.device)
        return {
            "z": z,
            "shared_pre": z,
            "shared": z,
            "u": z,
            "gate_weights": weights,
            "fused": fused,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encode_views(x)
        out["logits"] = self.classifier(out["fused"])
        return out
