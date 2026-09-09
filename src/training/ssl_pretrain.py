"""
Light self-supervised pre-training (SimCLR-style) for IMU windows.

Two augmented views of the same window → NT-Xent on backbone embeddings.
Labels are not used. This is a pragmatic SSL stage, not a foundation model.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


def imu_augment(x: torch.Tensor,
                noise_std: float = 0.05,
                scale_min: float = 0.9,
                scale_max: float = 1.1,
                mask_ratio: float = 0.1) -> torch.Tensor:
    """x: (B, T, C) → augmented copy."""
    y = x.clone()
    y = y + noise_std * torch.randn_like(y)
    scale = torch.empty(y.size(0), 1, y.size(-1), device=y.device).uniform_(
        scale_min, scale_max)
    y = y * scale
    if mask_ratio > 0:
        T = y.size(1)
        n_mask = max(1, int(T * mask_ratio))
        for i in range(y.size(0)):
            start = int(torch.randint(0, max(1, T - n_mask + 1), (1,)).item())
            y[i, start:start + n_mask] = 0.0
    return y


class ProjectionHead(nn.Module):
    def __init__(self, d_model: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent for two batches of size B (paired positives)."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = z @ z.T / temperature
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    # positives: i ↔ i+B
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)
    return F.cross_entropy(sim, targets)


def ssl_pretrain_backbone(
    backbone: nn.Module,
    X: np.ndarray,
    n_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    temperature: float = 0.2,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[nn.Module, list]:
    """
    Self-supervised pretrain of `backbone` only (in-place).

    Returns backbone and list of epoch losses.
    """
    device_t = torch.device(device)
    backbone = backbone.to(device_t)
    proj = ProjectionHead(d_model=backbone.d_model).to(device_t)

    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = AdamW(list(backbone.parameters()) + list(proj.parameters()),
                lr=lr, weight_decay=1e-4)
    history = []

    for epoch in range(1, n_epochs + 1):
        backbone.train()
        proj.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device_t)
            v1, v2 = imu_augment(xb), imu_augment(xb)
            z1 = proj(backbone(v1))
            z2 = proj(backbone(v2))
            loss = nt_xent(z1, z2, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(mean_loss)
        if verbose:
            print(f"  [SSL] epoch {epoch:3d}/{n_epochs}  loss={mean_loss:.4f}")

    return backbone, history
