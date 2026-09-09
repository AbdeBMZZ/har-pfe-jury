"""
Lightweight domain adaptation for cross-dataset HAR.

A small affine transform on backbone embeddings, fine-tuned with a few
labeled examples from the target domain.  Much cheaper than retraining
the full backbone and directly addresses sensor-placement shift (e.g.
HAPT pocket → PAMAP2 chest).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdapter(nn.Module):
    """Feature-wise affine calibration: x' = x * scale + bias."""

    def __init__(self, d_model: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias  = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


def adapt_domain(
    backbone: nn.Module,
    prototype_memory,
    adapter: DomainAdapter,
    X_adapt: torch.Tensor,
    y_adapt: torch.Tensor,
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-2,
    device: torch.device | None = None,
) -> DomainAdapter:
    """
    Few-shot adaptation: train only the DomainAdapter (backbone frozen).

    Trains against the *same* nearest-prototype objective used at eval time
    (mode="prototype" in eval_with_adapter): cross-entropy over negative
    squared distances to the frozen source-domain prototypes. Training
    against the linear softmax head instead would optimize a different
    decision rule than the one actually used for prediction, and the
    adapter would not transfer.

    Args:
        prototype_memory: frozen source-domain PrototypeMemory (not updated
            during adaptation — it defines the fixed target class geometry).
    """
    if device is None:
        device = next(backbone.parameters()).device

    adapter = adapter.to(device)
    X_adapt = X_adapt.to(device)
    y_adapt = y_adapt.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    proto_matrix, cls_ids = prototype_memory._proto_matrix(device)
    cls_id_to_idx = {c: i for i, c in enumerate(cls_ids)}
    target_idx = torch.tensor(
        [cls_id_to_idx[int(c)] for c in y_adapt.tolist()],
        dtype=torch.long, device=device,
    )

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

    n = len(X_adapt)
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx    = perm[start: start + batch_size]
            x_b    = X_adapt[idx]
            t_b    = target_idx[idx]
            optimizer.zero_grad()
            with torch.no_grad():
                emb = backbone(x_b)
            adapted = adapter(emb)
            dists   = (adapted.unsqueeze(1) - proto_matrix.unsqueeze(0)).pow(2).sum(-1)
            loss    = F.cross_entropy(-dists, t_b)
            loss.backward()
            optimizer.step()

    for p in backbone.parameters():
        p.requires_grad = True

    return adapter


def eval_with_adapter(
    backbone: nn.Module,
    har_head: nn.Module,
    adapter: DomainAdapter | None,
    prototype_memory,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    mode: str = "prototype",
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Evaluate using optional domain adapter + prototype or softmax."""
    backbone.eval()
    if adapter is not None:
        adapter.eval()

    if len(X) == 0:
        return 0.0, np.array([]), np.array([])

    preds, labels = [], []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_b = X[i:i + batch_size].to(device)
            emb = backbone(x_b)
            if adapter is not None:
                emb = adapter(emb)
            if mode == "prototype":
                if prototype_memory.n_classes() == 0:
                    return 0.0, torch.tensor([]), torch.tensor([])
                p = prototype_memory.predict(emb)
            else:
                p = har_head(emb).argmax(dim=-1)
            preds.append(p.cpu())
            labels.append(y[i:i + batch_size])

    from ..evaluation.metrics import macro_f1
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    return macro_f1(y_true, y_pred), y_pred, y_true
