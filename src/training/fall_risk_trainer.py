"""
Train fall-risk anticipation head (binary) on partial windows.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.fall_risk import FallRiskHead, labels_to_fall_risk
from ..models.har_model import HARContinualModel
from ..evaluation.metrics import macro_f1


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # macro-F1 over {0,1}
    return macro_f1(y_true.astype(np.int64), y_pred.astype(np.int64))


def train_fall_risk(
    model: HARContinualModel,
    datasets: dict,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 5e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[float, Dict]:
    """
    Freeze backbone; train a FallRiskHead for each obs ratio.
    datasets: {ratio: (train_ds, val_ds)} AnticipationDataset with multi-class y
              — we binarize targets with labels_to_fall_risk.
    """
    device_t = torch.device(device)
    model = model.to(device_t)
    for p in model.backbone.parameters():
        p.requires_grad = False

    d_model = model.backbone.d_model
    results = {}
    best_overall = None
    best_f1 = -1.0

    for obs_ratio, (train_ds, val_ds) in datasets.items():
        if len(train_ds) == 0:
            continue
        if verbose:
            yb = labels_to_fall_risk(train_ds.y)
            print(f"\n=== Fall-risk p={obs_ratio} | "
                  f"train={len(train_ds)} pos={int(yb.sum())} ===")

        head = FallRiskHead(d_model=d_model).to(device_t)
        opt = AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

        # class weight for positives
        y_bin = labels_to_fall_risk(train_ds.y)
        pos = max(1.0, float(y_bin.sum()))
        neg = max(1.0, float(len(y_bin) - y_bin.sum()))
        pos_weight = torch.tensor([neg / pos], device=device_t)

        train_loader = train_ds.dataloader(batch_size=batch_size, shuffle=True)
        val_loader = val_ds.dataloader(batch_size=batch_size, shuffle=False)

        best_state = None
        best_ratio_f1 = -1.0
        history = {"val_f1": [], "val_acc": []}

        for epoch in range(1, n_epochs + 1):
            head.train()
            model.backbone.eval()
            for X_seq, y_next in train_loader:
                X_seq = X_seq.to(device_t)
                y_bin_t = torch.tensor(
                    labels_to_fall_risk(y_next.numpy()),
                    device=device_t, dtype=torch.float32,
                )
                B, S, T, C = X_seq.shape
                with torch.no_grad():
                    emb = model.backbone(X_seq.view(B * S, T, C)).view(B, S, -1)
                logits = head(emb)
                loss = F.binary_cross_entropy_with_logits(
                    logits, y_bin_t, pos_weight=pos_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

            # val
            head.eval()
            preds, labels = [], []
            with torch.no_grad():
                for X_seq, y_next in val_loader:
                    X_seq = X_seq.to(device_t)
                    y_bin_np = labels_to_fall_risk(y_next.numpy())
                    B, S, T, C = X_seq.shape
                    emb = model.backbone(X_seq.view(B * S, T, C)).view(B, S, -1)
                    prob = torch.sigmoid(head(emb)).cpu().numpy()
                    preds.append((prob >= 0.5).astype(np.int64))
                    labels.append(y_bin_np.astype(np.int64))
            y_pred = np.concatenate(preds)
            y_true = np.concatenate(labels)
            f1 = _binary_f1(y_true, y_pred)
            acc = float((y_pred == y_true).mean())
            history["val_f1"].append(f1)
            history["val_acc"].append(acc)
            if f1 > best_ratio_f1:
                best_ratio_f1 = f1
                best_state = copy.deepcopy(head.state_dict())
            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  epoch {epoch:3d}/{n_epochs} | "
                      f"val_F1={f1:.4f} | val_acc={acc:.4f}")

        if best_state is not None:
            head.load_state_dict(best_state)
        results[obs_ratio] = {
            "history": history,
            "val_f1": best_ratio_f1,
            "val_acc": max(history["val_acc"]) if history["val_acc"] else 0.0,
            "head_state": head.state_dict(),
        }
        if best_ratio_f1 > best_f1:
            best_f1 = best_ratio_f1
            best_overall = copy.deepcopy(head.state_dict())
        if verbose:
            print(f"  Best fall-risk F1={best_ratio_f1:.4f}")

    if best_overall is not None:
        # stash on model for checkpoint convenience
        model.fall_risk_head = FallRiskHead(d_model=d_model)
        model.fall_risk_head.load_state_dict(best_overall)

    for p in model.backbone.parameters():
        p.requires_grad = True
    return results
