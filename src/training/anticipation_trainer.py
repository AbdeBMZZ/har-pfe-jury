"""
Training and evaluation loop for the activity anticipation head (Module 2).

The backbone is FROZEN during anticipation training — we only train
the LSTM head on top of the pre-trained backbone embeddings.
This keeps the shared representation intact for continual learning.

Improvements vs baseline:
  - inverse-frequency class weights in CrossEntropy
  - restore best anticipation_head weights by val macro-F1
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

from ..models.har_model import HARContinualModel
from ..evaluation.metrics import macro_f1


def train_anticipation(
    model:       HARContinualModel,
    datasets:    dict,          # {obs_ratio: (train_ds, val_ds)}
    n_epochs:    int   = 50,
    batch_size:  int   = 64,
    lr:          float = 5e-4,
    device:      str   = "cpu",
    verbose:     bool  = True,
    use_class_weights: bool = True,
    n_classes:   Optional[int] = None,
) -> Dict[float, Dict]:
    """
    Train the anticipation head for each observation ratio.

    Backbone weights are FROZEN. Only anticipation_head parameters
    are updated. Best head (by val macro-F1) is restored at the end
    of each ratio run.
    """
    device = torch.device(device)
    model  = model.to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.har_head.parameters():
        p.requires_grad = False
    for p in model.anticipation_head.parameters():
        p.requires_grad = True

    results = {}
    best_overall_f1 = -1.0
    best_overall_state = None
    best_overall_ratio = None

    for obs_ratio, (train_ds, val_ds) in datasets.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Observation ratio p={obs_ratio:.2f} "
                  f"({int(obs_ratio*100)}% of window)")
            print(train_ds.summary())

        if len(train_ds) == 0:
            print("  No samples — skipping.")
            continue

        majority_f1 = val_ds.majority_baseline()
        _reinit_head(model.anticipation_head)

        weight = None
        if use_class_weights and len(train_ds) > 0:
            weight = train_ds.class_weights(n_classes=n_classes).to(device)
            if verbose:
                present = (weight > 0).sum().item()
                print(f"  Class weights: {present} classes, "
                      f"min={weight[weight > 0].min():.3f} "
                      f"max={weight.max():.3f}")

        optimizer = AdamW(model.anticipation_head.parameters(),
                          lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs,
                                       eta_min=lr * 0.01)

        train_loader = train_ds.dataloader(batch_size=batch_size, shuffle=True)
        val_loader   = val_ds.dataloader(batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
        best_f1 = -1.0
        best_acc = 0.0
        best_state = None

        for epoch in range(1, n_epochs + 1):
            model.train()
            model.backbone.eval()
            model.har_head.eval()

            epoch_losses = []
            for X_seq, y_next in train_loader:
                X_seq  = X_seq.to(device)
                y_next = y_next.to(device)

                optimizer.zero_grad()
                logits = model.anticipate(X_seq)
                loss   = F.cross_entropy(logits, y_next, weight=weight)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.anticipation_head.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()

            val_loss, val_f1, val_acc = _eval_anticipation(
                model, val_loader, device)

            history["train_loss"].append(float(np.mean(epoch_losses)))
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_acc"].append(val_acc)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                best_state = copy.deepcopy(
                    model.anticipation_head.state_dict())

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  epoch {epoch:3d}/{n_epochs} | "
                      f"train_loss={history['train_loss'][-1]:.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"val_F1={val_f1:.4f} | "
                      f"val_acc={val_acc:.4f}")

        if best_state is not None:
            model.anticipation_head.load_state_dict(best_state)

        results[obs_ratio] = {
            "history": history,
            "val_f1":  best_f1,
            "val_acc": best_acc,
            "majority_baseline": majority_f1,
        }
        if verbose:
            print(f"  Best val F1={best_f1:.4f}  |  Best val acc={best_acc:.4f}"
                  f"  |  Majority baseline F1={majority_f1:.4f}")

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall_ratio = obs_ratio
            best_overall_state = copy.deepcopy(
                model.anticipation_head.state_dict())

    # Restore the globally best head for the saved checkpoint
    if best_overall_state is not None:
        model.anticipation_head.load_state_dict(best_overall_state)
        if verbose:
            print(f"\nRestored best head (p={best_overall_ratio}, "
                  f"F1={best_overall_f1:.4f}) for checkpoint.")

    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.har_head.parameters():
        p.requires_grad = True

    return results


def _eval_anticipation(model, loader, device):
    model.eval()
    all_preds, all_labels, losses = [], [], []

    with torch.no_grad():
        for X_seq, y_next in loader:
            X_seq  = X_seq.to(device)
            y_next = y_next.to(device)
            logits = model.anticipate(X_seq)
            losses.append(F.cross_entropy(logits, y_next).item())
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_next.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc    = float((y_pred == y_true).mean())
    f1     = macro_f1(y_true, y_pred)
    return float(np.mean(losses)), f1, acc


def _reinit_head(head: nn.Module):
    """Re-initialise LSTM and linear weights for a fresh training run."""
    for m in head.modules():
        if isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def print_anticipation_summary(results: Dict[float, Dict]):
    """Print a clean comparison table across observation ratios."""
    print("\n=== Anticipation Head Results ===")
    print(f"  {'Obs ratio':>10}  {'Val F1':>8}  {'Val Acc':>8}  {'Maj. F1':>8}")
    print(f"  {'-'*42}")
    for ratio in sorted(results):
        r = results[ratio]
        maj = r.get("majority_baseline", float("nan"))
        print(f"  {ratio*100:>8.0f}%  {r['val_f1']:>8.4f}  {r['val_acc']:>8.4f}"
              f"  {maj:>8.4f}")
    print()
