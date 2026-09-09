"""
Online adaptive calibration for streaming HAR predictions.

- OnlineTemperatureScaler: learns a scalar T so that softmax(logits/T)
  is better calibrated (NLL on a sliding labeled buffer).
- AdaptiveConfidenceThreshold: adjusts the accept/reject threshold online.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTemperatureScaler(nn.Module):
    """Single-parameter temperature scaling, updated online."""

    def __init__(self, init_temperature: float = 1.5, lr: float = 0.05):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(np.log(init_temperature))))
        self.opt = torch.optim.SGD([self.log_t], lr=lr)

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp().clamp(min=1e-3)

    @torch.no_grad()
    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(logits), dim=-1)

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """One gradient step on NLL. Returns loss value."""
        self.train()
        device = self.log_t.device
        # Re-enable grad path through temperature; logits are constants
        logits = logits.detach().to(device).requires_grad_(False)
        y = y.to(device)
        self.opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.forward(logits), y)
        loss.backward()
        self.opt.step()
        return float(loss.detach().item())


class AdaptiveConfidenceThreshold:
    """
    Online threshold on max softmax probability.

    If confidence < threshold → abstain / flag uncertain.
    Threshold drifts with EMA of recent accuracies.
    """

    def __init__(self,
                 init_threshold: float = 0.55,
                 target_acc: float = 0.85,
                 ema: float = 0.9,
                 min_t: float = 0.3,
                 max_t: float = 0.95):
        self.threshold = init_threshold
        self.target_acc = target_acc
        self.ema = ema
        self.min_t = min_t
        self.max_t = max_t
        self._acc_ema = target_acc

    def accept(self, confidence: float) -> bool:
        return confidence >= self.threshold

    def update(self, correct: bool, confidence: float) -> None:
        """Update running accuracy; raise threshold if too wrong when confident."""
        self._acc_ema = self.ema * self._acc_ema + (1 - self.ema) * float(correct)
        if confidence >= self.threshold:
            # only adjust based on decisions we actually accepted
            if self._acc_ema < self.target_acc:
                self.threshold = min(self.max_t, self.threshold + 0.01)
            else:
                self.threshold = max(self.min_t, self.threshold - 0.005)


class OnlineCalibrator:
    """Combines temperature scaling + adaptive threshold for a stream."""

    def __init__(self, n_classes: int, buffer_size: int = 256):
        self.temp = OnlineTemperatureScaler()
        self.threshold = AdaptiveConfidenceThreshold()
        self.buffer_logits: Deque[torch.Tensor] = deque(maxlen=buffer_size)
        self.buffer_y: Deque[int] = deque(maxlen=buffer_size)
        self.n_classes = n_classes

    def predict(self, logits: torch.Tensor) -> Tuple[int, float, bool]:
        """
        Returns (pred_class, confidence, accepted).
        logits: (C,) or (1, C)
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = self.temp.calibrate_probs(logits.detach())[0]
        conf, pred = float(probs.max().item()), int(probs.argmax().item())
        ok = self.threshold.accept(conf)
        return pred, conf, ok

    def feedback(self, logits: torch.Tensor, true_label: int) -> None:
        """Call when a ground-truth label becomes available (online)."""
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        y = torch.tensor([true_label], dtype=torch.long)
        # Assess the prediction before using its ground truth for calibration.
        pred, conf, accepted = self.predict(logits)
        self.temp.update(logits.detach(), y)
        if accepted:
            self.threshold.update(pred == true_label, conf)
        self.buffer_logits.append(logits.detach().cpu().squeeze(0))
        self.buffer_y.append(true_label)
