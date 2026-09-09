"""
Postural-transition prediction prototype (legacy API names retained).

Labels 9–12 are ordinary posture transitions. They are NOT ground-truth fall
risk, instability or loss of balance. Scores trained on these targets must not
be reported as fall-prevention performance. PhysicalFallDetector is only an
unvalidated heuristic requiring acceleration in g, not standardized windows.
"""

from __future__ import annotations

from typing import Set

import numpy as np
import torch
import torch.nn as nn


FALL_RISK_LABELS: Set[int] = {9, 10, 11, 12}


class FallRiskHead(nn.Module):
    """Binary head for the chosen posture-transition proxy; not a fall-risk probability."""

    def __init__(self, d_model: int = 128, lstm_hidden: int = 64,
                 lstm_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, embedding_seq: torch.Tensor) -> torch.Tensor:
        """
        embedding_seq: (B, S, d_model)
        returns: (B,) logits (positive = fall-risk)
        """
        out, _ = self.lstm(embedding_seq)
        return self.classifier(out[:, -1]).squeeze(-1)


class PhysicalFallDetector:
    """
    Non-learned heuristic: high acceleration spike then low-motion.
    For demo / safety layer alongside the learned fall-risk head.
    """

    def __init__(self,
                 spike_g: float = 2.5,
                 stillness: float = 0.15,
                 still_frames: int = 10):
        self.spike_g = spike_g
        self.stillness = stillness
        self.still_frames = still_frames

    def detect(self, window: np.ndarray) -> bool:
        """
        window: (T, C) with C>=3 accel in first 3 channels (approx in g).
        """
        acc = window[:, :3]
        mag = np.linalg.norm(acc, axis=-1)
        if mag.max() < self.spike_g:
            return False
        peak = int(mag.argmax())
        after = mag[peak + 1: peak + 1 + self.still_frames]
        if len(after) < self.still_frames:
            return False
        return float(after.mean()) < self.stillness


def labels_to_fall_risk(y: np.ndarray) -> np.ndarray:
    """Map activity labels → binary fall-risk targets."""
    return np.isin(y, list(FALL_RISK_LABELS)).astype(np.float32)
