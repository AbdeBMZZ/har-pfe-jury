"""
Minimal contextual-bandit RL to adapt an alert confidence threshold online.

Action = discrete threshold. Reward = +1 true alert, -cost false alert,
         +small for correct abstention on negatives.

This is intentionally small (fiche mentions RL as a direction) — not a
full deep-RL environment.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class ThresholdBandit:
    """Epsilon-greedy bandit over a grid of confidence thresholds."""

    def __init__(self,
                 thresholds: Sequence[float] | None = None,
                 epsilon: float = 0.1,
                 false_alert_cost: float = 0.5,
                 seed: int = 0):
        self.thresholds = list(thresholds or [0.4, 0.5, 0.6, 0.7, 0.8])
        self.epsilon = epsilon
        self.false_alert_cost = false_alert_cost
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros(len(self.thresholds), dtype=np.float64)
        self.n = np.zeros(len(self.thresholds), dtype=np.int64)
        self.history: List[dict] = []

    @property
    def current_threshold(self) -> float:
        return float(self.thresholds[int(self.q.argmax())])

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, len(self.thresholds)))
        return int(self.q.argmax())

    def step(self, confidence: float, is_positive: bool) -> dict:
        """
        One online step.
        is_positive: true if event (fall-risk) actually occurred.
        """
        a = self.select_action()
        thr = self.thresholds[a]
        alert = confidence >= thr

        if alert and is_positive:
            reward = 1.0
        elif alert and not is_positive:
            reward = -self.false_alert_cost
        elif (not alert) and is_positive:
            reward = -0.2  # missed alert
        else:
            reward = 0.05  # correct calm

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]
        rec = {"action": a, "threshold": thr, "alert": alert,
               "reward": reward, "confidence": confidence,
               "positive": is_positive}
        self.history.append(rec)
        return rec

    def summary(self) -> str:
        best = self.current_threshold
        mean_r = float(np.mean([h["reward"] for h in self.history])) if self.history else 0.0
        return (f"ThresholdBandit — best_thr={best:.2f}  "
                f"steps={len(self.history)}  mean_reward={mean_r:.3f}  "
                f"Q={dict(zip(self.thresholds, np.round(self.q, 3)))}")
