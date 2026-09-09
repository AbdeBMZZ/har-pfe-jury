"""
Dataset for activity anticipation training.

For each consecutive pair of windows (w_t, w_{t+1}) within the SAME subject:
  - Input:  last p% of each context window  →  partial observation
  - Target: label of w_{t+1} →  next activity to predict

We build sequences of S consecutive partial windows as context,
letting the LSTM in the anticipation head model temporal progression.

Observation ratios: p ∈ {0.25, 0.50, 0.75}
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except (ImportError, OSError):
    TORCH_OK = False


def _build_subject_sequences(
    X: np.ndarray,
    y: np.ndarray,
    obs_len: int,
    seq_len: int,
    transitions_only: bool,
    truncate_from: str = "end",
) -> Tuple[List[np.ndarray], List[int]]:
    """Build anticipation samples from one subject's contiguous windows."""
    contexts, targets = [], []

    for i in range(seq_len, len(X)):
        ctx_wins = X[i - seq_len: i]
        if truncate_from == "end":
            # Keep the last obs_len samples — transition cues are often late
            ctx_trunc = ctx_wins[:, -obs_len:, :]
        else:
            ctx_trunc = ctx_wins[:, :obs_len, :]
        next_label = int(y[i])

        if transitions_only and next_label == int(y[i - 1]):
            continue

        contexts.append(ctx_trunc)
        targets.append(next_label)

    return contexts, targets


class AnticipationDataset:
    """
    Builds (context_windows, next_label) pairs from a HARDataset.

    Sequences are built **within each subject** so that consecutive windows
    are truly temporally adjacent (critical for anticipation).

    Args:
        X:             (N, T, C) windows
        y:             (N,)     labels
        subjects:      (N,)     subject IDs — required for correct sequencing
        obs_ratio:     fraction of each window to observe (0.25, 0.50, 0.75)
        seq_len:       S — number of consecutive windows as context
        transitions_only: if True, only keep samples where label changes
        truncate_from: "end" (default, last p%) or "start" (first p%)
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 subjects: np.ndarray | None = None,
                 obs_ratio: float = 0.50,
                 seq_len: int = 5,
                 transitions_only: bool = False,
                 truncate_from: str = "end"):

        self.obs_ratio = obs_ratio
        if not 0 < obs_ratio <= 1 or seq_len < 1:
            raise ValueError("obs_ratio must be in (0, 1] and seq_len must be positive")
        if truncate_from not in {"start", "end"}:
            raise ValueError("truncate_from must be 'start' or 'end'")
        if subjects is None:
            raise ValueError("Subject IDs are required; random windows are not temporal sequences")
        self.seq_len   = seq_len
        self.truncate_from = truncate_from
        T              = X.shape[1]
        self.obs_len   = max(1, int(T * obs_ratio))

        contexts, targets = [], []

        if subjects is None:
            ctx, tgt = _build_subject_sequences(
                X, y, self.obs_len, seq_len, transitions_only, truncate_from)
            contexts.extend(ctx)
            targets.extend(tgt)
        else:
            for subj in np.unique(subjects):
                mask = subjects == subj
                X_s, y_s = X[mask], y[mask]
                if len(X_s) <= seq_len:
                    continue
                ctx, tgt = _build_subject_sequences(
                    X_s, y_s, self.obs_len, seq_len, transitions_only,
                    truncate_from)
                contexts.extend(ctx)
                targets.extend(tgt)

        if contexts:
            self.X = np.array(contexts, dtype=np.float32)
            self.y = np.array(targets,  dtype=np.int64)
        else:
            self.X = np.zeros((0, seq_len, self.obs_len, X.shape[-1]),
                              dtype=np.float32)
            self.y = np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        if TORCH_OK:
            return (torch.from_numpy(self.X[idx]),
                    torch.tensor(self.y[idx], dtype=torch.long))
        return self.X[idx], self.y[idx]

    def dataloader(self, batch_size: int = 64, shuffle: bool = True):
        if not TORCH_OK:
            raise RuntimeError("PyTorch not available.")
        from torch.utils.data import DataLoader
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    def class_weights(self, n_classes: Optional[int] = None) -> "torch.Tensor":
        """Inverse-frequency class weights for CrossEntropy (size n_classes)."""
        if not TORCH_OK:
            raise RuntimeError("PyTorch not available.")
        if n_classes is None:
            n_classes = int(self.y.max()) + 1 if len(self.y) else 1
        counts = np.bincount(self.y, minlength=n_classes).astype(np.float64)
        weights = np.zeros(n_classes, dtype=np.float64)
        present = counts > 0
        # median frequency balancing (robust to extreme imbalance)
        med = np.median(counts[present])
        weights[present] = med / counts[present]
        weights = weights / weights[present].mean()  # normalize mean=1
        return torch.tensor(weights, dtype=torch.float32)

    def majority_baseline(self) -> float:
        """Macro-F1 of always predicting the most frequent class."""
        if len(self.y) == 0:
            return 0.0
        from ..evaluation.metrics import macro_f1
        majority = int(np.bincount(self.y).argmax())
        preds    = np.full(len(self.y), majority, dtype=np.int64)
        return macro_f1(self.y, preds)

    def summary(self) -> str:
        unique, counts = np.unique(self.y, return_counts=True)
        lines = [
            f"AnticipationDataset — {len(self)} samples",
            (f"  obs_ratio={self.obs_ratio}  seq_len={self.seq_len}"
             f"  truncate_from={self.truncate_from}"),
            f"  context shape: {self.X.shape}",
            f"  majority baseline F1: {self.majority_baseline():.4f}",
            f"  classes: {dict(zip(unique.tolist(), counts.tolist()))}",
        ]
        return "\n".join(lines)


def build_anticipation_datasets(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray | None = None,
    test_ratio: float = 0.2,
    seq_len: int = 5,
    seed: int = 42,
    transitions_only: bool = True,
    truncate_from: str = "end",
    ratios: Optional[List[float]] = None,
):
    """
    Build train/val anticipation datasets for observation ratios.

    Split is by **subject** to avoid temporal leakage between train and val.

    Returns:
        {ratio: (train_ds, val_ds), ...}
    """
    if ratios is None:
        ratios = [0.25, 0.50, 0.75]

    rng = np.random.default_rng(seed)

    if subjects is None:
        raise ValueError("Subject IDs are required for a disjoint anticipation split")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be in (0, 1)")
    if subjects is not None:
        unique_subjects = np.unique(subjects)
        if len(unique_subjects) < 2:
            raise ValueError("At least two subjects are needed for a subject-disjoint split")
        rng.shuffle(unique_subjects)
        n_val = max(1, int(len(unique_subjects) * test_ratio))
        val_subjects  = set(unique_subjects[:n_val])
        train_mask    = np.array([s not in val_subjects for s in subjects])
        val_mask      = ~train_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        s_train = subjects[train_mask]
        s_val   = subjects[val_mask]
    else:
        idx = np.arange(len(X))
        rng.shuffle(idx)
        n_val = int(len(X) * test_ratio)
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        s_train, s_val   = None, None

    result = {}
    for ratio in ratios:
        train_ds = AnticipationDataset(
            X_train, y_train, subjects=s_train,
            obs_ratio=ratio, seq_len=seq_len,
            transitions_only=transitions_only,
            truncate_from=truncate_from,
        )
        val_ds = AnticipationDataset(
            X_val, y_val, subjects=s_val,
            obs_ratio=ratio, seq_len=seq_len,
            transitions_only=transitions_only,
            truncate_from=truncate_from,
        )
        result[ratio] = (train_ds, val_ds)

    return result
