"""
Foundation-style pipeline (lightweight):

  1) Self-supervised pretrain on IMU windows (no labels)
  2) Supervised fine-tune of HAR head (+ optional backbone)

This is NOT an industry foundation model — it follows the same
pretrain→adapt pattern at our dataset scale.

Usage:
    python scripts/train_foundation_style.py --data data/processed \\
        --ssl_epochs 5 --ft_epochs 5 --out_dir checkpoints
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.data.homogenization import load_processed
from src.models.har_model import build_model
from src.training.ssl_pretrain import ssl_pretrain_backbone
from src.evaluation.metrics import macro_f1


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supervised_finetune(model, X, y, epochs, batch_size, lr, device, val_ratio=0.2):
    device_t = torch.device(device)
    model = model.to(device_t)
    rng = np.random.default_rng(0)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = max(1, int(len(X) * val_ratio))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def loader(ii, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X[ii].astype(np.float32)),
            torch.from_numpy(y[ii].astype(np.int64)),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = loader(tr_idx, True)
    val_loader = loader(val_idx, False)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_f1, best_state = -1.0, None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device_t), yb.to(device_t)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device_t)
                preds.append(model(xb).argmax(-1).cpu().numpy())
                labels.append(yb.numpy())
        f1 = macro_f1(np.concatenate(labels), np.concatenate(preds))
        print(f"  [FT] epoch {ep}/{epochs}  val_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--ssl_epochs", type=int, default=5)
    p.add_argument("--ft_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_windows", type=int, default=15000)
    p.add_argument("--out_dir", default="checkpoints")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print("=== Foundation-style pipeline (SSL → supervised FT) ===")

    X, y, subjects, origins = load_processed(args.data)
    if args.max_windows and len(X) > args.max_windows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), args.max_windows, replace=False)
        X, y = X[idx], y[idx]

    n_classes = int(y.max()) + 1
    model = build_model(n_classes=n_classes)

    print("\n1) SSL pretrain...")
    ssl_pretrain_backbone(
        model.backbone, X, n_epochs=args.ssl_epochs,
        batch_size=args.batch_size, device=device)

    print("\n2) Supervised fine-tune...")
    model, f1 = supervised_finetune(
        model, X, y, epochs=args.ft_epochs,
        batch_size=args.batch_size, lr=1e-4, device=device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "foundation_style.pt"
    model.save(str(path))
    print(f"\nDone. val_F1={f1:.4f}  → {path}")
    print("Note: scale-limited foundation-*style* pipeline, not a public FM.")


if __name__ == "__main__":
    main()
