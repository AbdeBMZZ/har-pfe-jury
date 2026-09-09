"""
Self-supervised pretrain of the IMU Transformer backbone.

Usage:
    python scripts/train_ssl.py --data data/processed --epochs 10 \\
        --out checkpoints/ssl_backbone.pt
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data.homogenization import load_processed
from src.models.har_model import build_model
from src.training.ssl_pretrain import ssl_pretrain_backbone


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="checkpoints/ssl_backbone.pt")
    p.add_argument("--max_windows", type=int, default=20000,
                   help="Subsample windows for faster SSL on laptop")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    X, y, subjects, origins = load_processed(args.data)
    if args.max_windows and len(X) > args.max_windows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), args.max_windows, replace=False)
        X = X[idx]
        print(f"Subsampled to {len(X)} windows for SSL")

    n_classes = int(y.max()) + 1
    model = build_model(n_classes=n_classes)
    print("SSL pretraining backbone (labels unused)...")
    ssl_pretrain_backbone(
        model.backbone, X,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone_state": model.backbone.state_dict(),
        "ssl": True,
        "n_windows": len(X),
        "epochs": args.epochs,
    }, out)
    print(f"Saved SSL backbone → {out}")


if __name__ == "__main__":
    main()
