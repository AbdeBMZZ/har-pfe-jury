"""
Train fall-risk anticipation (binary) + optional RL threshold bandit demo.

Usage:
    python scripts/train_fall_risk.py --data data/processed \\
        --checkpoint checkpoints/pretrained.pt --epochs 20
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data.homogenization import load_processed
from src.data.anticipation_dataset import build_anticipation_datasets
from src.models.har_model import build_model
from src.models.fall_risk import FallRiskHead, labels_to_fall_risk, FALL_RISK_LABELS
from src.training.fall_risk_trainer import train_fall_risk
from src.training.rl_threshold_agent import ThresholdBandit
from src.models.online_calibration import OnlineCalibrator


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--rl_demo", action="store_true",
                   help="Run threshold bandit on val confidences after train")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Fall-risk labels: {sorted(FALL_RISK_LABELS)}")

    X, y, subjects, origins = load_processed(args.data)
    n_classes = int(y.max()) + 1
    model = build_model(n_classes=n_classes)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k.removeprefix("backbone."): v
             for k, v in ckpt["model_state"].items()
             if k.startswith("backbone.")}
    model.backbone.load_state_dict(state)
    print(f"Backbone loaded from {args.checkpoint}")

    datasets = build_anticipation_datasets(
        X, y, subjects=subjects, test_ratio=0.2, seq_len=5,
        transitions_only=True, truncate_from="end",
        ratios=[0.5, 0.75],
    )

    results = train_fall_risk(
        model, datasets, n_epochs=args.epochs,
        batch_size=args.batch_size, device=device)

    print("\n=== Fall-risk anticipation ===")
    for r, res in sorted(results.items()):
        print(f"  p={r:.2f}  F1={res['val_f1']:.4f}  Acc={res['val_acc']:.4f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "head_state"}
                    for k, v in results.items()},
        "fall_risk_head": getattr(model, "fall_risk_head", FallRiskHead()).state_dict()
        if hasattr(model, "fall_risk_head") else None,
        "backbone_from": args.checkpoint,
    }
    # prefer best head from results
    best_r = max(results, key=lambda k: results[k]["val_f1"])
    torch.save({
        "fall_risk_head": results[best_r]["head_state"],
        "obs_ratio": best_r,
        "val_f1": results[best_r]["val_f1"],
        "labels": sorted(FALL_RISK_LABELS),
    }, out / "fall_risk.pt")
    np.save(out / "fall_risk_results.npy", payload["results"])
    print(f"Saved {out / 'fall_risk.pt'}")

    if args.rl_demo and hasattr(model, "fall_risk_head"):
        print("\n=== RL threshold bandit + online calibration (val stream) ===")
        ratio = best_r
        _, val_ds = datasets[ratio]
        head = FallRiskHead(d_model=model.backbone.d_model).to(device)
        head.load_state_dict(results[ratio]["head_state"])
        head.eval()
        model.backbone.eval()
        bandit = ThresholdBandit(epsilon=0.15)
        calibrator = OnlineCalibrator(n_classes=2)
        loader = val_ds.dataloader(batch_size=1, shuffle=False)
        for X_seq, y_next in loader:
            with torch.no_grad():
                X_seq = X_seq.to(device)
                B, S, T, C = X_seq.shape
                emb = model.backbone(X_seq.view(B * S, T, C)).view(B, S, -1)
                logit = head(emb)[0]
                logits2 = torch.stack([torch.zeros_like(logit.cpu()), logit.cpu()])
                prob = torch.sigmoid(logit).item()
                is_pos = bool(labels_to_fall_risk(y_next.numpy())[0] > 0.5)
            bandit.step(confidence=prob, is_positive=is_pos)
            calibrator.feedback(logits2, int(is_pos))
        print(bandit.summary())
        print(f"OnlineCalibration temperature={calibrator.temp.temperature:.3f} "
              f"threshold={calibrator.threshold.threshold:.3f}")


if __name__ == "__main__":
    main()
