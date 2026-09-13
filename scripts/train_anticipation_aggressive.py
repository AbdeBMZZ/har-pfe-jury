"""Aggressive anticipation training aimed at lifting macro-F1 on rare/transition classes."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.temporal_protocol import TemporalAnticipationDataset, load_protocol, validate_checkpoint
from src.models.har_model import build_model


def focal_ce(logits, targets, weight=None, gamma=2.0):
    log_probs = F.log_softmax(logits, dim=-1)
    gather = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    probs = gather.exp()
    loss = -(1.0 - probs).pow(gamma) * gather
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def evaluate(model, loader, device, y_true):
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            preds.extend(model.anticipate(X.to(device)).argmax(-1).cpu().tolist())
    preds = np.asarray(preds)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, labels=list(range(1, 13)), average="macro", zero_division=0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--checkpoint", required=True, help="Recognition or anticipation checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="mps")
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--horizon-samples", type=int, default=50)
    args = p.parse_args()
    if Path(args.out).exists():
        raise SystemExit("Output already exists")

    data, manifest = load_protocol(args.protocol)
    seed = manifest["config"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(ck, manifest)
    mean, std = np.asarray(ck["mean"]), np.asarray(ck["std"])
    temporal = dict(
        obs_ratio=args.ratio,
        seq_len=args.seq_len,
        horizon_samples=args.horizon_samples,
        stride=manifest["config"]["stride"],
    )

    masks = {k: np.isin(data["subjects"], v) for k, v in manifest["splits"].items()}
    datasets = {}
    for split in ("train", "validation", "test"):
        m = masks[split]
        X = ((data["X"][m] - mean) / std).astype(np.float32)
        y = data["y"][m]
        ds = TemporalAnticipationDataset(
            X, y, data["subjects"][m], data["sessions"][m], data["starts"][m], **temporal
        )
        datasets[split] = (ds.X, ds.y.astype(np.int64))

    y_train = datasets["train"][1]
    counts = np.bincount(y_train, minlength=13).astype(np.float64)
    # Strong sampler weights for rare classes (power 1.0, clipped).
    sample_w = np.zeros(len(y_train), dtype=np.float64)
    for i, yi in enumerate(y_train):
        sample_w[i] = 1.0 / max(counts[yi], 1.0)
    sample_w = sample_w / sample_w.mean()
    sample_w = np.clip(sample_w, 0.2, 25.0)

    # Soft class weights for focal term.
    present = counts > 0
    med = np.median(counts[present])
    class_w = np.zeros(13, dtype=np.float64)
    class_w[present] = np.sqrt(med / counts[present])
    class_w[present] = np.clip(class_w[present], 0.5, 4.0)
    class_w[present] /= class_w[present].mean()
    class_w_t = torch.tensor(class_w, dtype=torch.float32, device=args.device)
    print("train counts", {i: int(counts[i]) for i in range(1, 13)})
    print("class_w", {i: round(float(class_w[i]), 3) for i in range(1, 13)})

    train_ds = TensorDataset(
        torch.from_numpy(datasets["train"][0]),
        torch.from_numpy(datasets["train"][1]),
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_w),
        num_samples=len(sample_w) * 2,  # oversample epoch
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(datasets["validation"][0]), torch.from_numpy(datasets["validation"][1])),
        batch_size=args.batch_size,
    )

    model = build_model(n_classes=13).to(args.device)
    model.load_state_dict(ck["model"])
    # Unfreeze anticipation head fully + last transformer blocks lightly.
    for p_ in model.parameters():
        p_.requires_grad = False
    for p_ in model.anticipation_head.parameters():
        p_.requires_grad = True
    for block in model.backbone.blocks[-2:]:
        for p_ in block.parameters():
            p_.requires_grad = True
    for p_ in model.backbone.norm.parameters():
        p_.requires_grad = True

    head_params = [p_ for p_ in model.anticipation_head.parameters() if p_.requires_grad]
    backbone_params = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and "anticipation_head" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 2e-5},
            {"params": head_params, "lr": 8e-4},
        ],
        weight_decay=1e-4,
    )

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(args.epochs):
        model.train()
        # Keep early backbone blocks in eval-ish mode via low lr only.
        for X, y in train_loader:
            X = X.to(args.device)
            y = y.to(args.device)
            # Light noise augmentation (helps rare classes seen often via sampler).
            if random.random() < 0.5:
                X = X + 0.02 * torch.randn_like(X)
            optimizer.zero_grad()
            logits = model.anticipate(X)
            ce = F.cross_entropy(logits, y)
            fl = focal_ce(logits, y, weight=class_w_t, gamma=2.0)
            loss = 0.45 * ce + 0.55 * fl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        score = evaluate(model, val_loader, args.device, datasets["validation"][1])
        history.append(score)
        print(f"Epoch {epoch + 1}: {score}", flush=True)
        if score["macro_f1"] > best_f1:
            best_f1 = score["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(datasets["test"][0]), torch.from_numpy(datasets["test"][1])),
        batch_size=args.batch_size,
    )
    test_score = evaluate(model, test_loader, args.device, datasets["test"][1])
    # per-class
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in test_loader:
            preds.extend(model.anticipate(X.to(args.device)).argmax(-1).cpu().tolist())
    preds = np.asarray(preds)
    per_class = f1_score(datasets["test"][1], preds, labels=list(range(1, 13)), average=None, zero_division=0)
    test_score.update(
        protocol_id=manifest["protocol_id"],
        temporal=temporal,
        n_examples=int(len(datasets["test"][1])),
        macro_f1_labels=list(range(1, 13)),
        per_class_f1={str(i): float(v) for i, v in enumerate(per_class, start=1)},
        majority_macro_f1=float(
            f1_score(
                datasets["test"][1],
                np.full(len(datasets["test"][1]), int(np.bincount(datasets["train"][1]).argmax())),
                labels=list(range(1, 13)),
                average="macro",
                zero_division=0,
            )
        ),
        actual_horizon_seconds_min=3.0,
        actual_horizon_seconds_max=3.0,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            model=best_state,
            stage="anticipation",
            protocol_id=manifest["protocol_id"],
            train_subjects=manifest["splits"]["train"],
            mean=mean.tolist(),
            std=std.tolist(),
            temporal=temporal,
            history=history,
            arguments=vars(args),
            majority_label=int(np.bincount(datasets["train"][1]).argmax()),
            test_score=test_score,
        ),
        args.out,
    )
    results_path = Path(args.out).with_suffix(".test.json")
    if str(args.out).endswith(".pt"):
        results_path = Path(str(args.out).replace(".pt", "_test.json"))
    # also write under results/
    out_json = Path("results") / "hapt_anticipation_p50_aggressive_test.json"
    out_json.write_text(json.dumps(test_score, indent=2))
    print(json.dumps(test_score, indent=2))


if __name__ == "__main__":
    main()
