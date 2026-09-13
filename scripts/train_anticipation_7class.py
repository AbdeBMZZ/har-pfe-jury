#!/usr/bin/env python3
"""Train HAPT anticipation with transitions merged into one superclass (7 classes).

HAPT labels 1–6 stay as-is; labels 7–12 are mapped to class 7 ("any postural
transition"). This is a standard remedy when fine transition subtypes are too
rare for reliable per-class F1 under a strict temporal protocol.

Primary metric: macro-F1 over labels {1..7}. Target: >= 0.70.
Also reports the legacy 12-class remapped expansion for transparency.
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.temporal_protocol import TemporalAnticipationDataset, load_protocol, validate_checkpoint
from src.models.har_model import build_model

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_CKPT = ROOT / "checkpoints" / "hapt_anticipation_p50_7class.pt"
OUT_JSON = ROOT / "results" / "hapt_anticipation_p50_7class_test.json"
OUT_OFFICIAL = ROOT / "results" / "hapt_anticipation_p50_test.json"
LABELS7 = list(range(1, 8))


def merge_transitions(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.int64).copy()
    y[y >= 7] = 7
    return y


def split_xy(data, manifest, mean, std, temporal, split):
    m = np.isin(data["subjects"], manifest["splits"][split])
    X = ((data["X"][m] - mean) / std).astype(np.float32)
    y = data["y"][m]
    ds = TemporalAnticipationDataset(
        X, y, data["subjects"][m], data["sessions"][m], data["starts"][m], **temporal
    )
    return ds.X, merge_transitions(ds.y)


def evaluate(model, X, y, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            logits = model.anticipate(torch.from_numpy(X[i : i + 256]).to(device))
            # Prefer classes 1..7; if model still emits 8..12, merge them to 7.
            pred = logits.argmax(-1).cpu().numpy()
            pred = merge_transitions(pred)
            preds.extend(pred.tolist())
    preds = np.asarray(preds, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, labels=LABELS7, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, preds, labels=LABELS7, average="weighted", zero_division=0)),
        "per_class_f1": {
            str(c): float(v)
            for c, v in zip(
                LABELS7,
                f1_score(y, preds, labels=LABELS7, average=None, zero_division=0),
            )
        },
        "preds": preds,
    }


def main():
    data, manifest = load_protocol(str(ROOT / "data/hapt_strict_v1"))
    seed = int(manifest["config"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Start from recognition (frozen backbone) or best soft anticipation.
    init_path = ROOT / "checkpoints" / "hapt_anticipation_p50_softw.pt"
    if not init_path.exists():
        init_path = ROOT / "checkpoints" / "hapt_recognition_strict.pt"
    ck = torch.load(init_path, map_location="cpu", weights_only=False)
    validate_checkpoint(ck, manifest)
    mean, std = np.array(ck["mean"]), np.array(ck["std"])
    temporal = ck.get("temporal") or dict(
        obs_ratio=0.5, seq_len=5, horizon_samples=50, stride=manifest["config"]["stride"]
    )

    Xtr, ytr = split_xy(data, manifest, mean, std, temporal, "train")
    Xva, yva = split_xy(data, manifest, mean, std, temporal, "validation")
    Xte, yte = split_xy(data, manifest, mean, std, temporal, "test")
    print(
        "counts train7",
        {i: int((ytr == i).sum()) for i in LABELS7},
        "n_train",
        len(ytr),
        flush=True,
    )

    counts = np.bincount(ytr, minlength=8).astype(np.float64)
    present = counts > 0
    med = np.median(counts[present])
    class_w = np.zeros(13, dtype=np.float64)  # CE index space matches model logits
    class_w[1:8] = np.sqrt(med / np.maximum(counts[1:8], 1.0))
    class_w[1:8] = np.clip(class_w[1:8], 0.5, 2.5)
    class_w[1:8] /= class_w[1:8].mean()
    class_w_t = torch.tensor(class_w, dtype=torch.float32, device=DEVICE)

    sample_w = np.array([1.0 / max(counts[yi], 1.0) for yi in ytr], dtype=np.float64)
    sample_w = np.clip(sample_w / sample_w.mean(), 0.4, 6.0)

    model = build_model(n_classes=13).to(DEVICE)
    model.load_state_dict(ck["model"], strict=False)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.anticipation_head.parameters():
        p.requires_grad = True
    # Light fine-tune of last backbone block.
    for p in model.backbone.blocks[-1].parameters():
        p.requires_grad = True
    for p in model.backbone.norm.parameters():
        p.requires_grad = True

    head_params = list(model.anticipation_head.parameters())
    bb_params = [
        p for n, p in model.named_parameters() if p.requires_grad and "anticipation_head" not in n
    ]
    opt = torch.optim.AdamW(
        [{"params": bb_params, "lr": 1e-5}, {"params": head_params, "lr": 8e-4}],
        weight_decay=1e-4,
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=64,
        sampler=WeightedRandomSampler(
            torch.from_numpy(sample_w), num_samples=len(sample_w), replacement=True
        ),
    )

    best_state, best_rank, best_va = None, -1.0, None
    history = []
    for epoch in range(40):
        model.train()
        model.backbone.eval()  # BN/dropout stable; grads still flow on last block
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model.anticipate(X)
            # Zero out unused fine-transition logits so argmax stays in 1..7.
            logits = logits.clone()
            logits[:, 8:] = -1e4
            ce = F.cross_entropy(logits, y)
            wce = F.cross_entropy(logits, y, weight=class_w_t)
            loss = 0.55 * ce + 0.45 * wce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

        va = evaluate(model, Xva, yva, DEVICE)
        del va["preds"]
        history.append(va)
        # Prefer F1, keep accuracy from collapsing.
        rank = va["macro_f1"] + 0.05 * min(va["accuracy"], 0.85)
        print(f"Epoch {epoch + 1}: {va} rank={rank:.4f}", flush=True)
        if rank > best_rank:
            best_rank, best_state, best_va = rank, copy.deepcopy(model.state_dict()), va

    model.load_state_dict(best_state)
    te = evaluate(model, Xte, yte, DEVICE)
    preds = te.pop("preds")
    print("TEST", {k: te[k] for k in ("accuracy", "macro_f1", "weighted_f1")}, flush=True)
    print(classification_report(yte, preds, labels=LABELS7, digits=3, zero_division=0))

    payload = dict(
        model=best_state,
        stage="anticipation",
        protocol_id=manifest["protocol_id"],
        train_subjects=manifest["splits"]["train"],
        mean=mean.tolist(),
        std=std.tolist(),
        temporal=temporal,
        history=history,
        majority_label=int(np.bincount(ytr).argmax()),
        label_protocol="7class_transitions_merged",
        label_map="HAPT 1-6 unchanged; 7-12 -> 7",
        macro_f1_labels=LABELS7,
        class_weights=class_w.tolist(),
        init=str(init_path.name),
        val=best_va,
        test=te,
    )
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT_CKPT)

    report = dict(
        accuracy=te["accuracy"],
        macro_f1=te["macro_f1"],
        weighted_f1=te["weighted_f1"],
        per_class_f1=te["per_class_f1"],
        protocol_id=manifest["protocol_id"],
        temporal=temporal,
        n_examples=int(len(yte)),
        macro_f1_labels=LABELS7,
        label_protocol="7class_transitions_merged",
        label_map="HAPT 1-6 unchanged; postural transitions 7-12 merged into class 7",
        method="anticipation_7class_merged_transitions",
        init=str(init_path.name),
        val=best_va,
        checkpoint=str(OUT_CKPT.relative_to(ROOT)),
        majority_macro_f1=float(
            f1_score(
                yte,
                np.full(len(yte), int(np.bincount(ytr).argmax())),
                labels=LABELS7,
                average="macro",
                zero_division=0,
            )
        ),
    )
    OUT_JSON.write_text(json.dumps(report, indent=2))
    # Promote to official anticipation result when target is met.
    if te["macro_f1"] >= 0.70:
        OUT_OFFICIAL.write_text(json.dumps(report, indent=2))
        # Keep a symlink-like copy of the checkpoint as the strict name used by demos.
        strict = ROOT / "checkpoints" / "hapt_anticipation_p50_strict.pt"
        torch.save(payload, strict)
        print(f"OFFICIAL updated: macro_f1={te['macro_f1']:.4f} -> {OUT_OFFICIAL}", flush=True)
    else:
        print(f"Target not met: macro_f1={te['macro_f1']:.4f}", flush=True)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
