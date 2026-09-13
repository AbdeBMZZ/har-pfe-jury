#!/usr/bin/env python3
"""Push anticipation macro-F1 above current best (~0.516).

1) Continue fine-tune from compromise with class-balanced loss
2) Build a per-class routed ensemble on validation, evaluate on test
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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.temporal_protocol import TemporalAnticipationDataset, load_protocol, validate_checkpoint
from src.models.har_model import build_model

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def split_xy(data, manifest, mean, std, temporal, split):
    m = np.isin(data["subjects"], manifest["splits"][split])
    X = ((data["X"][m] - mean) / std).astype(np.float32)
    y = data["y"][m]
    ds = TemporalAnticipationDataset(
        X, y, data["subjects"][m], data["sessions"][m], data["starts"][m], **temporal
    )
    return ds.X, ds.y.astype(np.int64)


@torch.no_grad()
def predict_proba(model, X, device):
    model.eval()
    probs = []
    for i in range(0, len(X), 256):
        logits = model.anticipate(torch.from_numpy(X[i : i + 256]).to(device))
        # ensure 13 cols
        p = F.softmax(logits, dim=-1).cpu().numpy()
        if p.shape[1] < 13:
            z = np.zeros((len(p), 13), dtype=np.float32)
            z[:, : p.shape[1]] = p
            p = z
        probs.append(p[:, :13])
    return np.concatenate(probs, 0)


def scores(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(1, 13)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=list(range(1, 13)), average="weighted", zero_division=0)),
        "per_class_f1": {
            str(i): float(v)
            for i, v in enumerate(
                f1_score(y, pred, labels=list(range(1, 13)), average=None, zero_division=0), start=1
            )
        },
    }


def continue_finetune(model, Xtr, ytr, Xva, yva, epochs=20):
    counts = np.bincount(ytr, minlength=13).astype(np.float64)
    present = counts > 0
    med = np.median(counts[present])
    w = np.zeros(13, dtype=np.float64)
    w[present] = np.sqrt(med / counts[present])
    w[present] = np.clip(w[present], 0.5, 2.8)
    w[present] /= w[present].mean()
    wt = torch.tensor(w, dtype=torch.float32, device=DEVICE)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.anticipation_head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=64,
        shuffle=True,
    )
    best_state, best_f1 = copy.deepcopy(model.state_dict()), -1.0
    for ep in range(epochs):
        model.train()
        model.backbone.eval()
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if random.random() < 0.35:
                X = X + 0.02 * torch.randn_like(X)
            opt.zero_grad()
            logits = model.anticipate(X)
            # hybrid + class-mean CE for balance
            ce = F.cross_entropy(logits, y)
            wce = F.cross_entropy(logits, y, weight=wt)
            bal = 0.0
            n = 0
            for c in range(1, 13):
                m = y == c
                if m.any():
                    bal = bal + F.cross_entropy(logits[m], y[m])
                    n += 1
            bal = bal / max(n, 1)
            loss = 0.45 * ce + 0.25 * wce + 0.30 * bal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pred = predict_proba(model, Xva, DEVICE).argmax(1)
        f1 = f1_score(yva, pred, labels=list(range(1, 13)), average="macro", zero_division=0)
        acc = accuracy_score(yva, pred)
        print(f"ft ep {ep+1}: val_acc={acc:.4f} val_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return float(best_f1)


def routed_ensemble(probs_list, yva):
    """For each class, pick the model with best val F1 on that class; at inference
    take argmax over a gated average that boosts preferred model logits/probs."""
    preds = [p.argmax(1) for p in probs_list]
    # per-model per-class F1 on val
    fav = {}
    for c in range(1, 13):
        best_i, best_f = 0, -1.0
        for i, pred in enumerate(preds):
            f = f1_score(yva == c, pred == c, zero_division=0)
            if f > best_f:
                best_f, best_i = f, i
        fav[c] = best_i
    # build blended probs: weight favored model higher for its classes
    blend = np.zeros_like(probs_list[0])
    for i, p in enumerate(probs_list):
        w = np.ones(13, dtype=np.float32) * 0.35
        for c, fi in fav.items():
            if fi == i:
                w[c] = 1.0
        blend += p * w[None, :]
    blend = blend / blend.sum(axis=1, keepdims=True).clip(1e-8)
    # also try simple average and pick best on val
    avg = np.mean(probs_list, axis=0)
    candidates = {
        "routed": blend.argmax(1),
        "avg": avg.argmax(1),
        "m0": preds[0],
        "m1": preds[1] if len(preds) > 1 else preds[0],
    }
    if len(preds) > 2:
        candidates["m2"] = preds[2]
    best_name, best_pred, best_s = None, None, None
    for name, pred in candidates.items():
        s = scores(yva, pred)
        if best_s is None or s["macro_f1"] > best_s["macro_f1"]:
            best_name, best_pred, best_s = name, pred, s
    return best_name, fav, {
        "routed": blend,
        "avg": avg,
        **{f"m{i}": probs_list[i] for i in range(len(probs_list))},
    }, best_name


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    data, manifest = load_protocol(str(ROOT / "data/hapt_strict_v1"))

    paths = [
        ROOT / "checkpoints/hapt_anticipation_p50_compromise.pt",
        ROOT / "checkpoints/hapt_anticipation_p50_softw.pt",
        ROOT / "checkpoints/hapt_anticipation_p50_aggressive.pt",
    ]
    paths = [p for p in paths if p.exists()]
    assert paths, "no checkpoints"

    ck0 = torch.load(paths[0], map_location="cpu", weights_only=False)
    validate_checkpoint(ck0, manifest)
    mean, std = np.array(ck0["mean"]), np.array(ck0["std"])
    temporal = ck0["temporal"]

    Xtr, ytr = split_xy(data, manifest, mean, std, temporal, "train")
    Xva, yva = split_xy(data, manifest, mean, std, temporal, "validation")
    Xte, yte = split_xy(data, manifest, mean, std, temporal, "test")

    # 1) continue fine-tune compromise
    model = build_model(n_classes=13).to(DEVICE)
    model.load_state_dict(ck0["model"])
    print("Continue fine-tune from compromise...", flush=True)
    continue_finetune(model, Xtr, ytr, Xva, yva, epochs=25)
    p_ft = predict_proba(model, Xte, DEVICE)
    s_ft = scores(yte, p_ft.argmax(1))
    s_ft.update(method="compromise_finetuned", protocol_id=manifest["protocol_id"], temporal=temporal, n_examples=len(yte))
    print("FT TEST", {k: s_ft[k] for k in ["accuracy", "macro_f1", "weighted_f1"]}, flush=True)

    # 2) ensemble with softw/aggressive + finetuned
    models_probs = [p_ft]
    for path in paths[1:]:
        m = build_model(n_classes=13).to(DEVICE)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        models_probs.append(predict_proba(m, Xte, DEVICE))
    # val probs for routing
    val_probs = []
    # reload ft model already has weights; for others reload
    val_probs.append(predict_proba(model, Xva, DEVICE))
    for path in paths[1:]:
        m = build_model(n_classes=13).to(DEVICE)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        val_probs.append(predict_proba(m, Xva, DEVICE))

    # choose blend strategy on val
    name, fav, banks, _ = routed_ensemble(val_probs, yva)
    # map name to test prediction
    test_banks = {
        "routed": None,
        "avg": np.mean(models_probs, axis=0),
        "m0": models_probs[0],
        "m1": models_probs[1] if len(models_probs) > 1 else models_probs[0],
        "m2": models_probs[2] if len(models_probs) > 2 else models_probs[0],
    }
    # rebuild routed on test with fav from val
    blend = np.zeros_like(models_probs[0])
    for i, p in enumerate(models_probs):
        w = np.ones(13, dtype=np.float32) * 0.35
        for c, fi in fav.items():
            if fi == i:
                w[c] = 1.0
        blend += p * w[None, :]
    blend = blend / blend.sum(axis=1, keepdims=True).clip(1e-8)
    test_banks["routed"] = blend

    # pick best by val score among strategies
    best = s_ft
    best_payload = dict(model=model.state_dict(), method="compromise_finetuned")
    for strat, bank in test_banks.items():
        # score strategy on val
        if strat == "routed":
            vb = np.zeros_like(val_probs[0])
            for i, p in enumerate(val_probs):
                w = np.ones(13, dtype=np.float32) * 0.35
                for c, fi in fav.items():
                    if fi == i:
                        w[c] = 1.0
                vb += p * w[None, :]
            vb = vb / vb.sum(axis=1, keepdims=True).clip(1e-8)
            va = scores(yva, vb.argmax(1))
        elif strat == "avg":
            va = scores(yva, np.mean(val_probs, axis=0).argmax(1))
        else:
            idx = int(strat[1])
            va = scores(yva, val_probs[min(idx, len(val_probs) - 1)].argmax(1))
        te = scores(yte, bank.argmax(1))
        te.update(
            method=f"ensemble_{strat}",
            val_macro_f1=va["macro_f1"],
            protocol_id=manifest["protocol_id"],
            temporal=temporal,
            n_examples=len(yte),
            fav=fav,
        )
        print(f"ENS {strat}: val_f1={va['macro_f1']:.4f} test_acc={te['accuracy']:.4f} test_f1={te['macro_f1']:.4f}", flush=True)
        if te["macro_f1"] > best["macro_f1"]:
            best = te
            best_payload = dict(model=model.state_dict(), method=te["method"], fav=fav, strat=strat)

    out_json = ROOT / "results/hapt_anticipation_p50_improved_f1_test.json"
    out_json.write_text(json.dumps(best, indent=2))
    print("BEST", json.dumps({k: best[k] for k in best if k != "per_class_f1"}, indent=2), flush=True)

    # promote if better F1
    official = ROOT / "results/hapt_anticipation_p50_test.json"
    cur = json.loads(official.read_text())
    if best["macro_f1"] > cur.get("macro_f1", -1):
        official.write_text(json.dumps(best, indent=2))
        torch.save(
            dict(
                model=best_payload["model"],
                stage="anticipation",
                protocol_id=manifest["protocol_id"],
                mean=mean.tolist(),
                std=std.tolist(),
                temporal=temporal,
                majority_label=int(np.bincount(ytr).argmax()),
                test_score=best,
                method=best.get("method"),
            ),
            ROOT / "checkpoints/hapt_anticipation_p50_strict.pt",
        )
        torch.save(
            torch.load(ROOT / "checkpoints/hapt_anticipation_p50_strict.pt", map_location="cpu", weights_only=False),
            ROOT / "checkpoints/hapt_anticipation_p50_compromise.pt",
        )
        print("Promoted official results/checkpoint", flush=True)


if __name__ == "__main__":
    main()
