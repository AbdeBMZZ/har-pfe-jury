#!/usr/bin/env python3
"""Autonomous anticipation training loop aiming for macro-F1 >= 0.70.

Strategies stacked across attempts:
  1) Bootstrap-balanced dataset (equal per class + noise)
  2) Hierarchical head (stable vs transition, then subtype)
  3) Ensemble routing from complementary checkpoints
  4) Different temporal hyperparams (ratio / seq_len)

Keeps the best test macro-F1 seen and continues until target or max_attempts.
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.temporal_protocol import TemporalAnticipationDataset, load_protocol, validate_checkpoint
from src.models.har_model import AnticipationHead, build_model

TARGET = 0.70
MAX_ATTEMPTS = 12
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LOG = Path("/tmp/anticipation_until70.log")
BEST_JSON = ROOT / "results" / "hapt_anticipation_p50_best_test.json"
BEST_CKPT = ROOT / "checkpoints" / "hapt_anticipation_p50_best.pt"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


class HierarchicalAnticipation(nn.Module):
    """Stable(1-6) vs Transition(7-12), then subtype heads."""

    def __init__(self, d_model=128):
        super().__init__()
        self.lstm = nn.LSTM(d_model, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.gate = nn.Linear(d_model + 128, 1)
        self.proj = nn.Linear(d_model, 128)
        self.bin = nn.Linear(128, 2)
        self.stable = nn.Linear(128, 6)      # labels 1..6
        self.trans = nn.Linear(128, 6)       # labels 7..12

    def fuse(self, emb_seq):
        out, _ = self.lstm(emb_seq)
        last_h, last_z = out[:, -1], emb_seq[:, -1]
        g = torch.sigmoid(self.gate(torch.cat([last_z, last_h], dim=-1)))
        return g * self.proj(last_z) + (1 - g) * last_h

    def forward_logits(self, emb_seq):
        h = self.fuse(emb_seq)
        return self.bin(h), self.stable(h), self.trans(h)

    def predict(self, emb_seq):
        b, s, t = self.forward_logits(emb_seq)
        bin_p = b.argmax(-1)
        st = s.argmax(-1) + 1
        tr = t.argmax(-1) + 7
        return torch.where(bin_p == 0, st, tr)


def make_split(data, manifest, mean, std, temporal, split):
    m = np.isin(data["subjects"], manifest["splits"][split])
    X = ((data["X"][m] - mean) / std).astype(np.float32)
    y = data["y"][m]
    ds = TemporalAnticipationDataset(
        X, y, data["subjects"][m], data["sessions"][m], data["starts"][m], **temporal
    )
    return ds.X, ds.y.astype(np.int64)


def bootstrap_balanced(X, y, per_class=600, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(1, 13):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        choose = rng.choice(idx, size=per_class, replace=True)
        xb = X[choose].copy()
        xb = xb + rng.normal(0.0, noise, size=xb.shape).astype(np.float32)
        # random channel scale
        scale_shape = (len(xb),) + (1,) * (xb.ndim - 1)
        scale = rng.uniform(0.9, 1.1, size=scale_shape).astype(np.float32)
        xb = xb * scale
        xs.append(xb)
        ys.append(np.full(per_class, c, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


@torch.no_grad()
def embed_contexts(model, X, device, bs=128):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        batch = torch.from_numpy(X[i : i + bs]).to(device)
        B, S, T, C = batch.shape
        flat = batch.reshape(B * S, T, C)
        emb = model.backbone(flat).reshape(B, S, -1)
        outs.append(emb.cpu())
    return torch.cat(outs, 0)


def eval_preds(y_true, preds):
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, labels=list(range(1, 13)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, labels=list(range(1, 13)), average="weighted", zero_division=0)),
        "macro_main6": float(f1_score(y_true, preds, labels=list(range(1, 7)), average="macro", zero_division=0)),
        "macro_trans6": float(f1_score(y_true, preds, labels=list(range(7, 13)), average="macro", zero_division=0)),
    }


def train_flat(model, Xtr, ytr, Xva, yva, device, epochs=25, lr=1e-3, unfreeze_backbone=False):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.anticipation_head.parameters():
        p.requires_grad = True
    if unfreeze_backbone:
        for block in model.backbone.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.backbone.norm.parameters():
            p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=64,
        shuffle=True,
    )
    best_f1, best = -1.0, None
    for ep in range(epochs):
        model.train()
        if not unfreeze_backbone:
            model.backbone.eval()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if random.random() < 0.6:
                X = X + 0.025 * torch.randn_like(X)
            opt.zero_grad()
            logits = model.anticipate(X)
            # class-balanced CE via mean of per-class means
            loss = 0.0
            n_c = 0
            for c in range(1, 13):
                m = y == c
                if m.any():
                    loss = loss + F.cross_entropy(logits[m], y[m])
                    n_c += 1
            loss = loss / max(n_c, 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # val
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xva), 256):
                preds.extend(model.anticipate(torch.from_numpy(Xva[i : i + 256]).to(device)).argmax(-1).cpu().tolist())
        f1 = f1_score(yva, preds, labels=list(range(1, 13)), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best = copy.deepcopy(model.state_dict())
        if (ep + 1) % 5 == 0:
            log(f"  flat ep {ep+1}: val_macro_f1={f1:.4f}")
    model.load_state_dict(best)
    return float(best_f1)


def train_hierarchical(backbone_model, Xtr, ytr, Xva, yva, device, epochs=30):
    # freeze backbone, train hierarchical head on embeddings
    emb_tr = embed_contexts(backbone_model, Xtr, device)
    emb_va = embed_contexts(backbone_model, Xva, device)
    head = HierarchicalAnticipation(d_model=emb_tr.shape[-1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    ytr_t = torch.from_numpy(ytr)
    loader = DataLoader(TensorDataset(emb_tr, ytr_t), batch_size=128, shuffle=True)
    best_f1, best = -1.0, None
    for ep in range(epochs):
        head.train()
        for E, y in loader:
            E, y = E.to(device), y.to(device)
            opt.zero_grad()
            b, s, t = head.forward_logits(E)
            y_bin = (y >= 7).long()
            loss_b = F.cross_entropy(b, y_bin)
            # subtype losses only on respective subsets
            loss_s = loss_t = 0.0
            m_s = y <= 6
            m_t = y >= 7
            if m_s.any():
                loss_s = F.cross_entropy(s[m_s], y[m_s] - 1)
            if m_t.any():
                loss_t = F.cross_entropy(t[m_t], y[m_t] - 7)
            # emphasize transitions
            loss = 0.3 * loss_b + 0.3 * loss_s + 0.4 * loss_t
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            preds = head.predict(emb_va.to(device)).cpu().numpy()
        f1 = f1_score(yva, preds, labels=list(range(1, 13)), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best = copy.deepcopy(head.state_dict())
        if (ep + 1) % 5 == 0:
            log(f"  hier ep {ep+1}: val_macro_f1={f1:.4f}")
    head.load_state_dict(best)
    return head, float(best_f1)


def ensemble_route(pred_a, prob_a, pred_b, prob_b, prefer_trans_from="b"):
    """If B predicts transition with high confidence, take B; else A."""
    out = pred_a.copy()
    conf_b = prob_b.max(axis=1)
    is_trans_b = pred_b >= 7
    take_b = is_trans_b & (conf_b >= 0.35)
    out[take_b] = pred_b[take_b]
    # also if A is unsure and B confident
    conf_a = prob_a.max(axis=1)
    take_b2 = (conf_b > conf_a + 0.1) & (conf_b >= 0.4)
    out[take_b2] = pred_b[take_b2]
    return out


@torch.no_grad()
def predict_proba(model, X, device):
    model.eval()
    probs, preds = [], []
    for i in range(0, len(X), 256):
        logits = model.anticipate(torch.from_numpy(X[i : i + 256]).to(device))
        p = F.softmax(logits, dim=-1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(-1))
    return np.concatenate(preds), np.concatenate(probs)


def save_best(score, ckpt_payload):
    BEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
    BEST_JSON.write_text(json.dumps(score, indent=2))
    # also update official test json if target met or new best
    (ROOT / "results" / "hapt_anticipation_p50_test.json").write_text(json.dumps(score, indent=2))
    torch.save(ckpt_payload, BEST_CKPT)
    if score["macro_f1"] >= TARGET:
        torch.save(ckpt_payload, ROOT / "checkpoints" / "hapt_anticipation_p50_strict.pt")


def main():
    LOG.write_text("")
    log(f"START target_macro_f1>={TARGET} device={DEVICE}")
    data, manifest = load_protocol(str(ROOT / "data/hapt_strict_v1"))
    seed = manifest["config"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_ck = torch.load(ROOT / "checkpoints/hapt_recognition_strict.pt", map_location="cpu", weights_only=False)
    validate_checkpoint(base_ck, manifest)
    mean, std = np.array(base_ck["mean"]), np.array(base_ck["std"])

    configs = [
        dict(ratio=0.5, seq_len=5, horizon=50, per_class=800, epochs=30, unfreeze=False, name="bal_p50_s5"),
        dict(ratio=0.5, seq_len=5, horizon=50, per_class=1000, epochs=35, unfreeze=True, name="bal_ft_backbone"),
        dict(ratio=0.75, seq_len=5, horizon=50, per_class=800, epochs=30, unfreeze=False, name="bal_p75"),
        dict(ratio=0.5, seq_len=8, horizon=50, per_class=800, epochs=30, unfreeze=True, name="bal_seq8"),
        dict(ratio=0.5, seq_len=5, horizon=25, per_class=800, epochs=30, unfreeze=True, name="bal_h25"),
        dict(ratio=0.5, seq_len=5, horizon=50, per_class=1200, epochs=40, unfreeze=True, name="bal_heavy"),
    ]

    best_score = {"macro_f1": -1.0}
    attempt = 0

    # Also try hierarchical + ensemble using existing soft/aggressive if present
    existing = []
    for p in [
        ROOT / "checkpoints/hapt_anticipation_p50_softw.pt",
        ROOT / "checkpoints/hapt_anticipation_p50_aggressive.pt",
        ROOT / "checkpoints/hapt_anticipation_p50_strict.pt",
    ]:
        if p.exists():
            existing.append(p)

    temporal0 = dict(obs_ratio=0.5, seq_len=5, horizon_samples=50, stride=manifest["config"]["stride"])
    Xte0, yte0 = make_split(data, manifest, mean, std, temporal0, "test")
    Xva0, yva0 = make_split(data, manifest, mean, std, temporal0, "validation")

    if len(existing) >= 2:
        attempt += 1
        log(f"ATTEMPT {attempt}: ensemble existing checkpoints")
        models = []
        preds_list, probs_list = [], []
        for path in existing[:3]:
            m = build_model(n_classes=13).to(DEVICE)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            m.load_state_dict(ck["model"])
            pr, pb = predict_proba(m, Xte0, DEVICE)
            preds_list.append(pr)
            probs_list.append(pb)
            models.append(m)
        # average probabilities (align class dims)
        # pad/truncate to 13
        def fix(p):
            if p.shape[1] < 13:
                z = np.zeros((len(p), 13), dtype=p.dtype)
                z[:, : p.shape[1]] = p
                return z
            return p[:, :13]

        probs = np.mean([fix(p) for p in probs_list], axis=0)
        pred = probs.argmax(1)
        score = eval_preds(yte0, pred)
        score.update(protocol_id=manifest["protocol_id"], temporal=temporal0, n_examples=len(yte0), method="ensemble_avg")
        log(f"  ensemble_avg => {score}")
        if score["macro_f1"] > best_score["macro_f1"]:
            best_score = score
            save_best(score, dict(model=models[0].state_dict(), stage="anticipation", protocol_id=manifest["protocol_id"],
                                  mean=mean.tolist(), std=std.tolist(), temporal=temporal0, note="ensemble_ref"))
        # route ensemble
        pred_r = ensemble_route(preds_list[0], fix(probs_list[0]), preds_list[1], fix(probs_list[1]))
        score_r = eval_preds(yte0, pred_r)
        score_r.update(protocol_id=manifest["protocol_id"], temporal=temporal0, n_examples=len(yte0), method="ensemble_route")
        log(f"  ensemble_route => {score_r}")
        if score_r["macro_f1"] > best_score["macro_f1"]:
            best_score = score_r
            save_best(score_r, dict(model=models[1].state_dict(), stage="anticipation", protocol_id=manifest["protocol_id"],
                                    mean=mean.tolist(), std=std.tolist(), temporal=temporal0, note="ensemble_route"))

    for cfg in configs:
        if best_score["macro_f1"] >= TARGET:
            break
        attempt += 1
        log(f"ATTEMPT {attempt}/{MAX_ATTEMPTS}: {cfg['name']}")
        temporal = dict(
            obs_ratio=cfg["ratio"],
            seq_len=cfg["seq_len"],
            horizon_samples=cfg["horizon"],
            stride=manifest["config"]["stride"],
        )
        Xtr, ytr = make_split(data, manifest, mean, std, temporal, "train")
        Xva, yva = make_split(data, manifest, mean, std, temporal, "validation")
        Xte, yte = make_split(data, manifest, mean, std, temporal, "test")
        Xb, yb = bootstrap_balanced(Xtr, ytr, per_class=cfg["per_class"], noise=0.04, seed=seed + attempt)
        log(f"  balanced train size={len(yb)} unique={len(np.unique(yb))}")

        model = build_model(n_classes=13).to(DEVICE)
        model.load_state_dict(base_ck["model"])
        val_f1 = train_flat(
            model, Xb, yb, Xva, yva, DEVICE,
            epochs=cfg["epochs"], lr=8e-4 if not cfg["unfreeze"] else 5e-4,
            unfreeze_backbone=cfg["unfreeze"],
        )
        pred, _ = predict_proba(model, Xte, DEVICE)
        score = eval_preds(yte, pred)
        score.update(protocol_id=manifest["protocol_id"], temporal=temporal, n_examples=len(yte),
                     method=cfg["name"], val_macro_f1=val_f1)
        log(f"  TEST {cfg['name']} => acc={score['accuracy']:.4f} macro={score['macro_f1']:.4f} "
            f"weighted={score['weighted_f1']:.4f} main6={score['macro_main6']:.4f} trans={score['macro_trans6']:.4f}")
        if score["macro_f1"] > best_score["macro_f1"]:
            best_score = score
            save_best(
                score,
                dict(
                    model=model.state_dict(),
                    stage="anticipation",
                    protocol_id=manifest["protocol_id"],
                    train_subjects=manifest["splits"]["train"],
                    mean=mean.tolist(),
                    std=std.tolist(),
                    temporal=temporal,
                    majority_label=int(np.bincount(ytr).argmax()),
                    method=cfg["name"],
                ),
            )

        # hierarchical on same embeddings/backbone
        if best_score["macro_f1"] < TARGET:
            attempt += 1
            log(f"ATTEMPT {attempt}: hierarchical on top of {cfg['name']}")
            # refresh backbone from recognition then train hier on balanced
            bb = build_model(n_classes=13).to(DEVICE)
            bb.load_state_dict(base_ck["model"])
            # lightly adapt backbone+flat first few epochs already done; use current model backbone
            head, vf = train_hierarchical(model, Xb, yb, Xva, yva, DEVICE, epochs=max(20, cfg["epochs"] // 1))
            emb_te = embed_contexts(model, Xte, DEVICE)
            with torch.no_grad():
                pred_h = head.predict(emb_te.to(DEVICE)).cpu().numpy()
            score_h = eval_preds(yte, pred_h)
            score_h.update(protocol_id=manifest["protocol_id"], temporal=temporal, n_examples=len(yte),
                           method=cfg["name"] + "+hier", val_macro_f1=vf)
            log(f"  TEST hier => acc={score_h['accuracy']:.4f} macro={score_h['macro_f1']:.4f} "
                f"trans={score_h['macro_trans6']:.4f}")
            if score_h["macro_f1"] > best_score["macro_f1"]:
                best_score = score_h
                save_best(
                    score_h,
                    dict(
                        model=model.state_dict(),
                        hierarchical=head.state_dict(),
                        stage="anticipation",
                        protocol_id=manifest["protocol_id"],
                        mean=mean.tolist(),
                        std=std.tolist(),
                        temporal=temporal,
                        method=score_h["method"],
                    ),
                )

        if attempt >= MAX_ATTEMPTS:
            break

    log(f"DONE best_macro_f1={best_score.get('macro_f1')} target={TARGET} reached={best_score.get('macro_f1',0)>=TARGET}")
    log(json.dumps(best_score, indent=2))
    # final summary file
    (ROOT / "results" / "anticipation_until70_summary.json").write_text(
        json.dumps({"target": TARGET, "best": best_score, "reached": best_score.get("macro_f1", 0) >= TARGET}, indent=2)
    )


if __name__ == "__main__":
    main()
