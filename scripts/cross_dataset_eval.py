"""
Évaluation cross-dataset — proxy pour la robustesse en vie réelle.

Scénarios :
  1. HAPT → HAPT   (référence intra-dataset)
  2. HAPT → WISDM  (autre appareil, même port)
  3. HAPT → PAMAP2 (placement capteur différent — domain shift)
  4. Few-shot domain adaptation on PAMAP2 (100 labeled windows)

Usage:
    python scripts/cross_dataset_eval.py \
        --data data/processed --checkpoint checkpoints/pretrained.pt
"""

import sys, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.homogenization  import load_processed, UNIFIED_LABELS
from src.data.har_dataset     import HARDataset
from src.models.har_model     import build_model
from src.models.domain_adapter import DomainAdapter, adapt_domain, eval_with_adapter
from src.models.prototype_memory import PrototypeMemory
from src.evaluation.metrics   import macro_f1


def get_device():
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_backbone(model, ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("backbone."): v
             for k, v in ckpt["model_state"].items()
             if k.startswith("backbone.")}
    model.backbone.load_state_dict(state)


def init_prototypes(model, X, y, device_t):
    """Initialize prototype memory from source-domain data."""
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 256):
            X_b = torch.from_numpy(X[i:i + 256]).to(device_t)
            y_b = torch.from_numpy(y[i:i + 256]).to(device_t)
            emb = model.backbone(X_b)
            model.har_head.prototype_memory.update(emb, y_b)


def eval_on_dataset(model, X, y, device_t, adapter=None, mode="prototype"):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    return eval_with_adapter(
        model.backbone, model.har_head, adapter,
        model.har_head.prototype_memory,
        X_t, y_t, device_t, mode=mode,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained.pt")
    parser.add_argument("--n_adapt",    type=int, default=300,
                        help="Few-shot adaptation samples from target domain")
    parser.add_argument("--adapt_epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the few-shot adaptation sample draw")
    args = parser.parse_args()

    device   = get_device()
    device_t = torch.device(device)
    print(f"Device: {device}\n")

    X, y, subjects, origins = load_processed(args.data)
    n_classes = int(y.max()) + 1

    hapt_mask   = np.array([o == "hapt"   for o in origins])
    wisdm_mask  = np.array([o == "wisdm"  for o in origins])
    pamap_mask  = np.array([o == "pamap2" for o in origins])
    mobi_mask   = np.array([o == "mobiact" for o in origins])

    X_hapt,  y_hapt  = X[hapt_mask],  y[hapt_mask]
    X_wisdm, y_wisdm = X[wisdm_mask], y[wisdm_mask]
    X_pamap, y_pamap = X[pamap_mask], y[pamap_mask]
    X_mobi,  y_mobi  = X[mobi_mask],  y[mobi_mask]

    print(f"HAPT  : {len(X_hapt):,} fenêtres")
    print(f"WISDM : {len(X_wisdm):,} fenêtres")
    print(f"PAMAP2: {len(X_pamap):,} fenêtres")
    if len(X_mobi):
        print(f"MobiAct: {len(X_mobi):,} fenêtres")
    print()

    model = build_model(n_classes=n_classes)
    load_backbone(model, args.checkpoint)
    model = model.to(device_t)

    # ── Prototypes from HAPT ─────────────────────────────────────────
    print("Initialisation des prototypes depuis HAPT...")
    init_prototypes(model, X_hapt, y_hapt, device_t)
    print(f"  {model.har_head.prototype_memory.n_classes()} classes\n")

    # ── Intra-dataset reference ────────────────────────────────────────
    f1_hapt, _, _ = eval_on_dataset(model, X_hapt, y_hapt, device_t)
    print(f"Intra-dataset  HAPT  → HAPT  : F1 = {f1_hapt:.4f}")

    # ── HAPT → WISDM ───────────────────────────────────────────────────
    common_w = set(np.unique(y_hapt).tolist()) & set(np.unique(y_wisdm).tolist())
    mask_w   = np.isin(y_wisdm, list(common_w))
    f1_cross, _, _ = eval_on_dataset(
        model, X_wisdm[mask_w], y_wisdm[mask_w], device_t)
    print(f"Cross-dataset  HAPT  → WISDM : F1 = {f1_cross:.4f}"
          f"  ({f1_cross / max(f1_hapt, 1e-8) * 100:.0f}% rétention)")

    # ── HAPT → PAMAP2 (before adaptation) ─────────────────────────────
    common_p = set(np.unique(y_hapt).tolist()) & set(np.unique(y_pamap).tolist())
    mask_p   = np.isin(y_pamap, list(common_p))
    X_p, y_p = X_pamap[mask_p], y_pamap[mask_p]

    f1_pamap = f1_pamap_adapt = None
    if len(X_p) == 0:
        print("Cross-dataset  HAPT  → PAMAP2: skipped (no PAMAP2 windows in data)")
    else:
        f1_pamap, _, _ = eval_on_dataset(model, X_p, y_p, device_t)
        print(f"Cross-dataset  HAPT  → PAMAP2: F1 = {f1_pamap:.4f}"
              f"  ({f1_pamap / max(f1_hapt, 1e-8) * 100:.0f}% rétention)"
              f"  [sans adaptation]")

        # ── Few-shot domain adaptation on PAMAP2 ──────────────────────
        # Balanced per-class draw: macro-F1 weighs every class equally, so a
        # uniform-random few-shot draw (which mirrors PAMAP2's own class
        # imbalance) starves rare classes and hurts macro-F1 disproportionately.
        print(f"\nAdaptation few-shot PAMAP2 ({args.n_adapt} fenêtres, équilibré par classe)...")
        rng = np.random.default_rng(args.seed)
        n_per_class = args.n_adapt // len(common_p)
        adapt_idx_parts = []
        for c in sorted(common_p):
            c_idx = np.where(y_p == c)[0]
            take  = min(n_per_class, len(c_idx))
            adapt_idx_parts.append(rng.choice(c_idx, take, replace=False))
        adapt_idx = np.concatenate(adapt_idx_parts)
        test_idx  = np.setdiff1d(np.arange(len(X_p)), adapt_idx)

        adapter = DomainAdapter(d_model=model.backbone.d_model)
        adapt_domain(
            model.backbone, model.har_head.prototype_memory, adapter,
            torch.from_numpy(X_p[adapt_idx]).float(),
            torch.from_numpy(y_p[adapt_idx]).long(),
            n_epochs=args.adapt_epochs,
            device=device_t,
        )

        model.har_head.prototype_memory = PrototypeMemory(
            d_model=model.backbone.d_model)
        init_prototypes(model, X_hapt, y_hapt, device_t)
        with torch.no_grad():
            for i in range(0, len(adapt_idx), 256):
                idx = adapt_idx[i:i + 256]
                X_b = torch.from_numpy(X_p[idx]).to(device_t)
                y_b = torch.from_numpy(y_p[idx]).to(device_t)
                emb = adapter(model.backbone(X_b))
                model.har_head.prototype_memory.update(emb, y_b)

        f1_pamap_adapt, _, _ = eval_on_dataset(
            model, X_p[test_idx], y_p[test_idx], device_t, adapter=adapter)
        print(f"Cross-dataset  HAPT  → PAMAP2: F1 = {f1_pamap_adapt:.4f}"
              f"  ({f1_pamap_adapt / max(f1_hapt, 1e-8) * 100:.0f}% rétention)"
              f"  [après adaptation {args.n_adapt}-shot]")

    # ── MobiAct (if available) ───────────────────────────────────────
    f1_mobi = None
    if len(X_mobi) > 0:
        common_m = set(np.unique(y_hapt).tolist()) & set(np.unique(y_mobi).tolist())
        mask_m   = np.isin(y_mobi, list(common_m))
        if mask_m.sum() > 0:
            f1_mobi, _, _ = eval_on_dataset(
                model, X_mobi[mask_m], y_mobi[mask_m], device_t)
            print(f"Cross-dataset  HAPT  → MobiAct: F1 = {f1_mobi:.4f}"
                  f"  (classes communes: {sorted(common_m)})")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Résumé évaluation cross-dataset")
    print(f"{'='*60}")
    print(f"  HAPT → HAPT   (référence)         : {f1_hapt:.4f}")
    print(f"  HAPT → WISDM  (autre appareil)    : {f1_cross:.4f}"
          f"  ({f1_cross / max(f1_hapt, 1e-8) * 100:.0f}%)")
    if f1_pamap is not None:
        print(f"  HAPT → PAMAP2 (sans adaptation)   : {f1_pamap:.4f}"
              f"  ({f1_pamap / max(f1_hapt, 1e-8) * 100:.0f}%)")
    if f1_pamap_adapt is not None:
        print(f"  HAPT → PAMAP2 ({args.n_adapt}-shot adapt.) : {f1_pamap_adapt:.4f}"
              f"  ({f1_pamap_adapt / max(f1_hapt, 1e-8) * 100:.0f}%)")
    if f1_mobi is not None:
        print(f"  HAPT → MobiAct                    : {f1_mobi:.4f}")

    results = {
        "hapt_hapt": f1_hapt,
        "hapt_wisdm": f1_cross,
        "n_adapt": args.n_adapt,
    }
    if f1_pamap is not None:
        results["hapt_pamap2"] = f1_pamap
    if f1_pamap_adapt is not None:
        results["hapt_pamap2_adapted"] = f1_pamap_adapt
    if f1_mobi is not None:
        results["hapt_mobiact"] = f1_mobi

    out = Path("checkpoints")
    out.mkdir(exist_ok=True)
    np.save(out / "cross_dataset_results.npy", results)
    if f1_pamap_adapt is not None:
        torch.save(adapter.state_dict(), out / "domain_adapter_pamap2.pt")
        print(f"             checkpoints/domain_adapter_pamap2.pt")
    print(f"\nSauvegardé : checkpoints/cross_dataset_results.npy")


if __name__ == "__main__":
    main()
