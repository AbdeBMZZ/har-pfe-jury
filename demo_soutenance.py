"""
Démonstration PFE — script pour la session demo (après présentation + débat).

Usage:
    .venv/bin/python3 demo_soutenance.py

Scénario narratif : « Karim reçoit une montre connectée »
  1. Reconnaissance d'activité (Module 1)
  2. Nouvel utilisateur sans ré-entraînement (continual learning)
  3. Anticipation de la prochaine activité (Module 2)
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data.homogenization import load_processed, load_subject_meta, resolve_subject, UNIFIED_LABELS
from src.data.har_dataset import HARDataset
from src.data.anticipation_dataset import AnticipationDataset
from src.models.har_model import build_model

# ── Affichage terminal ───────────────────────────────────────────────────────
G, Y, C, B, R = "\033[92m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"

EXAMPLES_RECO = [
    ("Marche",        "walking"),
    ("Course",        "jogging_running"),
    ("Escaliers ↑",   "walking_upstairs"),
    ("Assis/Debout",  "sitting_standing"),
]


def pause(msg="Appuyez sur Entrée pour continuer..."):
    try:
        input(f"\n{C}  ▶ {msg}{R}")
    except EOFError:
        pass


def titre(text):
    print(f"\n{B}{C}{'═' * 58}{R}")
    print(f"{B}{C}  {text}{R}")
    print(f"{B}{C}{'═' * 58}{R}\n")


def sous_titre(text):
    print(f"\n{B}  {text}{R}\n")


def label_name(y):
    return UNIFIED_LABELS.get(int(y), str(int(y)))


def load_model():
    print(f"{B}Chargement du système HAR...{R}")
    X, y, subjects, origins = load_processed("data/processed")
    subject_meta = load_subject_meta("data/processed")
    ds = HARDataset(X, y, subjects, origins)
    n_classes = int(y.max()) + 1
    model = build_model(n_classes=n_classes)

    ckpt = torch.load("checkpoints/pretrained.pt",
                      map_location="cpu", weights_only=False)
    backbone_state = {
        k.removeprefix("backbone."): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state)

    ant_path = Path("checkpoints/anticipation.pt")
    if ant_path.exists():
        ant_ckpt = torch.load(ant_path, map_location="cpu", weights_only=False)
        ant_state = {
            k.replace("anticipation_head.", ""): v
            for k, v in ant_ckpt["model_state"].items()
            if k.startswith("anticipation_head.")
        }
        model.anticipation_head.load_state_dict(ant_state, strict=False)
        print(f"  {G}✓{R} Backbone + tête anticipation chargés")
    else:
        print(f"  {Y}!{R} anticipation.pt absent — entraîner avec train_anticipation.py")

    model.eval()
    print(f"  Dataset : {len(ds):,} fenêtres | {len(np.unique(subjects))} sujets")
    print(f"  Modèle  : {sum(p.numel() for p in model.parameters()):,} paramètres\n")
    return ds, model, X, y, subjects, subject_meta


# ── DÉMO 1 ───────────────────────────────────────────────────────────────────
def demo1_recognition(ds, model, X, y):
    titre("DÉMO 1 — Reconnaissance d'activité")
    print("  Scénario : la montre lit 3 secondes de signal IMU")
    print("  et identifie ce que fait la personne.\n")

    with torch.no_grad():
        emb = model.backbone(torch.from_numpy(X[:2000]))
        model.har_head.prototype_memory.update(emb, torch.from_numpy(y[:2000]))

    print(f"  Mémoire initialisée : {model.har_head.prototype_memory.n_classes()} activités\n")
    print(f"  {'Activité réelle':<22} {'Prédiction':<22} {'Résultat'}")
    print(f"  {'-' * 55}")

    rng = np.random.default_rng(42)
    idx = rng.choice(range(2000, len(ds)), 6, replace=False)
    correct = 0

    for wi in idx:
        x, y_true = ds[wi]
        with torch.no_grad():
            pred = model.har_head.prototype_memory.predict(
                model.backbone(x.unsqueeze(0))).item()
        ok = pred == int(y_true)
        if ok:
            correct += 1
        t_name = label_name(y_true)
        p_name = label_name(pred)
        mark = f"{G}✓ Correct{R}" if ok else f"{Y}✗ Erreur{R}"
        print(f"  {t_name:<22} {p_name:<22} {mark}")

    print(f"\n  → Précision sur cet échantillon : {B}{correct}/6{R}")
    print(f"  {C}Entrée : fenêtre 150 points × 6 capteurs (acc + gyro){R}")


# ── DÉMO 2 ───────────────────────────────────────────────────────────────────
def demo2_karim(model, X, y, subjects, subject_meta):
    titre("DÉMO 2 — Karim, nouvel utilisateur (apprentissage continu)")
    print("  Scénario : Karim reçoit une montre déjà utilisée par 4 personnes.")
    print("  Le système s'adapte à LUI — sans ré-entraîner le réseau.\n")

    karim_id = resolve_subject(subject_meta, "wisdm", 5)
    if karim_id is None:
        karim_id = int(sorted(np.unique(subjects))[-1])
    hapt_base = sorted(
        int(gid) for gid, info in subject_meta.items()
        if info.get("dataset") == "hapt"
    )[:4]

    # Reset prototypes
    from src.models.prototype_memory import PrototypeMemory
    model.har_head.prototype_memory = PrototypeMemory(
        d_model=model.backbone.d_model)

    sous_titre("Étape A — Utilisateurs connus (4 participants HAPT)")
    base_mask = np.isin(subjects, hapt_base)
    with torch.no_grad():
        emb = model.backbone(torch.from_numpy(X[base_mask][:500]))
        model.har_head.prototype_memory.update(
            emb, torch.from_numpy(y[base_mask][:500]))
    print(f"  {G}✓{R} Modèle calibré sur les utilisateurs existants "
          f"(global ids {hapt_base})\n")

    sous_titre("Étape B — Karim arrive (wisdm-5)")
    karim_mask = subjects == karim_id
    X_k, y_k = X[karim_mask], y[karim_mask]
    print(f"  Karim porte la montre → {karim_mask.sum()} fenêtres disponibles")
    print(f"  On utilise seulement {B}100 fenêtres{R} pour l'adapter\n")

    with torch.no_grad():
        emb = model.backbone(torch.from_numpy(X_k[:100]))
        model.har_head.prototype_memory.update(
            emb, torch.from_numpy(y_k[:100]))
    print(f"  {G}✓{R} Prototypes mis à jour — {B}aucun ré-entraînement{R}\n")

    sous_titre("Étape C — Test sur Karim (fenêtres jamais vues)")
    print(f"  {'Activité réelle':<22} {'Prédiction':<22} {'Résultat'}")
    print(f"  {'-' * 55}")

    correct = 0
    with torch.no_grad():
        X_test = torch.from_numpy(X_k[100:106])
        preds = model.har_head.prototype_memory.predict(
            model.backbone(X_test)).numpy()

    for p, t in zip(preds, y_k[100:106]):
        ok = p == t
        if ok:
            correct += 1
        mark = f"{G}✓ Correct{R}" if ok else f"{Y}✗ Erreur{R}"
        print(f"  {label_name(t):<22} {label_name(p):<22} {mark}")

    print(f"\n  → Karim reconnu : {B}{correct}/6{R} activités correctes")
    print(f"  {C}Message clé : adaptation en ligne avec 100 exemples seulement{R}")


# ── DÉMO 3 ───────────────────────────────────────────────────────────────────
def demo3_anticipation(model, X, y, subjects):
    titre("DÉMO 3 — Anticipation")
    print(f"  {B}Parcours jury recommandé{R} : anticipation causale v2")
    print("  → streamlit run app_anticipation.py")
    print("  → ou page « Anticipation » / mode Causale v2 dans app_streamlit.py")
    print("  → results/anticipation_v2_summary.json\n")

    v2 = Path("checkpoints/anticipation_v2_p50/metadata.json")
    if v2.exists():
        print(f"  {G}✓ Bundle v2 présent{R} : {v2.parent}")
    else:
        print(f"  {Y}⚠ Bundle v2 absent{R} — voir LIVRABLE.md")

    print(f"\n  {C}Illustration LSTM historique (matrices fusionnées, non comparable à v2){R}\n")
    ant_ds = AnticipationDataset(
        X, y, subjects=subjects, obs_ratio=0.50,
        seq_len=5, transitions_only=True)

    print(f"  Observation : {B}50 %{R} de chaque fenêtre (1,5 s sur 3 s)")
    print(f"  Tâche       : prédire la transition imminente\n")
    print(f"  {'Prochaine activité (vraie)':<28} {'Prédiction':<22} {'Résultat'}")
    print(f"  {'-' * 58}")

    rng = np.random.default_rng(7)
    idx = rng.choice(len(ant_ds), 6, replace=False)
    correct = 0
    model.eval()

    for wi in idx:
        x_seq, y_next = ant_ds[wi]
        with torch.no_grad():
            pred = model.anticipate(x_seq.unsqueeze(0)).argmax(dim=-1).item()
        ok = pred == int(y_next)
        if ok:
            correct += 1
        mark = f"{G}✓ Correct{R}" if ok else f"{Y}✗ Erreur{R}"
        print(f"  {label_name(y_next):<28} {label_name(pred):<22} {mark}")

    print(f"\n  → LSTM historique : {B}{correct}/6{R} (démo uniquement)")
    print(f"  {C}Ne pas présenter comme prévention de chute ni comme score v2{R}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{B}{'=' * 58}{R}")
    print(f"{B}  DÉMONSTRATION PFE — HAR Apprentissage Continu{R}")
    print(f"{B}  NEDJAM Walaa — ESI 2025/2026{R}")
    print(f"{B}{'=' * 58}{R}")

    ds, model, X, y, subjects, subject_meta = load_model()

    print(f"{C}Structure de la démo :{R}")
    print("  1. Reconnaissance d'activité")
    print("  2. Nouvel utilisateur (Karim) — sans ré-entraînement")
    print("  3. Anticipation (v2 causale recommandée + LSTM historique)\n")

    pause("Commencer la démo 1...")
    demo1_recognition(ds, model, X, y)

    pause("Passer à la démo 2 (Karim)...")
    demo2_karim(model, X, y, subjects, subject_meta)

    pause("Passer à la démo 3 (anticipation)...")
    demo3_anticipation(model, X, y, subjects)

    titre("FIN DE LA DÉMONSTRATION")
    print("  Module 1 : Transformer + prototypes + replay + contrastive")
    print("  Module 2 : anticipation causale v2 (+ LSTM historique)")
    print("  Données  : HAPT + WISDM (capteurs IMU publics)")
    print(f"\n  {G}Merci — questions ?{R}\n")


if __name__ == "__main__":
    main()
