"""
Interface Streamlit — démonstration PFE HAR devant le jury.

Lancement :
    .venv/bin/streamlit run app_streamlit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.anticipation_dataset import AnticipationDataset
from src.data.homogenization import (
    UNIFIED_LABELS,
    load_processed,
    load_subject_meta,
    resolve_subject,
    subject_slug,
)
from src.data.har_dataset import HARDataset
from src.models.har_model import build_model
from src.models.prototype_memory import PrototypeMemory
from src.models.fall_risk import FallRiskHead, labels_to_fall_risk, FALL_RISK_LABELS
from src.models.online_calibration import OnlineCalibrator
from src.training.rl_threshold_agent import ThresholdBandit
from src.scenarios.adl_scenarios import meal_status

# Prénoms fictifs pour la démo jury — clé = slug « dataset-local_id » (ex. wisdm-5).
# Les IDs globaux uniques sont dans data/processed/subject_meta.json.
DEMO_SUBJECT_NAMES: dict[str, str] = {
    "hapt-1": "Marie",
    "hapt-2": "Thomas",
    "hapt-3": "Léa",
    "hapt-4": "Nadia",
    "wisdm-5": "Karim",
    "hapt-5": "Youssef",
    "hapt-6": "Claire",
    "hapt-7": "Hugo",
    "hapt-8": "Sarah",
    "hapt-9": "Omar",
    "hapt-10": "Inès",
    "hapt-11": "Lucas",
    "hapt-12": "Fatima",
    "hapt-13": "Antoine",
    "hapt-14": "Zoé",
    "hapt-15": "Mehdi",
    "hapt-16": "Julie",
    "hapt-17": "Adam",
    "hapt-18": "Camille",
    "hapt-19": "Rachid",
    "hapt-20": "Élise",
    "hapt-21": "Paul",
    "hapt-22": "Amina",
    "hapt-23": "Nicolas",
    "hapt-24": "Chloé",
    "hapt-25": "Samir",
    "hapt-26": "Manon",
    "hapt-27": "Julien",
    "hapt-28": "Sofia",
    "hapt-29": "Romain",
    "hapt-30": "Laura",
    "wisdm-31": "David",
    "wisdm-32": "Nour",
    "wisdm-33": "Maxime",
    "wisdm-34": "Lina",
    "wisdm-35": "Alexandre",
    "wisdm-36": "Sonia",
    "pamap2-1": "Philippe",
    "pamap2-2": "Céline",
    "pamap2-3": "Marc",
    "pamap2-4": "Aïcha",
    "pamap2-5": "Benoît",
    "pamap2-6": "Salma",
    "pamap2-7": "Luc",
    "pamap2-8": "Emma",
    "pamap2-9": "Rayan",
}


def subject_label(subject_id: int, meta: dict | None = None) -> str:
    """Affichage jury : « 42-Karim (wisdm-5) » — 1 ID = 1 personne réelle."""
    sid = int(subject_id)
    if not meta:
        return str(sid)
    slug = subject_slug(sid, meta)
    name = DEMO_SUBJECT_NAMES.get(slug)
    if name:
        return f"{sid}-{name} ({slug})"
    return f"{sid}-{slug}"


st.set_page_config(
    page_title="HAR — Démo PFE",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 700; color: #0050a0; }
    .subtitle   { color: #555; margin-bottom: 1rem; }
    .explain-box {
        background: #f8f9fa; border-left: 4px solid #0050a0;
        padding: 10px 14px; margin: 8px 0 16px 0; border-radius: 4px;
        font-size: 0.92rem; color: #333;
    }
    div[data-testid="stMetric"] {
        background: #f0f6ff; padding: 12px; border-radius: 8px;
        border-left: 4px solid #0050a0;
    }
</style>
""", unsafe_allow_html=True)


def explain(text: str):
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)


def label_name(y: int) -> str:
    return UNIFIED_LABELS.get(int(y), str(int(y)))


@st.cache_resource(show_spinner="Chargement du modèle et des données…")
def load_system():
    data_dir = ROOT / "data" / "processed"
    ckpt_dir = ROOT / "checkpoints"
    ckpt_path = ckpt_dir / "pretrained.pt"
    ant_path = ckpt_dir / "anticipation.pt"
    fall_path = ckpt_dir / "fall_risk.pt"
    ssl_path = ckpt_dir / "ssl_backbone.pt"
    found_path = ckpt_dir / "foundation_style.pt"

    if not data_dir.exists():
        st.error(f"Dossier introuvable : {data_dir}")
        st.stop()
    if not ckpt_path.exists():
        st.error(f"Checkpoint introuvable : {ckpt_path}")
        st.stop()

    X, y, subjects, origins = load_processed(data_dir)
    subject_meta = load_subject_meta(data_dir)
    ds = HARDataset(X, y, subjects, origins)
    n_classes = int(y.max()) + 1
    model = build_model(n_classes=n_classes)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone_state = {
        k.removeprefix("backbone."): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state)

    has_anticipation = ant_path.exists()
    if has_anticipation:
        ant_ckpt = torch.load(ant_path, map_location="cpu", weights_only=False)
        ant_state = {
            k.replace("anticipation_head.", ""): v
            for k, v in ant_ckpt["model_state"].items()
            if k.startswith("anticipation_head.")
        }
        model.anticipation_head.load_state_dict(ant_state, strict=False)

    fall_risk_head = None
    fall_meta = {}
    has_fall_risk = fall_path.exists()
    if has_fall_risk:
        fr = torch.load(fall_path, map_location="cpu", weights_only=False)
        fall_risk_head = FallRiskHead(d_model=model.backbone.d_model)
        fall_risk_head.load_state_dict(fr["fall_risk_head"])
        fall_risk_head.eval()
        fall_meta = {"obs_ratio": fr.get("obs_ratio"), "val_f1": fr.get("val_f1")}

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": model,
        "ds": ds,
        "X": X,
        "y": y,
        "subjects": subjects,
        "origins": origins,
        "subject_meta": subject_meta,
        "n_params": n_params,
        "has_anticipation": has_anticipation,
        "has_fall_risk": has_fall_risk,
        "fall_risk_head": fall_risk_head,
        "has_ssl": ssl_path.exists(),
        "has_foundation": found_path.exists(),
        "fall_risk_meta": fall_meta,
    }


def predict_prototype(model, x: torch.Tensor) -> int:
    with torch.no_grad():
        emb = model.backbone(x.unsqueeze(0) if x.dim() == 2 else x)
        return model.har_head.prototype_memory.predict(emb).item()


def results_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "correct" in df.columns:
        df["Résultat"] = df["correct"].map({True: "✓ Correct", False: "✗ Erreur"})
    return df


def plot_imu_window(window: np.ndarray, title: str = "Signal IMU (3 s)"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    t = np.arange(window.shape[0]) / 50.0
    for i, lab in enumerate(["acc_x", "acc_y", "acc_z"]):
        axes[0].plot(t, window[:, i], label=lab, alpha=0.85)
    axes[0].set_ylabel("Accélération (m/s²)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)
    for i, lab in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
        axes[1].plot(t, window[:, i + 3], label=lab, alpha=0.85)
    axes[1].set_ylabel("Gyroscope (rad/s)")
    axes[1].set_xlabel("Temps (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


def run_demo1(model, ds, X, y, n_samples: int, seed: int):
    model.har_head.prototype_memory = PrototypeMemory(d_model=model.backbone.d_model)
    with torch.no_grad():
        emb = model.backbone(torch.from_numpy(X[:2000]))
        model.har_head.prototype_memory.update(emb, torch.from_numpy(y[:2000]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(range(2000, len(ds)), n_samples, replace=False)
    rows = []
    for wi in idx:
        x, y_true = ds[wi]
        pred = predict_prototype(model, x)
        rows.append({
            "Activité réelle": label_name(y_true),
            "Prédiction": label_name(pred),
            "correct": pred == int(y_true),
            "index": int(wi),
        })
    return rows, model.har_head.prototype_memory.n_classes()


def run_demo2(model, X, y, subjects, new_user: int, n_adapt: int, n_test: int,
              subject_meta: dict | None = None):
    model.har_head.prototype_memory = PrototypeMemory(d_model=model.backbone.d_model)
    # Utilisateurs connus : 4 participants HAPT (autre personne que le nouvel utilisateur)
    if subject_meta:
        hapt_ids = sorted(
            int(gid) for gid, info in subject_meta.items()
            if info.get("dataset") == "hapt"
        )
        base_subjects = [s for s in hapt_ids if s != new_user][:4]
    else:
        base_subjects = [s for s in sorted(np.unique(subjects)) if s != new_user][:4]
    base_mask = np.isin(subjects, base_subjects)
    with torch.no_grad():
        emb = model.backbone(torch.from_numpy(X[base_mask][:500]))
        model.har_head.prototype_memory.update(
            emb, torch.from_numpy(y[base_mask][:500]))
    user_mask = subjects == new_user
    X_u, y_u = X[user_mask], y[user_mask]
    n_adapt = min(n_adapt, len(X_u) - n_test)
    n_test = min(n_test, len(X_u) - n_adapt)
    with torch.no_grad():
        emb = model.backbone(torch.from_numpy(X_u[:n_adapt]))
        model.har_head.prototype_memory.update(
            emb, torch.from_numpy(y_u[:n_adapt]))
    rows = []
    with torch.no_grad():
        X_test = torch.from_numpy(X_u[n_adapt: n_adapt + n_test])
        preds = model.har_head.prototype_memory.predict(
            model.backbone(X_test)).numpy()
    for p, t in zip(preds, y_u[n_adapt: n_adapt + n_test]):
        rows.append({
            "Activité réelle": label_name(t),
            "Prédiction": label_name(p),
            "correct": int(p) == int(t),
        })
    meta = {
        "base_subjects": base_subjects,
        "n_windows": int(user_mask.sum()),
        "n_adapt": n_adapt,
        "n_test": n_test,
    }
    return rows, meta


def run_demo3(model, X, y, subjects, n_samples: int, seed: int, obs_ratio: float):
    ant_ds = AnticipationDataset(
        X, y, subjects=subjects,
        obs_ratio=obs_ratio, seq_len=5, transitions_only=True,
        truncate_from="end",
    )
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(ant_ds))
    idx = rng.choice(len(ant_ds), n, replace=False) if n > 0 else []
    rows = []
    model.eval()
    for wi in idx:
        x_seq, y_next = ant_ds[wi]
        with torch.no_grad():
            pred = model.anticipate(x_seq.unsqueeze(0)).argmax(dim=-1).item()
        rows.append({
            "Prochaine activité (vraie)": label_name(y_next),
            "Prédiction": label_name(pred),
            "correct": pred == int(y_next),
        })
    baseline = ant_ds.majority_baseline() if len(ant_ds) > 0 else 0.0
    return rows, baseline, int(obs_ratio * 100)


def run_demo_fall_risk(model, fall_head, X, y, subjects, n_samples, seed, obs_ratio):
    ant_ds = AnticipationDataset(
        X, y, subjects=subjects,
        obs_ratio=obs_ratio, seq_len=5, transitions_only=True,
        truncate_from="end",
    )
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(ant_ds))
    idx = rng.choice(len(ant_ds), n, replace=False) if n > 0 else []
    rows = []
    model.eval()
    fall_head.eval()
    for wi in idx:
        x_seq, y_next = ant_ds[wi]
        y_bin = int(labels_to_fall_risk(np.array([int(y_next)]))[0])
        with torch.no_grad():
            S, T, C = x_seq.shape
            emb = model.backbone(x_seq.view(S, T, C)).view(1, S, -1)
            logit = fall_head(emb)[0]
            prob = float(torch.sigmoid(logit).item())
            pred = int(prob >= 0.5)
        rows.append({
            "Prochaine activité": label_name(y_next),
            "Risque réel": "Oui" if y_bin else "Non",
            "Prédiction": "Oui" if pred else "Non",
            "Confiance risque": f"{prob:.2f}",
            "correct": pred == y_bin,
            "prob": prob,
            "y_bin": y_bin,
        })
    return rows


def run_demo_calibration_rl(model, fall_head, X, y, subjects, n_steps, seed, obs_ratio):
    ant_ds = AnticipationDataset(
        X, y, subjects=subjects,
        obs_ratio=obs_ratio, seq_len=5, transitions_only=True,
        truncate_from="end",
    )
    rng = np.random.default_rng(seed)
    n = min(n_steps, len(ant_ds))
    idx = rng.choice(len(ant_ds), n, replace=False) if n > 0 else []
    bandit = ThresholdBandit(epsilon=0.15, seed=seed)
    calibrator = OnlineCalibrator(n_classes=2)
    model.eval()
    fall_head.eval()
    history = []
    for wi in idx:
        x_seq, y_next = ant_ds[wi]
        y_bin = int(labels_to_fall_risk(np.array([int(y_next)]))[0])
        with torch.no_grad():
            S, T, C = x_seq.shape
            emb = model.backbone(x_seq.view(S, T, C)).view(1, S, -1)
            logit = fall_head(emb)[0].cpu()
            prob = float(torch.sigmoid(logit).item())
            logits2 = torch.stack([-logit, logit])
        rec = bandit.step(confidence=prob, is_positive=bool(y_bin))
        calibrator.feedback(logits2, y_bin)
        history.append({
            "Risque réel": "Oui" if y_bin else "Non",
            "Confiance": f"{prob:.2f}",
            "Seuil bandit": f"{rec['threshold']:.2f}",
            "Alerte": "🔔" if rec["alert"] else "—",
            "Reward": f"{rec['reward']:+.2f}",
        })
    return history, bandit, calibrator


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⌚ HAR — Démo PFE")
st.sidebar.markdown("**NEDJAM Walaa** · ESI 2025/2026")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Scénario de démonstration",
    [
        "🏠 Accueil",
        "🧭 Pipeline (brut → résultats)",
        "1️⃣ Reconnaissance",
        "2️⃣ Nouvel utilisateur (Karim)",
        "3️⃣ Anticipation",
        "4️⃣ Fall-risk (équilibre)",
        "5️⃣ Calibration + RL",
        "6️⃣ SSL / Foundation-style",
    ],
    help="Choisissez la partie du système à montrer au jury.",
)

explain_sidebar = st.sidebar.expander("ℹ️ Lien avec notre modèle HARContinualModel")
with explain_sidebar:
    st.markdown("""
**Fenêtres (X)** → entrée de `model.backbone(x)`  
Shape `(B, 150, 6)` · `IMUTransformerEncoder` (4 blocs, 128 dim)

**y** → label supervisé ou vérité pour `prototype_memory.update(emb, y)`

**Sujets** → ID global unique (1 personne = 1 id) · voir `subject_meta.json`

**823 452 paramètres** → `HARContinualModel` complet :
backbone + `ContinualHARHead` + `AnticipationHead` (LSTM)

**Checkpoints** :
- `pretrained.pt` → backbone
- `anticipation_v2_p50/` → anticipation causale (jury)
- `anticipation.pt` → AnticipationHead LSTM (historique)
- `fall_risk.pt` → FallRiskHead (proxy postural)
- `ssl_backbone.pt` / `foundation_style.pt` → SSL / foundation-style
    """)

sys_data = load_system()
model = sys_data["model"]
ds = sys_data["ds"]
X, y, subjects = sys_data["X"], sys_data["y"], sys_data["subjects"]
subject_meta = sys_data["subject_meta"]
label_subject = lambda sid: subject_label(sid, subject_meta)
has_anticipation_v2 = any(Path("checkpoints").glob("anticipation_v2*/metadata.json"))

st.sidebar.markdown("---")
st.sidebar.metric("Fenêtres (X)", f"{len(ds):,}")
st.sidebar.metric("Sujets", len(np.unique(subjects)))
st.sidebar.metric("Paramètres modèle", f"{sys_data['n_params']:,}")
if has_anticipation_v2:
    st.sidebar.success("Anticipation v2 : OK")
elif sys_data["has_anticipation"]:
    st.sidebar.info("Anticipation LSTM : OK")
else:
    st.sidebar.warning("Anticipation absente")
if sys_data["has_fall_risk"]:
    st.sidebar.success("Fall-risk : OK")
else:
    st.sidebar.warning("fall_risk.pt absent")
if sys_data["has_ssl"]:
    st.sidebar.info("SSL backbone : OK")
if sys_data["has_foundation"]:
    st.sidebar.info("Foundation-style : OK")

# ── Pages ────────────────────────────────────────────────────────────────────
if page == "🏠 Accueil":
    st.markdown('<p class="main-title">Apprentissage adaptatif et continu pour la HAR</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Démo live — capteurs IMU · Transformer · Prototypes</p>',
                unsafe_allow_html=True)

    explain(
        "<b>Parcours recommandé :</b> page <b>Pipeline</b> "
        "(données brutes → stockage → modèle → scores), "
        "puis démos live 1 → 2 → 3.<br><br>"
        "<b>Flux modèle :</b><br>"
        "<code>X (150×6)</code> → <code>IMUTransformerEncoder</code> → embedding (128 dim)<br>"
        "→ <code>ContinualHARHead</code> : softmax (pretrain) ou <code>PrototypeMemory</code><br>"
        "→ Anticipation (causale v2 / LSTM) pour la fenêtre future"
    )

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.info("**0 — Pipeline**\n\nBrut → fenêtres → ckpt.")
    with c1:
        st.info("**1 — Reconnaissance**\n\nTransformer + prototypes.")
    with c2:
        st.success("**2 — Continual**\n\nNouvel utilisateur sans retrain.")
    with c3:
        st.warning("**3 — Anticipation**\n\nv2 / LSTM (scores JSON).")

    st.markdown("Pages **4–6** (fall-risk, calibration, SSL) en option selon le temps.")
    st.caption(meal_status())

    with st.expander("📖 Composants du modèle (har_model.py)"):
        st.markdown("""
| Composant | Fichier | Rôle dans la démo |
|-----------|---------|-------------------|
| `IMUTransformerEncoder` | `backbone.py` | `model.backbone(X)` → vecteur 128 dim |
| `ContinualHARHead` | `har_model.py` | Classifieur + prototypes |
| `PrototypeMemory` | `prototype_memory.py` | `update()` puis `predict()` nearest-mean |
| `ReplayBuffer` / UWR | `replay_buffer.py` | Entraînement continual (pas live) |
| `AnticipationHead` | `har_model.py` | `model.anticipate()` |
| `FallRiskHead` | `fall_risk.py` | Alerte transitions posturales |
| `OnlineCalibrator` | `online_calibration.py` | Temperature + seuil |
| `ThresholdBandit` | `rl_threshold_agent.py` | RL léger sur seuil d'alerte |
        """)

elif page == "🧭 Pipeline (brut → résultats)":
    import importlib
    import src.ui.pipeline_panel as pipeline_panel
    importlib.reload(pipeline_panel)
    pipeline_panel.render()

elif page == "1️⃣ Reconnaissance":
    st.markdown("## Démo 1 — Reconnaissance d'activité")
    explain(
        "<b>Dans le code :</b> <code>emb = model.backbone(X)</code> puis "
        "<code>model.har_head.prototype_memory.predict(emb)</code><br>"
        "Le backbone vient de <code>checkpoints/pretrained.pt</code> (pretrain 50 epochs, CrossEntropy)."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("### Paramètres")
        n_samples = st.slider(
            "Nombre d'exemples à tester",
            3, 12, 6,
            help="Nombre de fenêtres passées à model.backbone() puis prototype_memory.predict().",
        )
        explain(
            "<b>Modèle :</b> pour chaque fenêtre test, appel à <code>predict_prototype(model, x)</code> "
            "→ <code>backbone(x)</code> shape (1,128) → distance euclidienne aux prototypes.<br>"
            "Les N fenêtres sont tirées avec <code>index &gt; 2000</code> (non vues à l'init)."
        )

        seed = st.number_input(
            "Seed (graine aléatoire)",
            0, 9999, 42,
            help="Fixe np.random.default_rng(seed) pour choisir les mêmes indices de fenêtres.",
        )
        explain(
            "Fixe <code>rng.choice(range(2000, len(ds)), N)</code> — mêmes fenêtres X à chaque démo."
        )

        run = st.button("▶ Lancer la démo", type="primary", use_container_width=True)
        explain(
            "<b>Au clic — Étape 1 :</b><br>"
            "<code>emb = model.backbone(X[:2000])</code><br>"
            "<code>prototype_memory.update(emb, y[:2000])</code> → crée 1 prototype/classe<br><br>"
            "<b>Étape 2 :</b> pour chaque fenêtre test → <code>predict(emb)</code> nearest-mean"
        )

    if run:
        rows, n_cls = run_demo1(model, ds, X, y, n_samples, seed)
        correct = sum(r["correct"] for r in rows)
        df = results_dataframe(rows)

        with col_b:
            st.markdown("### Résultats")
            m1, m2, m3 = st.columns(3)
            m1.metric("Précision", f"{correct}/{n_samples}")
            m2.metric("Prototypes", n_cls)
            m3.metric("Entrée X", "150 × 6")

            explain(
                "<b>Précision</b> : compare <code>prototype_memory.predict()</code> vs <code>y_true</code>.<br>"
                "<b>Prototypes</b> : <code>prototype_memory.n_classes()</code> — une entrée dans "
                "<code>self.prototypes[cls_id]</code> par activité (128 dim).<br>"
                "<b>Entrée X</b> : shape attendue par <code>IMUTransformerEncoder</code> : (B, 150, 6)."
            )

        st.markdown("### Tableau de prédictions")
        explain(
            "<b>Activité réelle</b> = <code>y[i]</code> du dataset (label unifié 1–12).<br>"
            "<b>Prédiction</b> = <code>argmin_c ||emb - prototype[c]||</code> — "
            "plus proche centroïde dans l'espace appris par le backbone."
        )
        st.dataframe(
            df[["Activité réelle", "Prédiction", "Résultat"]],
            use_container_width=True,
            hide_index=True,
        )

        wi = rows[0]["index"]
        x0, y0 = ds[wi]
        st.markdown("### Visualisation du signal X (1 exemple)")
        explain(
            f"Cette courbe = <b>X[{wi}]</b> avant <code>model.backbone()</code>. "
            f"Label associé <code>y[{wi}]</code> = <b>{label_name(y0)}</b>. "
            "Le Transformer applique 4 blocs d'attention sur ces 150 pas de temps "
            "pour produire le token CLS (embedding 128 dim)."
        )
        fig = plot_imu_window(
            x0.numpy() if hasattr(x0, "numpy") else np.array(x0),
            title=f"X — activité réelle : {label_name(y0)}",
        )
        st.pyplot(fig)
        plt.close(fig)

elif page == "2️⃣ Nouvel utilisateur (Karim)":
    st.markdown("## Démo 2 — Karim, nouvel utilisateur")
    explain(
        "<b>Différence avec l'entraînement continual :</b> ici on n'appelle PAS "
        "<code>model.continual_step()</code> (pas de replay, pas de loss, pas de backprop).<br>"
        "On fait seulement <code>prototype_memory.update(emb, y)</code> — "
        "les poids de <code>model.backbone</code> restent figés (chargés depuis pretrained.pt)."
    )

    available = sorted(np.unique(subjects).tolist())
    karim_id = resolve_subject(subject_meta, "wisdm", 5)
    default_user = karim_id if karim_id in available else available[-1]

    c1, c2 = st.columns(2)
    with c1:
        new_user = st.selectbox(
            "Nouvel utilisateur",
            available,
            index=available.index(default_user),
            format_func=label_subject,
            help="ID global unique + prénom fictif + slug dataset-local "
                 "(ex. 42-Karim (wisdm-5)). Métadonnées : data/processed/subject_meta.json.",
        )
        explain(
            "<b>Code :</b> <code>karim_mask = subjects == N</code> → "
            "<code>X_k, y_k = X[mask], y[mask]</code>. "
            "Même backbone pour tous — seuls les prototypes s'adaptent au style de mouvement de Karim."
        )

        n_adapt = st.slider(
            "Fenêtres d'adaptation",
            50, 300, 100, step=10,
            help="Passées à prototype_memory.update() — équivalent few-shot sans continual_step.",
        )
        explain(
            "<b>Code :</b> <code>emb = model.backbone(X_k[:n_adapt])</code><br>"
            "<code>prototype_memory.update(emb, y_k[:n_adapt])</code><br>"
            "Moyenne glissante (Welford) dans <code>prototype_memory.py</code> — "
            "recalcule le centroïde de chaque classe vue chez Karim."
        )

    with c2:
        n_test = st.slider(
            "Fenêtres de test",
            3, 15, 6,
            help="Passées à prototype_memory.predict() — jamais dans update() avant.",
        )
        explain(
            "<b>Code :</b> <code>preds = prototype_memory.predict(backbone(X_k[n_adapt:]))</code><br>"
            "Évalue si les prototypes recalibrés généralisent sur Karim."
        )

        run = st.button("▶ Simuler l'arrivée de Karim", type="primary",
                        use_container_width=True)
        explain(
            "<b>A</b> <code>PrototypeMemory()</code> reset → init sujets 1–4<br>"
            "<b>B</b> <code>update()</code> sur fenêtres du nouvel utilisateur<br>"
            "<b>C</b> <code>predict()</code> sur fenêtres hold-out — "
            "équivalent <code>model.predict(x, mode='prototype')</code>"
        )

    if run:
        rows, meta = run_demo2(
            model, X, y, subjects, new_user, n_adapt, n_test,
            subject_meta=subject_meta,
        )
        correct = sum(r["correct"] for r in rows)

        st.markdown("### Déroulement")
        s1, s2, s3 = st.columns(3)
        base_labels = [label_subject(s) for s in meta["base_subjects"]]
        s1.success(f"**A** — {', '.join(base_labels)} (utilisateurs connus HAPT)")
        s2.info(
            f"**B** — {label_subject(new_user)} : {meta['n_adapt']} fenêtres d'adaptation "
            f"(sur {meta['n_windows']} total)"
        )
        s3.success(f"**C** — Test : {meta['n_test']} fenêtres jamais vues")

        explain(
            f"<b>{label_subject(new_user)}</b> a <b>{meta['n_windows']} fenêtres</b> "
            f"dans le dataset (une seule personne réelle). "
            f"On utilise {meta['n_adapt']} pour apprendre son profil, "
            f"{meta['n_test']} pour tester."
        )

        st.metric("Précision nouvel utilisateur", f"{correct}/{meta['n_test']}")
        explain(
            "Si élevé → le système s'adapte au nouveau porteur "
            "<b>sans ré-entraînement</b> du réseau (objectif du PFE)."
        )

        st.dataframe(
            results_dataframe(rows)[["Activité réelle", "Prédiction", "Résultat"]],
            use_container_width=True,
            hide_index=True,
        )
        st.success("✓ Transformer gelé — seuls les prototypes (vecteurs moyens par classe) ont changé.")

elif page == "3️⃣ Anticipation":
    st.markdown("## Démo 3 — Anticipation d'activité")
    mode = st.radio(
        "Parcours à montrer",
        [
            "Causale v2 (recommandé jury)",
            "LSTM historique (matrices fusionnées)",
        ],
        horizontal=True,
        help="v2 = protocole session/horizon. LSTM = démo historique, scores non comparables à v2.",
    )

    if mode.startswith("Causale"):
        explain(
            "Modèle <b>indépendant</b> du reconnaisseur : préfixe observé uniquement, "
            "cible = fenêtre <b>future</b> (pas une alerte chute validée). "
            "Détails : <code>results/anticipation_v2_summary.json</code> / <code>LIVRABLE.md</code>."
        )
        from src.ui.anticipation_panel import render as render_anticipation_v2
        render_anticipation_v2()
        st.caption("App autonome : `streamlit run app_anticipation.py`")
    else:
        explain(
            "Entrée : <b>5 fenêtres partielles</b> (seq_len=5). "
            "Sortie : prédiction de l'activité de la <b>fenêtre suivante</b>. "
            "Protocole historique sur matrices fusionnées — ne pas confondre avec v2."
        )

        if not sys_data["has_anticipation"]:
            st.error("Checkpoint absent : `python scripts/train_anticipation.py`")
            st.stop()

        obs_pct = st.select_slider(
            "Part du signal observé (%)",
            options=[25, 50, 75], value=50,
            help="25 % = 0,75 s visible sur 3 s. 50 % = 1,5 s. 75 % = 2,25 s.",
        )
        explain(
            f"<b>{obs_pct} %</b> = tronque chaque fenêtre X à "
            f"{int(150 * obs_pct / 100)} points sur 150."
        )

        n_samples = st.slider(
            "Nombre d'exemples", 3, 12, 6,
            help="Transitions (activité actuelle ≠ suivante).",
            key="ant_n",
        )
        seed = st.number_input("Seed", 0, 9999, 7, key="ant_seed")
        run = st.button("▶ Lancer l'anticipation LSTM", type="primary")

        if run:
            rows, baseline, pct = run_demo3(
                model, X, y, subjects, n_samples, seed, obs_ratio=obs_pct / 100.0)
            correct = sum(r["correct"] for r in rows)

            m1, m2, m3 = st.columns(3)
            m1.metric("Anticipation correcte", f"{correct}/{len(rows)}")
            m2.metric("Signal observé", f"{pct} %")
            m3.metric("Baseline (hasard)", f"{baseline:.3f}")

            st.dataframe(
                results_dataframe(rows)[["Prochaine activité (vraie)", "Prédiction", "Résultat"]],
                use_container_width=True,
                hide_index=True,
            )
            st.warning(
                "Démo illustrative. Ne pas présenter ces scores comme validation "
                "d'anticipation causale ni de prévention de chutes."
            )

elif page == "4️⃣ Fall-risk (équilibre)":
    st.warning(
        "Proxy de transitions posturales (labels 9–12). "
        "Ce n'est **pas** une validation de prévention des chutes."
    )
    st.markdown("## Démo 4 — Transitions posturales (proxy historique)")
    explain(
        "Tête binaire <code>FallRiskHead</code> : prédit si la <b>prochaine</b> activité "
        "est une transition posturale "
        f"(labels {sorted(FALL_RISK_LABELS)} : stand↔sit, sit↔lie).<br>"
        "Ce n'est pas la classe <code>fall</code> (absente du merge HAPT+WISDM)."
    )
    if not sys_data["has_fall_risk"] or sys_data["fall_risk_head"] is None:
        st.error("Checkpoint manquant : `checkpoints/fall_risk.pt` "
                 "(lancer `python scripts/train_fall_risk.py`)")
        st.stop()

    meta = sys_data.get("fall_risk_meta") or {}
    if meta.get("val_f1") is not None:
        st.caption(f"Checkpoint val F1 ≈ {meta['val_f1']:.3f} "
                   f"(p={meta.get('obs_ratio', '?')})")

    obs_pct = st.select_slider("Part du signal observé (%)",
                               options=[50, 75], value=75, key="fr_obs")
    n_samples = st.slider("Nombre d'exemples", 5, 20, 10, key="fr_n")
    seed = st.number_input("Seed", 0, 9999, 11, key="fr_seed")
    run = st.button("▶ Tester fall-risk", type="primary")

    if run:
        rows = run_demo_fall_risk(
            model, sys_data["fall_risk_head"], X, y, subjects,
            n_samples, seed, obs_pct / 100.0)
        correct = sum(r["correct"] for r in rows)
        m1, m2 = st.columns(2)
        m1.metric("Correct", f"{correct}/{len(rows)}")
        m2.metric("Signal observé", f"{obs_pct}%")
        st.dataframe(
            results_dataframe(rows)[
                ["Prochaine activité", "Risque réel", "Prédiction",
                 "Confiance risque", "Résultat"]
            ],
            use_container_width=True, hide_index=True,
        )

elif page == "5️⃣ Calibration + RL":
    st.markdown("## Démo 5 — Calibration online + bandit RL")
    explain(
        "<b>Calibration</b> : temperature scaling + seuil de confiance adaptatif "
        "(<code>OnlineCalibrator</code>).<br>"
        "<b>RL</b> : bandit ε-greedy qui choisit le seuil d'alerte "
        "(<code>ThresholdBandit</code>) avec reward +1 / −0.5 selon vrais/faux alertes."
    )
    if not sys_data["has_fall_risk"] or sys_data["fall_risk_head"] is None:
        st.error("Nécessite `fall_risk.pt`.")
        st.stop()

    n_steps = st.slider("Pas du flux (stream)", 20, 200, 80, key="rl_n")
    seed = st.number_input("Seed", 0, 9999, 3, key="rl_seed")
    run = st.button("▶ Simuler le flux online", type="primary")

    if run:
        history, bandit, calibrator = run_demo_calibration_rl(
            model, sys_data["fall_risk_head"], X, y, subjects,
            n_steps, seed, obs_ratio=0.75)
        st.success(bandit.summary())
        c1, c2 = st.columns(2)
        c1.metric("Temperature T", f"{calibrator.temp.temperature:.3f}")
        c2.metric("Seuil adaptatif", f"{calibrator.threshold.threshold:.3f}")
        st.dataframe(pd.DataFrame(history).tail(30),
                     use_container_width=True, hide_index=True)
        rewards = [float(h["Reward"]) for h in history]
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(np.cumsum(rewards), color="#0050a0")
        ax.set_title("Reward cumulée du bandit")
        ax.set_xlabel("Pas")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

elif page == "6️⃣ SSL / Foundation-style":
    st.markdown("## Démo 6 — SSL & pipeline foundation-style")
    explain(
        "<b>SSL</b> : prétrain sans labels (SimCLR-style IMU).<br>"
        "<b>Foundation-style</b> : SSL → fine-tune supervisé "
        "(pattern fondation à l'échelle de nos datasets — pas un FM public)."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("ssl_backbone.pt",
              "✓" if sys_data["has_ssl"] else "✗")
    c2.metric("foundation_style.pt",
              "✓" if sys_data["has_foundation"] else "✗")
    c3.metric("pretrained.pt (réf. HAR)", "✓")

    st.markdown("""
| Checkpoint | Rôle | Comment entraîner |
|------------|------|-------------------|
| `ssl_backbone.pt` | Backbone SSL seul | `scripts/train_ssl.py` |
| `foundation_style.pt` | SSL + FT (F1~0.47 run court) | `scripts/train_foundation_style.py` |
| `pretrained.pt` | HAR supervisé fort (prod démo) | `scripts/train.py --mode pretrain` |
    """)
    st.info(
        "Pour le jury : on montre que le **pattern** SSL→FT est implémenté. "
        "La démo live reconnaissance/anticipation utilise `pretrained.pt` "
        "(meilleures perfs)."
    )
    st.caption(meal_status())

    if sys_data["has_foundation"] and st.button(
            "▶ Tester foundation_style.pt (quelques fenêtres)", type="primary"):
        found = build_model(n_classes=int(y.max()) + 1)
        ck = torch.load(ROOT / "checkpoints" / "foundation_style.pt",
                        map_location="cpu", weights_only=False)
        found.load_state_dict(ck["model_state"], strict=False)
        found.eval()
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), 8, replace=False)
        rows = []
        with torch.no_grad():
            for i in idx:
                logits = found(torch.from_numpy(X[i:i+1].astype(np.float32)))
                pred = int(logits.argmax(-1).item())
                rows.append({
                    "Activité réelle": label_name(y[i]),
                    "Prédiction (FT)": label_name(pred),
                    "correct": pred == int(y[i]),
                })
        correct = sum(r["correct"] for r in rows)
        st.metric("Accuracy échantillon", f"{correct}/{len(rows)}")
        st.dataframe(results_dataframe(rows)[
            ["Activité réelle", "Prédiction (FT)", "Résultat"]],
            use_container_width=True, hide_index=True)

st.sidebar.markdown("---")
st.sidebar.caption("`streamlit run app_streamlit.py`")
