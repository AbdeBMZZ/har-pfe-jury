"""Parcours expérimental du pipeline HAR — présentation visuelle et académique."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RAW_CANDIDATES = [
    ROOT / "data" / "raw" / "hapt" / "RawData",
    ROOT.parent / "har_project" / "data" / "raw" / "hapt" / "RawData",
]
PROCESSED = ROOT / "data" / "processed"
STRICT = ROOT / "data" / "hapt_strict_v1"

STEPS = [
    {"key": "story", "title": "Objectifs du système"},
    {"key": "raw", "title": "1 · Données brutes IMU"},
    {"key": "clean", "title": "2 · Prétraitement du signal"},
    {"key": "windows", "title": "3 · Fenêtrage temporel"},
    {"key": "store", "title": "4 · Stockage processed"},
    {"key": "strict", "title": "5 · Protocole d'évaluation strict"},
    {"key": "model", "title": "6 · Architecture du modèle"},
    {"key": "reco", "title": "7 · Reconnaissance d'activité"},
    {"key": "cl", "title": "8 · Apprentissage continu"},
    {"key": "ant", "title": "9 · Anticipation"},
    {"key": "scores", "title": "10 · Résultats expérimentaux"},
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.block-container { padding-top: 1rem !important; max-width: 1100px; }
.k { font-family:'IBM Plex Sans',sans-serif; font-size:.72rem; letter-spacing:.12em;
  text-transform:uppercase; color:#64748b; }
.t { font-family:'DM Sans',sans-serif; font-size:1.7rem; font-weight:700;
  color:#0b3d6e; margin:.1rem 0 .65rem 0; }
.bar-wrap { height:7px; background:#e2e8f0; border-radius:999px; overflow:hidden; margin-bottom:.8rem; }
.bar { height:100%; background:linear-gradient(90deg,#0b3d6e,#0f766e); }
.h { font-family:'DM Sans',sans-serif; font-size:1.05rem; font-weight:700; color:#0b3d6e;
  margin:1rem 0 .35rem 0; }
.p { font-family:'IBM Plex Sans',sans-serif; font-size:1.02rem; color:#1e293b;
  line-height:1.6; margin:0 0 .55rem 0; }
.idea {
  background: linear-gradient(90deg,#eff6ff,#f0fdfa);
  border: 1px solid #bfdbfe; border-radius: 14px;
  padding: 14px 16px; margin: .5rem 0 .9rem 0;
  font-family:'IBM Plex Sans',sans-serif; font-size:1.02rem; color:#0f172a; line-height:1.55;
}
.idea b { color:#0b3d6e; }
.simple {
  background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
  padding:14px 16px; margin:.4rem 0 .85rem 0;
  font-family:'IBM Plex Sans',sans-serif; font-size:.98rem; color:#334155; line-height:1.55;
}
.path { font-family:ui-monospace,Menlo,monospace; font-size:.74rem; background:#f1f5f9;
  color:#64748b; padding:8px 10px; border-radius:8px; word-break:break-all; margin:.3rem 0 .8rem 0; }
.flow { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:.7rem 0 1rem 0; }
.node { background:#fff; border:1.5px solid #93c5fd; border-radius:999px; padding:8px 14px;
  font-family:'DM Sans',sans-serif; font-weight:600; color:#0b3d6e; font-size:.9rem; }
.ar { color:#94a3b8; font-weight:700; }
.arch { display:grid; grid-template-columns:1fr 30px 1.15fr 30px 1fr; gap:8px; margin:.7rem 0; }
.box { border-radius:14px; padding:16px 10px; text-align:center; border:2px solid #cbd5e1;
  background:#fff; color:#0b3d6e; font-family:'DM Sans',sans-serif; min-height:90px;
  display:flex; flex-direction:column; justify-content:center; }
.box strong { font-size:1rem; } .box span { font-size:.8rem; color:#64748b; margin-top:5px; }
.box.mid { background:#0b3d6e; color:#fff; border-color:#0b3d6e; }
.box.mid span { color:#cbd5e1; }
.box.out { background:#ecfdf5; border-color:#5eead4; color:#115e59; }
.box.out span { color:#0f766e; }
.aa { display:flex; align-items:center; justify-content:center; font-size:1.35rem; color:#94a3b8; }
.sg { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:.5rem 0 .9rem 0; }
.sc { border-radius:16px; padding:20px 12px; text-align:center; color:#fff; font-family:'DM Sans',sans-serif; }
.sc .v { font-size:2rem; font-weight:700; } .sc .l { margin-top:8px; font-size:.9rem; opacity:.92; }
.sc.a{background:#0b3d6e;} .sc.b{background:#0f766e;} .sc.c{background:#334155;}
.schema {
  background:#f8fafc; border:1px solid #e2e8f0; border-radius:16px;
  padding:18px 16px; margin:.55rem 0 1rem 0;
}
.schema-title {
  font-family:'DM Sans',sans-serif; font-size:.78rem; letter-spacing:.1em;
  text-transform:uppercase; color:#64748b; margin:0 0 12px 0; text-align:center;
}
.schema-row { display:flex; flex-wrap:wrap; gap:10px; align-items:stretch; justify-content:center; }
.schema-card {
  flex:1; min-width:140px; max-width:220px;
  background:#fff; border:2px solid #bfdbfe; border-radius:14px;
  padding:14px 12px; text-align:center;
  font-family:'DM Sans',sans-serif; color:#0b3d6e;
}
.schema-card .ic { font-size:1.6rem; margin-bottom:6px; }
.schema-card strong { display:block; font-size:1rem; margin-bottom:4px; }
.schema-card span { display:block; font-size:.82rem; color:#64748b; font-weight:500; line-height:1.35; }
.schema-card.acc { border-color:#93c5fd; background:#eff6ff; }
.schema-card.gyro { border-color:#6ee7b7; background:#ecfdf5; }
.schema-card.lab { border-color:#fdba74; background:#fff7ed; }
.schema-plus {
  display:flex; align-items:center; justify-content:center;
  font-family:'DM Sans',sans-serif; font-size:1.5rem; font-weight:700; color:#94a3b8;
  min-width:28px;
}
.schema-eq {
  margin-top:14px; text-align:center; font-family:'DM Sans',sans-serif;
  font-weight:700; color:#0b3d6e; font-size:1.05rem;
}
.schema-eq span { display:inline-block; background:#0b3d6e; color:#fff;
  padding:8px 14px; border-radius:999px; margin-left:6px; }
.chip-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:.4rem 0 .2rem 0; }
.chip {
  background:#fff; border:1px solid #e2e8f0; border-radius:10px;
  padding:10px 8px; text-align:center; font-family:'IBM Plex Sans',sans-serif;
  font-size:.82rem; color:#334155;
}
.chip b { display:block; color:#0b3d6e; font-family:'DM Sans',sans-serif; font-size:.9rem; margin-bottom:2px; }
.timeline {
  display:grid; grid-template-columns: 90px 1fr 1fr 1fr 1fr;
  gap:6px; align-items:center; margin:.5rem 0;
  font-family:'IBM Plex Sans',sans-serif; font-size:.85rem;
}
.timeline .hd { font-family:'DM Sans',sans-serif; font-size:.72rem; color:#64748b;
  text-transform:uppercase; letter-spacing:.06em; text-align:center; }
.timeline .tag {
  background:#0b3d6e; color:#fff; border-radius:8px; padding:8px 6px; text-align:center; font-weight:600;
}
.timeline .cell {
  background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:8px 6px; text-align:center; color:#334155;
}
.code-ref {
  font-family: ui-monospace, Menlo, Consolas, monospace;
  font-size: .78rem;
  line-height: 1.55;
  color: #334155;
  background: #f1f5f9;
  border-left: 4px solid #2563eb;
  border-radius: 0 10px 10px 0;
  padding: 10px 14px;
  margin: .55rem 0 1rem 0;
}
.code-ref-t {
  font-family: 'DM Sans', sans-serif;
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #2563eb;
  margin-bottom: 6px;
}
.code-ref code {
  background: #e2e8f0;
  color: #0f172a;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: .76rem;
}
</style>
"""


def _find_raw():
    for p in RAW_CANDIDATES:
        if (p / "labels.txt").exists():
            return p
    return None


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linestyle="--")
    ax.tick_params(labelsize=9)


def _plot_raw(acc, gyro, n=400):
    t = np.arange(n) / 50.0
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 3.5), sharex=True, facecolor="white")
    cols = ["#0b3d6e", "#0f766e", "#c2410c"]
    for i, lab in enumerate("xyz"):
        axes[0].plot(t, acc[:n, i], color=cols[i], lw=1.2, label=lab)
        axes[1].plot(t, gyro[:n, i], color=cols[i], lw=1.2, label=lab)
    axes[0].set_ylabel("Accélération\n(m/s²)")
    axes[1].set_ylabel("Gyroscope\n(rad/s)")
    axes[1].set_xlabel("Temps (s)")
    axes[0].legend(ncols=3, frameon=False, fontsize=9, loc="upper right")
    for ax in axes:
        _style(ax)
    fig.suptitle("Signaux IMU bruts — extrait de 8 s (exp01, user01)", color="#0b3d6e", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def _plot_windows():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 500)
    sig = np.sin(2 * np.pi * 0.35 * t) + 0.12 * rng.normal(size=t.size)
    fig, ax = plt.subplots(figsize=(10.5, 2.5), facecolor="white")
    ax.plot(t, sig, color="#cbd5e1", lw=1.1)
    for start, color, name in [
        (0, "#0b3d6e", "fenêtre 1"),
        (1.5, "#0f766e", "fenêtre 2"),
        (3, "#c2410c", "fenêtre 3"),
    ]:
        m = (t >= start) & (t < start + 3)
        ax.fill_between(t[m], -1.8, 1.8, color=color, alpha=0.16)
        ax.plot(t[m], sig[m], color=color, lw=2)
        ax.text(start + 1.5, 1.5, name, ha="center", color=color, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 8)
    ax.set_ylim(-2, 2)
    ax.set_yticks([])
    ax.set_xlabel("Temps (s)")
    _style(ax)
    ax.set_title("Fenêtrage glissant — durée 3 s, pas de 1,5 s (chevauchement 50 %)", color="#0b3d6e", fontsize=12)
    plt.tight_layout()
    return fig


def _plot_window(window, title):
    t = np.arange(window.shape[0]) / 50.0
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 3.3), sharex=True, facecolor="white")
    cols = ["#0b3d6e", "#0f766e", "#c2410c"]
    for i, lab in enumerate("xyz"):
        axes[0].plot(t, window[:, i], color=cols[i], lw=1.25, label=lab)
        axes[1].plot(t, window[:, i + 3], color=cols[i], lw=1.25, label=lab)
    axes[0].set_ylabel("Accélération")
    axes[1].set_ylabel("Gyroscope")
    axes[1].set_xlabel("Temps (s)")
    axes[0].legend(ncols=3, frameon=False, fontsize=9, loc="upper right")
    for ax in axes:
        _style(ax)
    fig.suptitle(title, color="#0b3d6e", fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def _go(step: int):
    st.session_state.pipe_step = max(0, min(int(step), len(STEPS) - 1))


def _nav(idx, *, where):
    left, mid, right = st.columns([1, 2.2, 1])
    with left:
        if st.button("← Précédent", use_container_width=True, disabled=idx == 0, key=f"pprev_{where}"):
            _go(idx - 1)
            st.rerun()
    with mid:
        st.caption(f"{idx + 1} / {len(STEPS)}")
    with right:
        if st.button(
            "Suivant →",
            use_container_width=True,
            type="primary",
            disabled=idx >= len(STEPS) - 1,
            key=f"pnext_{where}",
        ):
            _go(idx + 1)
            st.rerun()


def _on_jump_change():
    _go(st.session_state.pipe_jump)


def _h(t):
    st.markdown(f'<div class="h">{t}</div>', unsafe_allow_html=True)


def _p(t):
    st.markdown(f'<p class="p">{t}</p>', unsafe_allow_html=True)


def _idea(t):
    st.markdown(f'<div class="idea">{t}</div>', unsafe_allow_html=True)


def _simple(t):
    st.markdown(f'<div class="simple">{t}</div>', unsafe_allow_html=True)


def _code(funcs_html: str):
    st.markdown(
        f'<div class="code-ref"><div class="code-ref-t">Implémentation</div>{funcs_html}</div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if "pipe_step" not in st.session_state:
        st.session_state.pipe_step = 0
    # Sync jump dropdown BEFORE the widget exists (never after).
    st.session_state.pipe_jump = int(st.session_state.pipe_step)

    with st.expander("Aller à une étape", expanded=False):
        st.selectbox(
            "Étape",
            options=list(range(len(STEPS))),
            format_func=lambda i: STEPS[i]["title"],
            key="pipe_jump",
            on_change=_on_jump_change,
            label_visibility="collapsed",
        )

    idx = max(0, min(int(st.session_state.pipe_step), len(STEPS) - 1))
    st.session_state.pipe_step = idx
    step = STEPS[idx]
    pct = int(100 * (idx + 1) / len(STEPS))

    st.markdown(
        '<div class="k">Pipeline HAR · présentation du parcours expérimental</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="t">{step["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bar-wrap"><div class="bar" style="width:{pct}%"></div></div>', unsafe_allow_html=True)
    _nav(idx, where="top")

    key = step["key"]
    raw = _find_raw()

    if key == "story":
        _idea(
            "<b>Objectif :</b> à partir de mesures inertielles (montre ou téléphone), "
            "le système reconnaît <i>l'activité en cours</i>, "
            "s'adapte à <i>un nouvel utilisateur</i>, "
            "et peut <i>anticiper l'activité future</i>."
        )
        _h("Trois capacités")
        _simple(
            "<b>1. Reconnaissance</b> — identification de l'activité courante (ex. marche).<br>"
            "<b>2. Adaptation</b> — intégration d'un nouvel utilisateur sans réentraînement complet.<br>"
            "<b>3. Anticipation</b> — prédiction de l'activité à venir à partir du contexte observé."
        )
        st.markdown(
            """
<div class="flow">
  <div class="node">Capteur IMU</div><div class="ar">→</div>
  <div class="node">Prétraitement</div><div class="ar">→</div>
  <div class="node">Fenêtres</div><div class="ar">→</div>
  <div class="node">Modèle</div><div class="ar">→</div>
  <div class="node">Décision</div>
</div>
""",
            unsafe_allow_html=True,
        )
        _code("<code>build_model</code> · <code>src/models/har_model.py</code>")
        _p("Les étapes suivantes détaillent ce parcours dans l'ordre expérimental.")

    elif key == "raw":
        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Chaîne d'acquisition</div>
  <div class="schema-row">
    <div class="schema-card"><div class="ic">👤</div><strong>Sujet</strong><span>activités de la vie quotidienne</span></div>
    <div class="schema-plus">→</div>
    <div class="schema-card"><div class="ic">📱</div><strong>IMU</strong><span>accéléromètre + gyroscope</span></div>
    <div class="schema-plus">→</div>
    <div class="schema-card"><div class="ic">📊</div><strong>Séries temporelles</strong><span>+ annotations d'activité</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Six canaux inertiels</div>
  <div class="schema-row">
    <div class="schema-card acc">
      <div class="ic">↕️</div>
      <strong>Accélération ×3</strong>
      <span>axes X, Y, Z<br>translations<br><b>discriminant pour la locomotion</b></span>
    </div>
    <div class="schema-plus">+</div>
    <div class="schema-card gyro">
      <div class="ic">🔄</div>
      <strong>Gyroscope ×3</strong>
      <span>axes X, Y, Z<br>rotations<br><b>discriminant pour les transitions</b></span>
    </div>
  </div>
  <div class="schema-eq">concaténation → <span>(T × 6)</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Fréquence d'échantillonnage</div>
  <div class="schema-row">
    <div class="schema-card lab">
      <strong>10 Hz</strong>
      <span>insuffisant<br>transitions mal résolues</span>
    </div>
    <div class="schema-card" style="border-color:#0b3d6e;background:#0b3d6e;color:#fff">
      <strong style="color:#fff">50 Hz ✓</strong>
      <span style="color:#dbeafe">1 échantillon / 20 ms<br>adapté à marche & postures<br>standard HAPT</span>
    </div>
    <div class="schema-card lab">
      <strong>200 Hz</strong>
      <span>coût élevé<br>gain limité ici</span>
    </div>
  </div>
  <div class="schema-eq">bande utile ≈ <span>10–20 Hz</span> → 50 Hz suffit</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Fichiers bruts HAPT</div>
  <div class="schema-row">
    <div class="schema-card acc"><strong>acc_….txt</strong><span>accélération x y z</span></div>
    <div class="schema-card gyro"><strong>gyro_….txt</strong><span>rotation x y z</span></div>
    <div class="schema-card lab"><strong>labels.txt</strong><span>activité [start → end]</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Structure d'une annotation (labels.txt)</div>
  <div class="timeline">
    <div class="hd"></div><div class="hd">exp</div><div class="hd">user</div><div class="hd">activité</div><div class="hd">intervalle</div>
    <div class="tag">ex.</div>
    <div class="cell">1</div><div class="cell">1</div>
    <div class="cell">5 = debout</div><div class="cell">250 → 1232</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Douze classes HAPT</div>
  <div class="chip-grid">
    <div class="chip"><b>1 · 2 · 3</b>marche / escaliers ↑ ↓</div>
    <div class="chip"><b>4 · 5 · 6</b>assis / debout / allongé</div>
    <div class="chip"><b>7 → 12</b>transitions posturales</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        _code(
            "<code>load_hapt()</code> · <code>src/data/dataset_loaders.py</code><br>"
            "<code>scripts/fetch_hapt_raw.py</code>"
        )

        if raw is None:
            st.warning("Fichiers bruts introuvables dans les chemins candidats.")
        else:
            acc_path = raw / "acc_exp01_user01.txt"
            gyro_path = raw / "gyro_exp01_user01.txt"
            labels_path = raw / "labels.txt"
            acc_lines = acc_path.read_text().splitlines()[:5]
            gyro_lines = gyro_path.read_text().splitlines()[:5]
            label_lines = labels_path.read_text().splitlines()[:5]

            _h("Extraits des fichiers bruts")
            c_acc, c_gyro, c_lab = st.columns(3)
            with c_acc:
                st.caption("acc_exp01_user01.txt — 5 premières lignes")
                st.code("\n".join(acc_lines), language=None)
            with c_gyro:
                st.caption("gyro_exp01_user01.txt — 5 premières lignes")
                st.code("\n".join(gyro_lines), language=None)
            with c_lab:
                st.caption("labels.txt — 5 premières lignes")
                st.code("\n".join(label_lines), language=None)

            acc = np.loadtxt(acc_path)
            gyro = np.loadtxt(gyro_path)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fréquence", "50 Hz")
            c2.metric("Canaux", "6")
            c3.metric("Durée (exp01)", f"{len(acc)/50:.0f} s")
            c4.metric("Échantillons", f"{len(acc):,}")
            st.pyplot(_plot_raw(acc, gyro), clear_figure=True)
            st.markdown('<div class="path">data/raw/hapt/RawData/</div>', unsafe_allow_html=True)

    elif key == "clean":
        _idea(
            "<b>Principe :</b> avant l'apprentissage, le signal est homogénéisé et filtré "
            "afin de conserver le mouvement utile et d'atténuer les composantes parasites "
            "(gravité, dérives, hétérogénéité des unités)."
        )
        _h("Opérations de prétraitement")
        _simple(
            "<b>1. Homogénéisation</b> — unités et fréquence unifiées (50 Hz) pour fusionner plusieurs sources.<br>"
            "<b>2. Filtrage</b> — atténuation des composantes très lentes (ex. gravité) "
            "pour mettre en évidence le geste dynamique.<br>"
            "<b>3. Filtrage causal</b> — le filtre n'utilise que le passé "
            "(condition nécessaire pour une anticipation sans fuite temporelle)."
        )
        _code(
            "<code>preprocess_signal</code>, <code>remove_gravity</code>, "
            "<code>resample_signal</code>, <code>convert_g_to_ms2</code> · "
            "<code>src/data/preprocessing.py</code><br>"
            "<code>homogenize_sample</code> · <code>src/data/homogenization.py</code><br>"
            "<code>causal_imu</code> · <code>src/data/temporal_protocol.py</code>"
        )
        _p("Résultat : un signal aligné, prêt pour le fenêtrage.")

    elif key == "windows":
        _idea(
            "<b>Principe :</b> une série longue n'est pas présentée en une seule pièce au modèle. "
            "Elle est découpée en <b>fenêtres de 3 s</b> (150 échantillons × 6 canaux), "
            "chacune associée à une étiquette d'activité."
        )
        _h("Choix de la durée")
        _simple(
            "Assez longue pour caractériser une activité stable (ex. marche).<br>"
            "Assez courte pour limiter le mélange d'activités distinctes dans une même fenêtre."
        )
        _h("Chevauchement")
        _simple(
            "Sans chevauchement, une transition peut tomber entre deux fenêtres. "
            "Un pas de <b>1,5 s</b> (chevauchement 50 %) améliore la couverture des changements d'état."
        )
        st.pyplot(_plot_windows(), clear_figure=True)
        _code(
            "<code>sliding_windows_with_labels</code> · <code>src/data/preprocessing.py</code><br>"
            "<code>recording_windows</code> · <code>src/data/temporal_protocol.py</code>"
        )
        _p("Chaque fenêtre constitue un exemple d'apprentissage : signal + label.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Durée", "3 s")
        c2.metric("Forme", "150 × 6")
        c3.metric("Pas", "1,5 s")

    elif key == "store":
        _idea(
            "<b>Principe :</b> après prétraitement et fenêtrage, les tenseurs sont sérialisés "
            "dans <code>data/processed/</code> pour un chargement rapide (entraînement, évaluation, démonstration)."
        )
        _h("Contenu de data/processed/")
        _simple(
            "<b>X.npy</b> — fenêtres de mouvement.<br>"
            "<b>y.npy</b> — labels d'activité.<br>"
            "<b>subjects.npy</b> — identifiants des sujets.<br>"
            "<b>origins.npy</b> — jeu de données d'origine.<br>"
            "<b>subject_meta.json</b> — métadonnées des sujets."
        )
        _code(
            "<code>build_unified_dataset</code>, <code>save_processed</code>, "
            "<code>load_processed</code> · <code>src/data/homogenization.py</code><br>"
            "<code>scripts/preprocess.py</code>"
        )
        _p("Intérêt : relancer une expérience sans recalculer toute la chaîne depuis les fichiers texte.")
        if (PROCESSED / "X.npy").exists():
            X = np.load(PROCESSED / "X.npy", mmap_mode="r")
            y = np.load(PROCESSED / "y.npy")
            subjects = np.load(PROCESSED / "subjects.npy")
            origins = np.load(PROCESSED / "origins.npy")
            left, right = st.columns([1.55, 1])
            with left:
                st.pyplot(
                    _plot_window(X[12], "Exemple de fenêtre stockée (3 s, 6 canaux)"),
                    clear_figure=True,
                )
            with right:
                st.metric("Fenêtres", f"{len(X):,}")
                st.metric("Sujets", f"{len(np.unique(subjects))}")
                uniq, counts = np.unique(origins.astype(str), return_counts=True)
                st.bar_chart({"fenêtres": dict(zip(uniq.tolist(), [int(c) for c in counts]))}, height=160)
            st.markdown('<div class="path">data/processed/</div>', unsafe_allow_html=True)

    elif key == "strict":
        _idea(
            "<b>Principe :</b> l'évaluation sépare strictement les sujets "
            "(entraînement / validation / test) et impose un traitement causal, "
            "afin d'éviter la fuite d'information et de garantir des scores reproductibles."
        )
        _h("Garanties du protocole")
        _simple(
            "<b>Partition par sujet</b> — les personnes du test n'apparaissent pas en entraînement.<br>"
            "<b>Filtrage causal</b> — aucune information future dans le prétraitement.<br>"
            "<b>Ordre temporel</b> — pour l'anticipation, la cible est effectivement postérieure au contexte.<br>"
            "<b>Empreinte du protocole</b> — chaque checkpoint est lié aux règles d'évaluation utilisées."
        )
        _code(
            "<code>scripts/prepare_hapt_protocol.py</code><br>"
            "<code>save_protocol</code>, <code>load_protocol</code>, "
            "<code>subject_partition</code>, <code>validate_checkpoint</code> · "
            "<code>src/data/temporal_protocol.py</code>"
        )
        if STRICT.exists():
            proto = json.loads((STRICT / "protocol.json").read_text())
            Xs = np.load(STRICT / "X.npy", mmap_mode="r")
            splits = proto.get("splits", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fenêtres", f"{Xs.shape[0]:,}")
            c2.metric("Sujets train", len(splits.get("train", [])))
            c3.metric("Sujets val", len(splits.get("validation", [])))
            c4.metric("Sujets test", len(splits.get("test", [])))
            st.markdown('<div class="path">data/hapt_strict_v1/</div>', unsafe_allow_html=True)
        _p("Ce protocole sert aux scores officiels de reconnaissance et d'anticipation.")

    elif key == "model":
        _idea(
            "<b>Architecture :</b> un encodeur transforme chaque fenêtre en représentation latente ; "
            "des têtes spécialisées produisent la reconnaissance d'activité et, le cas échéant, l'anticipation."
        )
        st.markdown(
            """
<div class="arch">
  <div class="box"><strong>Fenêtre 3 s</strong><span>signal IMU (T×6)</span></div>
  <div class="aa">→</div>
  <div class="box mid"><strong>Encodeur</strong><span>représentation latente</span></div>
  <div class="aa">→</div>
  <div>
    <div class="box out" style="min-height:42px;margin-bottom:8px"><strong>Reconnaissance</strong><span>activité courante</span></div>
    <div class="box out" style="min-height:42px"><strong>Anticipation</strong><span>activité future</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        _h("Deux mécanismes de décision")
        _simple(
            "<b>Classifieur softmax</b> — scores de classes appris pendant le pré-entraînement.<br>"
            "<b>Mémoire de prototypes</b> — un centroïde par activité ; "
            "la décision suit la similarité au prototype le plus proche. "
            "Utile pour intégrer un nouvel utilisateur sans reconstruire tout le modèle."
        )
        _code(
            "<code>build_model</code>, <code>HARContinualModel</code> · "
            "<code>src/models/har_model.py</code><br>"
            "<code>IMUTransformerEncoder</code>, <code>ContinualHARHead</code>, "
            "<code>AnticipationHead</code> · <code>src/models/backbone.py</code>"
        )

    elif key == "reco":
        _idea(
            "<b>Tâche :</b> à partir d'une fenêtre de 3 s, "
            "prédire l'activité réalisée par le sujet à l'instant considéré."
        )
        _h("Pipeline d'inférence")
        _simple(
            "1. Chargement d'une fenêtre prétraitée.<br>"
            "2. Encodage en représentation compacte.<br>"
            "3. Décision via softmax ou <code>PrototypeMemory.predict</code>.<br>"
            "4. Affichage de l'activité la plus probable."
        )
        _code(
            "<code>pretrain</code> · <code>src/training/trainer.py</code><br>"
            "<code>scripts/run_hapt_protocol.py --stage recognition</code><br>"
            "<code>PrototypeMemory.predict</code>"
        )
        _h("Performances sur le jeu de test")
        st.markdown(
            """
<div class="sg">
  <div class="sc a"><div class="v">85,9%</div><div class="l">Accuracy</div></div>
  <div class="sc b"><div class="v">72,6%</div><div class="l">F1 macro</div></div>
  <div class="sc c"><div class="v">Live</div><div class="l">Page Reconnaissance</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        _simple(
            "<b>Accuracy 85,9 %</b> — proportion de fenêtres correctement classées.<br>"
            "<b>F1 macro 72,6 %</b> — moyenne des F1 par classe "
            "(plus exigeante lorsque certaines activités sont rares)."
        )

    elif key == "cl":
        _idea(
            "<b>Contexte :</b> le modèle a été entraîné sur un ensemble de sujets. "
            "Un nouvel utilisateur arrive avec un style moteur différent. "
            "L'objectif est de l'intégrer sans effacer les connaissances antérieures."
        )
        _h("Écueil à éviter")
        _simple(
            "Réentraîner uniquement sur le nouvel utilisateur conduit à un "
            "<b>oubli catastrophique</b> des sujets précédents."
        )
        _h("Stratégie retenue")
        _simple(
            "Conservation de l'encodeur (compréhension des gestes).<br>"
            "Mise à jour de la <b>mémoire de prototypes</b> avec quelques exemples du nouvel utilisateur.<br>"
            "Rejeu d'exemples antérieurs (<b>ReplayBuffer</b>) pour stabiliser les performances globales."
        )
        st.markdown(
            """
<div class="arch">
  <div class="box"><strong>Sujets connus</strong><span>connaissances antérieures</span></div>
  <div class="aa">+</div>
  <div class="box mid"><strong>Mise à jour</strong><span>prototypes + rejeu</span></div>
  <div class="aa">→</div>
  <div class="box out"><strong>Nouvel utilisateur</strong><span>ancien savoir conservé</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        _code(
            "<code>continual_train</code> · <code>src/training/trainer.py</code><br>"
            "<code>PrototypeMemory.update</code> · <code>ReplayBuffer</code><br>"
            "<code>model.continual_step</code>"
        )
        st.caption("Démonstration : page « Nouvel utilisateur (Karim) ».")

    elif key == "ant":
        _idea(
            "<b>Tâche :</b> à partir d'un contexte partiel (début du mouvement), "
            "prédire l'activité qui suivra — sans accès au futur (filtrage causal)."
        )
        _h("Difficultés")
        _simple(
            "Information partielle → incertitude plus élevée.<br>"
            "Les transitions posturales sont <b>rares</b> et proches entre elles → "
            "taux d'erreur plus important sur ces classes."
        )
        _h("Granularité des classes")
        _simple(
            "<b>12 classes</b> — distinction fine des transitions "
            "(ex. assis→allongé vs assis→debout). Tâche difficile ; F1 macro plus basse (~52 %).<br><br>"
            "<b>7 classes</b> — six activités stables + une classe « transition » regroupée. "
            "Plus robuste avec peu d'exemples ; meilleur équilibre (~77 %)."
        )
        st.markdown(
            """
<div class="sg">
  <div class="sc a"><div class="v">71,8%</div><div class="l">Accuracy anticipation (7 classes)</div></div>
  <div class="sc b"><div class="v">76,6%</div><div class="l">F1 macro anticipation (7 classes)</div></div>
  <div class="sc c"><div class="v">51,6%</div><div class="l">Meilleur F1 macro (12 classes)</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        _code(
            "<code>TemporalAnticipationDataset</code> · <code>src/data/temporal_protocol.py</code><br>"
            "<code>model.anticipate</code><br>"
            "<code>scripts/train_anticipation_7class.py</code><br>"
            "<code>anticipation_pipeline.py</code> / <code>CausalForecaster</code>"
        )
        st.caption("Démonstration : page Anticipation.")

    else:  # scores
        _idea(
            "<b>Synthèse :</b> la reconnaissance de l'activité courante atteint un niveau élevé ; "
            "l'anticipation reste crédible, surtout lorsque les transitions rares "
            "sont regroupées (protocole à 7 classes)."
        )
        st.markdown(
            """
<div class="sg">
  <div class="sc a"><div class="v">85,9%</div><div class="l">Reconnaissance (accuracy)</div></div>
  <div class="sc b"><div class="v">76,6%</div><div class="l">Anticipation F1 (7 classes)</div></div>
  <div class="sc c"><div class="v">71,4%</div><div class="l">Autre modèle d'anticipation</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        _h("Parcours expérimental (résumé)")
        st.markdown(
            """
<div class="flow">
  <div class="node">IMU</div><div class="ar">→</div>
  <div class="node">Bruts</div><div class="ar">→</div>
  <div class="node">Prétraitement</div><div class="ar">→</div>
  <div class="node">Fenêtres 3 s</div><div class="ar">→</div>
  <div class="node">Processed</div><div class="ar">→</div>
  <div class="node">Modèle</div><div class="ar">→</div>
  <div class="node">Reco / CL / Ant.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        _code(
            "Résultats JSON (checkpoints / logs)<br>"
            "<code>accuracy_score</code> / <code>f1_score</code> (sklearn)"
        )
        _simple(
            "Poursuite dans le menu : <b>Reconnaissance</b>, puis <b>Karim</b>, puis <b>Anticipation</b>."
        )

    st.markdown("---")
    _nav(idx, where="bottom")
