"""Plain-language walkthrough of the full HAR pipeline."""
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
    {"key": "story", "title": "De quoi parle ce projet ?"},
    {"key": "raw", "title": "1 · Données brutes IMU"},
    {"key": "clean", "title": "2 · Nettoyer le signal"},
    {"key": "windows", "title": "3 · Couper en petits morceaux"},
    {"key": "store", "title": "4 · Où on range tout ça"},
    {"key": "strict", "title": "5 · Règles d’évaluation propres"},
    {"key": "model", "title": "6 · Le « cerveau » du système"},
    {"key": "reco", "title": "7 · Reconnaître l’activité"},
    {"key": "cl", "title": "8 · Apprendre un nouvel utilisateur"},
    {"key": "ant", "title": "9 · Anticiper la suite"},
    {"key": "scores", "title": "10 · Les scores obtenus"},
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
    axes[0].set_ylabel("Mouvement\n(accélération)")
    axes[1].set_ylabel("Rotation\n(gyroscope)")
    axes[1].set_xlabel("Temps (secondes)")
    axes[0].legend(ncols=3, frameon=False, fontsize=9, loc="upper right")
    for ax in axes:
        _style(ax)
    fig.suptitle("Ce que le téléphone enregistre pendant 8 secondes", color="#0b3d6e", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def _plot_windows():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 500)
    sig = np.sin(2 * np.pi * 0.35 * t) + 0.12 * rng.normal(size=t.size)
    fig, ax = plt.subplots(figsize=(10.5, 2.5), facecolor="white")
    ax.plot(t, sig, color="#cbd5e1", lw=1.1)
    for start, color, name in [(0, "#0b3d6e", "morceau 1"), (1.5, "#0f766e", "morceau 2"), (3, "#c2410c", "morceau 3")]:
        m = (t >= start) & (t < start + 3)
        ax.fill_between(t[m], -1.8, 1.8, color=color, alpha=0.16)
        ax.plot(t[m], sig[m], color=color, lw=2)
        ax.text(start + 1.5, 1.5, name, ha="center", color=color, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 8)
    ax.set_ylim(-2, 2)
    ax.set_yticks([])
    ax.set_xlabel("Temps (secondes)")
    _style(ax)
    ax.set_title("On découpe le long enregistrement en morceaux de 3 secondes", color="#0b3d6e", fontsize=12)
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

    st.markdown('<div class="k">Comprendre le système · étape par étape</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="t">{step["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bar-wrap"><div class="bar" style="width:{pct}%"></div></div>', unsafe_allow_html=True)
    _nav(idx, where="top")

    key = step["key"]
    raw = _find_raw()

    if key == "story":
        _idea(
            "<b>En une phrase :</b> une montre ou un téléphone mesure les mouvements. "
            "Notre programme apprend à dire <i>ce que la personne fait</i>, "
            "à s’adapter à <i>une nouvelle personne</i>, "
            "et parfois à <i>deviner ce qui va suivre</i>."
        )
        _h("Les 3 capacités")
        _simple(
            "<b>1. Reconnaître</b> — « là, la personne marche ».<br>"
            "<b>2. S’adapter</b> — un nouvel utilisateur arrive ; on apprend son style sans tout recommencer.<br>"
            "<b>3. Anticiper</b> — « avec ce qu’on voit maintenant, la suite ressemble à… »."
        )
        st.markdown(
            """
<div class="flow">
  <div class="node">Mouvement</div><div class="ar">→</div>
  <div class="node">Nettoyage</div><div class="ar">→</div>
  <div class="node">Petits morceaux</div><div class="ar">→</div>
  <div class="node">Modèle</div><div class="ar">→</div>
  <div class="node">Réponse</div>
</div>
""",
            unsafe_allow_html=True,
        )
        _p("Les prochaines étapes suivent exactement ce chemin, dans l’ordre.")

    elif key == "raw":
        # —— Schéma 1 : capture ——
        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Ce qu’on capture</div>
  <div class="schema-row">
    <div class="schema-card"><div class="ic">👤</div><strong>Personne</strong><span>activités du quotidien</span></div>
    <div class="schema-plus">→</div>
    <div class="schema-card"><div class="ic">📱</div><strong>IMU</strong><span>acc + gyro</span></div>
    <div class="schema-plus">→</div>
    <div class="schema-card"><div class="ic">📊</div><strong>Séries temporelles</strong><span>+ labels d’activité</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # —— Schéma 2 : 6 canaux ——
        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Pourquoi 6 canaux ?</div>
  <div class="schema-row">
    <div class="schema-card acc">
      <div class="ic">↕️</div>
      <strong>Accélération ×3</strong>
      <span>X Y Z<br>translations / secousses<br><b>utile pour la marche</b></span>
    </div>
    <div class="schema-plus">+</div>
    <div class="schema-card gyro">
      <div class="ic">🔄</div>
      <strong>Gyroscope ×3</strong>
      <span>X Y Z<br>rotations<br><b>utile pour les transitions</b></span>
    </div>
  </div>
  <div class="schema-eq">on colle les deux → <span>(temps × 6)</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

        # —— Schéma 3 : 50 Hz ——
        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Pourquoi 50 mesures / seconde (50 Hz) ?</div>
  <div class="schema-row">
    <div class="schema-card lab">
      <strong>10 Hz</strong>
      <span>trop lent<br>on rate les transitions</span>
    </div>
    <div class="schema-card" style="border-color:#0b3d6e;background:#0b3d6e;color:#fff">
      <strong style="color:#fff">50 Hz ✓</strong>
      <span style="color:#dbeafe">1 mesure / 20 ms<br>assez pour marche & postures<br>standard HAPT</span>
    </div>
    <div class="schema-card lab">
      <strong>200 Hz</strong>
      <span>trop lourd<br>peu de gain ici</span>
    </div>
  </div>
  <div class="schema-eq">mouvements utiles ≈ <span>10–20 Hz</span> → 50 Hz suffit</div>
</div>
""",
            unsafe_allow_html=True,
        )

        # —— Schéma 4 : fichiers ——
        st.markdown(
            """
<div class="schema">
  <div class="schema-title">3 fichiers bruts</div>
  <div class="schema-row">
    <div class="schema-card acc"><strong>acc_….txt</strong><span>mouvement x y z</span></div>
    <div class="schema-card gyro"><strong>gyro_….txt</strong><span>rotation x y z</span></div>
    <div class="schema-card lab"><strong>labels.txt</strong><span>activité entre start → end</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # —— Schéma 5 : label row ——
        st.markdown(
            """
<div class="schema">
  <div class="schema-title">Lecture d’une ligne de labels.txt</div>
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

        # —— Schéma 6 : 12 classes ——
        st.markdown(
            """
<div class="schema">
  <div class="schema-title">12 activités HAPT</div>
  <div class="chip-grid">
    <div class="chip"><b>1 · 2 · 3</b>marche / escaliers ↑ ↓</div>
    <div class="chip"><b>4 · 5 · 6</b>assis / debout / allongé</div>
    <div class="chip"><b>7 → 12</b>transitions posturales</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if raw is None:
            st.warning("Fichiers bruts introuvables.")
        else:
            acc = np.loadtxt(raw / "acc_exp01_user01.txt")
            gyro = np.loadtxt(raw / "gyro_exp01_user01.txt")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fréquence", "50 Hz")
            c2.metric("Canaux", "6")
            c3.metric("Durée (exp01)", f"{len(acc)/50:.0f} s")
            c4.metric("Mesures", f"{len(acc):,}")
            st.pyplot(_plot_raw(acc, gyro), clear_figure=True)

    elif key == "clean":
        _idea(
            "<b>Analogie :</b> avant de lire un message, on enlève le bruit de fond. "
            "Ici on « nettoie » le signal pour que le modèle voie le mouvement utile."
        )
        _h("Ce qu’on fait")
        _simple(
            "<b>1. Aligner</b> — même unité, même rythme (50 Hz) si on mélange plusieurs sources.<br>"
            "<b>2. Filtrer</b> — atténuer les très lentes dérives (ex. la gravité qui tire toujours vers le bas) "
            "pour mieux voir le vrai geste.<br>"
            "<b>3. Version « evaluation propre »</b> — le filtre ne regarde <b>que le passé</b> "
            "(important si on veut anticiper le futur : on n’a pas le droit de tricher en regardant demain)."
        )
        _p("Résultat : un signal plus clair, prêt à être découpé.")

    elif key == "windows":
        _idea(
            "<b>Analogie :</b> on ne donne pas un film de 20 minutes d’un coup au modèle. "
            "On lui donne des <b>extraits de 3 secondes</b>, comme des clips courts."
        )
        _h("Pourquoi 3 secondes ?")
        _simple(
            "Assez long pour voir « ah, c’est de la marche ».<br>"
            "Assez court pour éviter qu’un seul clip mélange marche + assis + autre chose."
        )
        _h("Pourquoi les clips se chevauchent ?")
        _simple(
            "Si on coupe sans chevauchement, un changement d’activité peut tomber pile entre deux clips "
            "et être mal vu. On avance d’<b>1,5 s</b> à chaque fois (chevauchement 50 %) "
            "pour mieux couvrir les transitions."
        )
        st.pyplot(_plot_windows(), clear_figure=True)
        _p("Chaque clip devient un exemple d’apprentissage : <b>le signal + le nom de l’activité</b>.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Durée d’un clip", "3 s")
        c2.metric("Taille", "150 × 6")
        c3.metric("Avance", "1,5 s")

    elif key == "store":
        _idea(
            "<b>Analogie :</b> après le montage des clips, on ne garde pas les rushes bruts ouverts. "
            "On sauvegarde une bibliothèque propre prête à l’emploi."
        )
        _h("Dossier data/processed/")
        _simple(
            "<b>X</b> — tous les clips de mouvement.<br>"
            "<b>y</b> — l’activité de chaque clip.<br>"
            "<b>subjects</b> — qui a produit le clip (personne 1, 2, 3…).<br>"
            "<b>origins</b> — de quel jeu de données ça vient.<br>"
            "<b>subject_meta.json</b> — carnet d’adresses des personnes (ex. « Karim »)."
        )
        _p("Intérêt : ouvrir la démo ou lancer un entraînement sans tout recalculer depuis les fichiers texte.")
        if (PROCESSED / "X.npy").exists():
            X = np.load(PROCESSED / "X.npy", mmap_mode="r")
            y = np.load(PROCESSED / "y.npy")
            subjects = np.load(PROCESSED / "subjects.npy")
            origins = np.load(PROCESSED / "origins.npy")
            left, right = st.columns([1.55, 1])
            with left:
                st.pyplot(_plot_window(X[12], "Un clip déjà stocké (3 secondes)"), clear_figure=True)
            with right:
                st.metric("Nombre de clips", f"{len(X):,}")
                st.metric("Personnes", f"{len(np.unique(subjects))}")
                uniq, counts = np.unique(origins.astype(str), return_counts=True)
                st.bar_chart({"clips": dict(zip(uniq.tolist(), [int(c) for c in counts]))}, height=160)
            st.markdown(f'<div class="path">{PROCESSED.resolve()}</div>', unsafe_allow_html=True)

    elif key == "strict":
        _idea(
            "<b>Analogie :</b> pour un examen, on ne fait pas réviser avec les mêmes questions que le contrôle. "
            "Ici : les personnes du test ne sont <b>pas</b> celles de l’entraînement."
        )
        _h("Pourquoi ces règles ?")
        _simple(
            "<b>Séparer les personnes</b> — sinon le système « reconnaît Marie » au lieu d’apprendre la marche en général.<br>"
            "<b>Ne pas regarder le futur</b> dans le nettoyage — sinon l’anticipation triche.<br>"
            "<b>Garder l’ordre du temps</b> — pour anticiper, la suite doit vraiment arriver <i>après</i> ce qu’on a vu.<br>"
            "<b>Empreinte du protocole</b> — on sait exactement avec quelles règles un score a été calculé."
        )
        if STRICT.exists():
            proto = json.loads((STRICT / "protocol.json").read_text())
            Xs = np.load(STRICT / "X.npy", mmap_mode="r")
            splits = proto.get("splits", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clips", f"{Xs.shape[0]:,}")
            c2.metric("Personnes train", len(splits.get("train", [])))
            c3.metric("Personnes val", len(splits.get("validation", [])))
            c4.metric("Personnes test", len(splits.get("test", [])))
            st.markdown(f'<div class="path">{STRICT.resolve()}</div>', unsafe_allow_html=True)
        _p("C’est ce protocole qui sert aux <b>scores officiels</b> reconnaissance / anticipation.")

    elif key == "model":
        _idea(
            "<b>Analogie :</b> le modèle est un étudiant. "
            "D’abord il lit le clip (compréhension), ensuite il répond à la question "
            "(quelle activité ? / que va-t-il se passer ?)."
        )
        st.markdown(
            """
<div class="arch">
  <div class="box"><strong>Clip 3 s</strong><span>mouvement mesuré</span></div>
  <div class="aa">→</div>
  <div class="box mid"><strong>Encodeur</strong><span>« comprend » le geste</span></div>
  <div class="aa">→</div>
  <div>
    <div class="box out" style="min-height:42px;margin-bottom:8px"><strong>Reconnaître</strong><span>maintenant</span></div>
    <div class="box out" style="min-height:42px"><strong>Anticiper</strong><span>plus tard</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        _h("Deux façons de « se souvenir » des activités")
        _simple(
            "<b>Classique</b> — une grille de scores (softmax) apprise pendant l’entraînement.<br>"
            "<b>Prototypes</b> — pour chaque activité, on garde un « portrait-robot » moyen du geste. "
            "Pour décider : on regarde quel portrait ressemble le plus au clip actuel. "
            "Pratique pour ajouter une nouvelle personne sans tout reconstruire."
        )

    elif key == "reco":
        _idea(
            "<b>Question posée au système :</b> « Regarde ces 3 secondes. "
            "Qu’est-ce que la personne est en train de faire ? »"
        )
        _h("Comment ça se passe")
        _simple(
            "1. On prend un clip déjà nettoyé.<br>"
            "2. L’encodeur en fait un résumé compact.<br>"
            "3. On compare aux activités connues (ou on lit la grille de scores).<br>"
            "4. On affiche l’activité la plus probable : marche, assis, etc."
        )
        _h("Qualité sur le test")
        st.markdown(
            """
<div class="sg">
  <div class="sc a"><div class="v">85,9%</div><div class="l">Bonnes réponses</div></div>
  <div class="sc b"><div class="v">72,6%</div><div class="l">Équilibre entre classes</div></div>
  <div class="sc c"><div class="v">Live</div><div class="l">Page Reconnaissance</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        _simple(
            "<b>85,9 %</b> = sur 100 clips, environ 86 sont corrects.<br>"
            "<b>72,6 %</b> = moyenne qui traite chaque type d’activité à parts égales "
            "(plus sévère si certaines activités sont rares)."
        )

    elif key == "cl":
        _idea(
            "<b>Problème de la vie réelle :</b> le système a appris avec Marie et Thomas. "
            "Arrive <b>Karim</b>, qui marche un peu différemment. Que fait-on ?"
        )
        _h("Mauvaise idée")
        _simple(
            "Tout effacer et réapprendre seulement sur Karim → on <b>oublie</b> Marie et Thomas "
            "(oubli catastrophique)."
        )
        _h("Idée retenue")
        _simple(
            "On garde la « compréhension » des gestes (encodeur).<br>"
            "On met à jour la <b>mémoire des portraits</b> (prototypes) avec quelques exemples de Karim.<br>"
            "On peut aussi <b>revoir de vieux exemples</b> (rejeu) pour ne pas perdre l’ancien."
        )
        st.markdown(
            """
<div class="arch">
  <div class="box"><strong>Anciens users</strong><span>déjà connus</span></div>
  <div class="aa">+</div>
  <div class="box mid"><strong>Mémoire mise à jour</strong><span>sans tout recommencer</span></div>
  <div class="aa">→</div>
  <div class="box out"><strong>Karim inclus</strong><span>ancien savoir conservé</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption("À voir en live : page « Nouvel utilisateur (Karim) ».")

    elif key == "ant":
        _idea(
            "<b>Question différente :</b> on ne montre qu’une partie du mouvement "
            "(par ex. le début), et on demande : « d’après ça, que va-t-il se passer ensuite ? »"
        )
        _h("Pourquoi c’est plus dur")
        _simple(
            "Moins d’information → plus d’hésitation.<br>"
            "Les changements de posture (assis↔debout, etc.) sont <b>rares</b> et se ressemblent → "
            "le système se trompe plus souvent sur ces cas."
        )
        _h("Deux façons de compter les classes")
        _simple(
            "<b>12 classes</b> — on exige le détail exact de la transition "
            "(ex. assis→allongé vs assis→debout). Très difficile → score « équilibre » plus bas (~52 %).<br><br>"
            "<b>7 classes</b> — on garde les 6 activités stables, et on regroupe toutes les transitions "
            "dans une seule case « transition ». Plus réaliste quand on a peu d’exemples → "
            "meilleur équilibre (~77 %)."
        )
        st.markdown(
            """
<div class="sg">
  <div class="sc a"><div class="v">71,8%</div><div class="l">Anticipation · bonnes réponses (7 classes)</div></div>
  <div class="sc b"><div class="v">76,6%</div><div class="l">Anticipation · équilibre (7 classes)</div></div>
  <div class="sc c"><div class="v">51,6%</div><div class="l">Meilleur équilibre en 12 classes</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption("Démo : page Anticipation.")

    else:  # scores
        _idea(
            "<b>À retenir :</b> le système marche bien pour dire ce qui se passe maintenant, "
            "et correctement pour deviner la suite — surtout quand on ne demande pas le détail "
            "fin des transitions rares."
        )
        st.markdown(
            """
<div class="sg">
  <div class="sc a"><div class="v">85,9%</div><div class="l">Reconnaître maintenant</div></div>
  <div class="sc b"><div class="v">76,6%</div><div class="l">Anticiper (7 classes)</div></div>
  <div class="sc c"><div class="v">71,4%</div><div class="l">Autre modèle d’anticipation</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        _h("Chemin complet (résumé)")
        st.markdown(
            """
<div class="flow">
  <div class="node">Capteur</div><div class="ar">→</div>
  <div class="node">Fichiers bruts</div><div class="ar">→</div>
  <div class="node">Nettoyage</div><div class="ar">→</div>
  <div class="node">Clips 3 s</div><div class="ar">→</div>
  <div class="node">Stockage</div><div class="ar">→</div>
  <div class="node">Modèle</div><div class="ar">→</div>
  <div class="node">Reco / Adapt / Anticip.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        _simple(
            "Ensuite, dans le menu : essayez <b>Reconnaissance</b>, puis <b>Karim</b>, puis <b>Anticipation</b>."
        )

    st.markdown("---")
    _nav(idx, where="bottom")
