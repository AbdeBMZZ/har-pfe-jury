# Livrable jury — HAR Continual Learning

Version fusionnée pour la soutenance PFE : **base auditée Desktop** + **anticipation causale v2**.

## Ce que contient ce dossier

| Provenance | Contenu conservé |
|---|---|
| `Desktop/pfe/har_project` | Code reconnaissance, corrections audit 2026-09-08, démos PFE, tests de régression |
| `Downloads/har-continual-learning-main` | Anticipation causale v2 (`CausalForecaster`), pipeline, UI, exemples, checkpoint `anticipation_v2_p50` |

Les corrections d’audit (prétraitement gyro, splits, métriques CL, UWR, calibration) sont **conservées**. Elles ne sont pas remplacées par l’ancienne version Downloads.

## Démos recommandées devant le jury

```bash
pip install -r requirements.txt

# Démo principale (reconnaissance + continual + anticipation)
streamlit run app_streamlit.py

# Anticipation causale seule (sans checkpoint de reconnaissance)
streamlit run app_anticipation.py

# Script narratif terminal
python demo_soutenance.py
```

Dans Streamlit, page **Anticipation** → choisir **Causale v2 (recommandé jury)**.  
Exemple fourni : `examples/observed_context_p50.npy`.

## Ce qu’il ne faut pas affirmer

- L’anticipation v2 prédit une **fenêtre future HAPT**, pas une chute.
- Le module fall-risk est un **proxy de transitions posturales**, pas une validation clinique.
- Les scores LSTM historiques (matrices fusionnées) **ne sont pas comparables** aux scores v2.
- Les figures dans `results/figures/` peuvent précéder certaines corrections de protocole.

## Tests légers

```bash
python -m unittest discover -s tests -v
```

## Données incluses pour une autre machine

Inclus (suffisant pour les démos complètes) :

- `data/processed/` — fenêtres HAPT+WISDM (~163 Mo) + `subject_meta.json`
- `checkpoints/` — reconnaissance, anticipation LSTM, fall-risk, SSL, anticipation v2
- `examples/` — fichiers `.npy` pour l’UI d’anticipation

**Non inclus** (uniquement pour ré-entraîner depuis zéro) :

- `data/raw/` (~2,5 Go)
- protocole HAPT strict à reconstruire

Sur une autre machine :

```bash
unzip har_project_jury.zip && cd har_project_jury
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Pour ré-entraîner l’anticipation v2 :

```bash
python scripts/fetch_hapt_raw.py --out data/HAPT
python scripts/prepare_hapt_protocol.py --raw data/HAPT/RawData --acc-unit g --out data/hapt_strict
python scripts/anticipation_pipeline.py train --protocol data/hapt_strict --out checkpoints/new_p50 --ratio 0.5
```

Rapports : `results/anticipation_v2_summary.json`, `results/anticipation_v2_p50*_test.json`.
