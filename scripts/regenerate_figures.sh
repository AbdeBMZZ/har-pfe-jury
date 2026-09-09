#!/usr/bin/env bash
# Regenerate all result figures for the PFE memoire.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-30}"
EPOCHS_CL="${EPOCHS_CL:-10}"
N_TASKS_SOTA="${N_TASKS_SOTA:-15}"
FIG_DIR="results/figures"
CKPT_DIR="checkpoints"
LOG="results/regenerate_figures.log"

mkdir -p "$FIG_DIR" "$CKPT_DIR" results

exec > >(tee -a "$LOG") 2>&1
echo "=== Figure regeneration started $(date) ==="

# 1. Preprocess (skip if already done)
if [[ ! -f data/processed/X.npy ]]; then
  $PY scripts/preprocess.py --data_root data/raw --out data/processed
fi

# 2. Pre-train backbone
if [[ ! -f "$CKPT_DIR/pretrained.pt" ]]; then
  echo "--- Pre-training ($EPOCHS_PRETRAIN epochs) ---"
  $PY scripts/train.py --mode pretrain --data data/processed \
    --epochs "$EPOCHS_PRETRAIN" --out_dir "$CKPT_DIR"
fi

CKPT="$CKPT_DIR/pretrained.pt"

# 3. Continual learning (user + class)
if [[ ! -f "$CKPT_DIR/results_matrix_user.npy" ]]; then
  echo "--- User-incremental continual learning ---"
  $PY scripts/train.py --mode continual --scenario user \
    --data data/processed --checkpoint "$CKPT" \
    --epochs "$EPOCHS_CL" --out_dir "$CKPT_DIR"
fi

if [[ ! -f "$CKPT_DIR/results_matrix_class.npy" ]]; then
  echo "--- Class-incremental continual learning ---"
  $PY scripts/train.py --mode continual --scenario class \
    --data data/processed --checkpoint "$CKPT" \
    --epochs "$EPOCHS_CL" --out_dir "$CKPT_DIR"
fi

# 4. Naive baseline (for comparison plot)
if [[ ! -f "$CKPT_DIR/baseline_matrix_user.npy" ]]; then
  echo "--- Naive baseline (user) ---"
  $PY scripts/run_baseline.py --data data/processed --checkpoint "$CKPT" \
    --scenario user --epochs "$EPOCHS_CL" --out_dir "$CKPT_DIR"
fi

# 5. SOTA comparison (EWC, iCaRL, ours)
if [[ ! -f "$FIG_DIR/sota_comparison.png" ]]; then
  echo "--- SOTA comparison ($N_TASKS_SOTA tasks) ---"
  $PY scripts/run_sota_comparison.py --data data/processed --checkpoint "$CKPT" \
    --n_tasks "$N_TASKS_SOTA" --epochs "$EPOCHS_CL" \
    --out_dir "$CKPT_DIR" --fig_dir "$FIG_DIR"
fi

# 6. Anticipation module
if [[ ! -f "$CKPT_DIR/anticipation_results.npy" ]]; then
  echo "--- Anticipation training ---"
  $PY scripts/train_anticipation.py --data data/processed --checkpoint "$CKPT" \
    --epochs 30 --out_dir "$CKPT_DIR"
fi

# 7. Plot all matrices / history
echo "--- visualize.py (user) ---"
$PY scripts/visualize.py --results_dir "$CKPT_DIR" --scenario user --out_dir "$FIG_DIR"
echo "--- visualize.py (class) ---"
$PY scripts/visualize.py --results_dir "$CKPT_DIR" --scenario class --out_dir "$FIG_DIR"

# 8. t-SNE + attention
echo "--- t-SNE embeddings ---"
$PY scripts/visualize_embeddings.py --data data/processed --checkpoint "$CKPT" \
  --out_dir "$FIG_DIR" --max_samples 2000

# 9. Class-incremental diagnostic (buffer / classes per task)
if [[ ! -f "$FIG_DIR/class_incremental_analysis.png" ]]; then
  echo "--- Class-incremental analysis ---"
  $PY scripts/analysis_class_incremental.py --data data/processed \
    --checkpoint "$CKPT" --epochs "$EPOCHS_CL" --out_dir "$FIG_DIR"
fi

# 10. Copy to LaTeX project
DEST="../pfe_finale_conc/figures"
mkdir -p "$DEST"
cp -f "$FIG_DIR"/*.png "$DEST/" 2>/dev/null || true
cp -f assets/*.png "$DEST/" 2>/dev/null || true

echo "=== Done $(date) — figures in $FIG_DIR and $DEST ==="
ls -la "$FIG_DIR"/*.png 2>/dev/null || true
