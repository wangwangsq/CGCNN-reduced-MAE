#!/usr/bin/env bash
# End-to-end run. Usage: bash run_all.sh [folds] [epochs] [seeds]
set -e

FOLDS="${1:-5}"
EPOCHS="${2:-80}"
SEEDS="${3:-42,43}"

mkdir -p results figures models
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

echo "=== Baselines ==="
python src/run_baseline.py

echo "=== GNN smoke test (1 fold, 20 epochs) ==="
python src/run_gnn.py --folds 1 --epochs 20 --out results/gnn_smoke.csv --models models/smoke

echo "=== GNN full run (${FOLDS} folds, ${EPOCHS} epochs, seeds ${SEEDS}) ==="
python src/run_gnn.py --folds "$FOLDS" --epochs "$EPOCHS" --seeds "$SEEDS"

echo "=== Quantization ==="
python src/run_quantization.py --fold 0 --model models/seed42/fold0.pt

echo "=== Figure ==="
python src/make_plots.py

echo "Done. See results/ and figures/."
