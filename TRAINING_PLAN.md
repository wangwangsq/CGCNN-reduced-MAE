# Refined Training Plan (GPU Server)

Project: Efficient Graph Neural Networks for Materials Property Prediction

## 1. Data

| Dataset | Samples | Input | Target | Role |
|---|---|---|---|---|
| `matbench_dielectric` | 4,764 | crystal structure | refractive index (unitless) | primary |
| `matbench_phonons` | 1,265 | crystal structure | phonon peak frequency | generalization check |

Use the official MatBench 5-fold splits. Never create custom splits.

Metrics: MAE (primary), R² (secondary). Report mean ± std across folds and seeds.

## 2. Base configuration

| Parameter | Value |
|---|---|
| Graph cutoff | 4.0 Å |
| Node features | element one-hot (100) |
| Edge features | 40-bin Gaussian RBF |
| Conv layers | 3 (gated convolution) |
| Hidden dim | 64 |
| Dropout | 0.1 |
| Batch size | 64 (GPU; can try 128) |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| LR schedule | 5-epoch warmup + cosine decay |
| Epochs | 150 |
| Early stopping | patience 20 on validation MAE |
| Gradient clipping | 5.0 |
| AMP | enabled on CUDA |
| EMA | decay 0.999 |
| Seeds | 42, 43, 44 |

Server GPU recommendation: any 16 GB+ GPU (RTX 3090/4090, A5000, A100).
Estimated time: 3–6 min per fold for dielectric at 150 epochs; 5 folds × 3 seeds ≈ 1–1.5 h.

## 3. Experiment matrix

Priority order; stop after E5 if time is limited.

| ID | Purpose | Configuration | Runs |
|---|---|---|---|
| E0 | Baselines | mean; composition + Ridge; optional HistGradientBoosting | 5 folds |
| E1 | Main model | base config | 5 folds × 3 seeds |
| E2 | Depth ablation | layers = 2 / 3 / 4 | 5 folds × 1 seed |
| E3 | Cutoff ablation | cutoff = 3.0 / 4.0 / 5.0 Å | 5 folds × 1 seed |
| E4 | Width ablation | hidden = 64 / 128 | 5 folds × 1 seed |
| E5 | Quantization | FP32 vs dynamic INT8 | fold 0 (all seeds if time) |
| E6 | Generalization | `matbench_phonons`, base config | 5 folds × 1 seed |
| E7 (optional) | Static quantization / QAT | compare with dynamic INT8 | fold 0 |
| E8 (optional) | Error analysis | per chemical family + worst-5 samples | E1 predictions |

## 4. Training-loop refinements (implemented)

`src/run_gnn.py` now implements:

- gated CGCNN convolution + dropout 0.1;
- AdamW with 5-epoch warmup and cosine decay;
- AMP on CUDA with gradient clipping at 5.0;
- EMA weights (decay 0.999) and selection between best-checkpoint and EMA on validation;
- early stopping with patience 20;
- per-epoch logging in `results/logs/seed{seed}_fold{fold}.csv`;
- multi-seed runs via `--seeds 42,43`.

## 5. Running on the server

```bash
# environment
conda create -n ai4s python=3.11 -y
conda activate ai4s
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric matbench pymatgen pandas matplotlib

# data cache
export MATBENCH_DATA_DIR=$PWD/data

# E0
python src/run_baseline.py

# smoke test: one fold
python src/run_gnn.py --folds 1 --epochs 30

# E1: three seeds
for s in 42 43 44; do
  python src/run_gnn.py --folds 5 --epochs 150 --seed $s \
    --out results/gnn_seed${s}.csv --models models/seed${s}
done

# E2 depth
for l in 2 3 4; do
  python src/run_gnn.py --folds 5 --epochs 150 --layers $l \
    --out results/ablation_layers${l}.csv --models models/layers${l}
done

# E3 cutoff
for c in 3.0 4.0 5.0; do
  python src/run_gnn.py --folds 5 --epochs 150 --cutoff $c \
    --out results/ablation_cutoff${c}.csv --models models/cutoff${c}
done

# E5 quantization (after E1, using models/seed42/fold0.pt)
python src/run_quantization.py --fold 0 --model models/seed42/fold0.pt
python src/make_plots.py
```

## 6. Evaluation and reporting

- For each configuration, report MAE mean ± std over 5 folds × 3 seeds.
- Compare CGCNN against the composition baseline (relative MAE reduction).
- For quantization: report MAE, model size (MB), CPU latency (batch=1, median of 100 runs),
  and throughput (samples/s at batch=64).
- Run latency on the same CPU for FP32 and INT8; do not compare GPU latency with CPU latency.
- Error analysis: group by chemical family (oxide, halide, sulfide, contains transition metal)
  and inspect the five largest errors.

## 7. Success criteria

- CGCNN MAE at least 20% lower than the composition baseline.
- Seed-to-seed std below 10% of the mean MAE.
- Quantized model at least 2× smaller with MAE increase below 2%.
- Honest reporting: message passing dominates runtime, so dynamic quantization may not reduce
  single-sample latency.

## 8. Artifacts

```
results/
  baselines.csv
  gnn_seed42.csv, gnn_seed43.csv, gnn_seed44.csv
  ablation_layers{2,3,4}.csv
  ablation_cutoff{3,4,5}.csv
  efficiency.csv
  logs/
models/
  seed42/fold0.pt ... fold4.pt
figures/
  accuracy_vs_latency.png
  error_by_family.png
report/
  AI4S_Report.docx
```

