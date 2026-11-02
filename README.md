# Efficient Graph Neural Networks for Materials Property Prediction

Independent AI-for-Science project · Shaoquan Wang

This project studies the accuracy–efficiency trade-off of graph neural networks for
materials property prediction on the MatBench benchmark. It compares:

1. a mean predictor,
2. a composition-only ridge regression,
3. a crystal graph convolutional network (CGCNN),
4. a dynamically quantized (INT8) version of the CGCNN.

## Results

Dataset: `matbench_dielectric` (4,764 crystal structures; official 5-fold splits).
Target: refractive index (unitless). Metric: MAE (lower is better).

| Model | MAE | R² | Size (MB) | CPU latency (ms) |
|---|---|---|---|---|
| Mean predictor | 0.809 ± 0.072 | -0.004 ± 0.007 | — | — |
| Composition + Ridge | 0.520 ± 0.062 | 0.214 ± 0.153 | — | — |
| CGCNN (FP32, 5-fold) | **0.340 ± 0.076** | **0.273 ± 0.186** | — | — |
| CGCNN (FP32, fold 0) | 0.233 | 0.562 | 0.172 | 1.90 |
| CGCNN (INT8, fold 0) | 0.234 | 0.560 | **0.054** | 3.91 |

Key findings:

- Structural information reduces MAE by **34.5%** relative to the composition baseline.
- Dynamic INT8 quantization makes the model **3.2× smaller** with negligible accuracy loss.
- Single-sample CPU latency does **not** improve (1.90 → 3.91 ms): message passing dominates
  runtime, and quantization adds quantize/dequantize overhead. This is reported honestly rather
  than framed as an acceleration.

## Training strategy

- Gated crystal graph convolution (sigmoid gate), residual connections and LayerNorm.
- Dropout 0.1 in the convolution and regression head.
- AdamW with 5-epoch warmup and cosine decay.
- Mixed precision (AMP) on CUDA and gradient clipping at 5.0.
- Exponential moving average (EMA, decay 0.999) of model weights; the better of the
  best-validation checkpoint and the EMA model is used for the test set.
- Early stopping with patience 20.
- Multi-seed support (`--seeds 42,43`).
- Per-epoch logs in `results/logs/`.

The results above are a reference run from the 30-epoch, single-seed configuration. The current
code adds gated convolution and EMA, so a fresh run may differ slightly.

## Quick start (server)

```bash
# 1. Environment (choose cu121, cu124 or cpu)
bash setup.sh cu121
source .venv/bin/activate

# 2. Smoke test
python src/run_gnn.py --folds 1 --epochs 20

# 3. Full pipeline
bash run_all.sh 5 80
```

## Manual run

```bash
python src/run_baseline.py
python src/run_gnn.py --folds 5 --epochs 80
python src/run_quantization.py --fold 0
python src/make_plots.py
```

Multi-seed run (recommended for the final report):

```bash
for s in 42 43 44; do
  python src/run_gnn.py --folds 5 --epochs 150 --seed $s \
    --out results/gnn_seed${s}.csv --models models/seed${s}
done
```

## Environment notes

- Python **3.11** is recommended.
- `matbench==0.6` declares outdated pins (scikit-learn 1.0.1, scipy 1.7.3) that have no
  Python 3.11 wheels. `setup.sh` installs matbench with `--no-deps` and then installs a working
  scientific stack. The runtime has been verified with the versions in `requirements.txt`.
- PyTorch is installed separately because the correct wheel depends on the server CUDA version.
  Check with `nvidia-smi` and choose `cu121`, `cu124`, or `cpu`.

## Data

The first run downloads `matbench_dielectric.json.gz` automatically to the matminer dataset
cache. If the server has no external network, download
`https://ml.materialsproject.org/projects/matbench_dielectric.json.gz` manually and place it in
the matminer dataset directory.

## Repository structure

```
src/
  ai4s_common.py       graph construction, CGCNN, metrics
  run_baseline.py      mean and composition ridge baselines
  run_gnn.py           CGCNN training on official folds
  run_quantization.py  FP32 vs INT8 evaluation
  make_plots.py        accuracy and accuracy-latency figures
results/               CSV results
models/                checkpoints and predictions
figures/               figures
report/                two-page report
```

## Expected outputs

- `results/baselines.csv` — mean and ridge results per fold
- `results/gnn.csv` — CGCNN results per fold
- `results/efficiency.csv` — FP32/INT8 MAE, size and latency
- `figures/accuracy_vs_latency.png` — MAE comparison and accuracy-latency plot
- `models/fold0.pt … fold4.pt` — trained checkpoints

## Citation

Dunn, A., Wang, Q., Ganose, A. et al. Benchmarking materials property prediction methods:
the Matbench test set and Automat miner reference algorithm. *npj Comput Mater* 6, 138 (2020).
