# Efficient Graph Neural Networks for Materials Property Prediction

Independent AI-for-Science project - Shaoquan Wang

## Results (MatBench `matbench_dielectric`, 5-fold)

| Model | MAE | R2 | Size (MB) | CPU latency (ms) |
|---|---|---|---|---|
| Mean predictor | 0.809 +/- 0.072 | -0.004 +/- 0.007 | - | - |
| Composition + Ridge | 0.520 +/- 0.062 | 0.214 +/- 0.153 | - | - |
| CGCNN (FP32, 5-fold) | 0.353 +/- 0.075 | 0.272 +/- 0.181 | - | - |
| CGCNN (FP32, fold 0) | 0.244 | 0.565 | 0.172 | 1.89 |
| CGCNN (INT8) | 0.244 | 0.565 | 0.054 | 3.99 |

The CGCNN reduces MAE by 32.0% relative to the composition
baseline. Dynamic quantization makes the model about
3.2x smaller with negligible accuracy loss, but
does not reduce single-sample CPU latency because message passing dominates runtime.

## Training strategy

- CGCNN convolution with residual connections and LayerNorm; an optional sigmoid gate
  (`--gated`) is disabled by default because gating + dropout gave 0.366 MAE, worse than
  0.353 without them on this dataset.
- AdamW with 5-epoch warmup and cosine decay.
- Mixed precision (AMP) on CUDA and gradient clipping at 5.0.
- EMA weights (decay 0.99), compared with the best checkpoint on validation.
- Early stopping, multi-seed support (`--seeds 42,43`) and per-epoch logs.

## Data

- Dataset: `matbench_dielectric` (4,764 crystal structures)
- Target: refractive index (unitless)
- Splits: official MatBench 5-fold splits

## Run

```bash
pip install -r requirements.txt
python src/run_baseline.py
python src/run_gnn.py --folds 5 --epochs 30
python src/run_quantization.py --fold 0
python src/make_plots.py
```

## Limitations

One dataset and target; CPU-only latency; dynamic quantization only affects linear layers.
