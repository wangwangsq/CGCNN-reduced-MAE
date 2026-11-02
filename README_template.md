# Efficient Graph Neural Networks for Materials Property Prediction

Independent AI-for-Science project · Shaoquan Wang

## Overview

This project studies the accuracy–efficiency trade-off of graph neural networks for materials property prediction on the MatBench benchmark. It compares a composition-only baseline, a crystal graph neural network, and a dynamically quantized version of the same network.

## Results

| Model | MAE (eV) | R² | Size (MB) | CPU latency (ms) |
|---|---|---|---|---|
| Mean predictor | | | — | — |
| Composition + XGBoost | | | | |
| CGCNN / MEGNet (FP32) | | | | |
| CGCNN / MEGNet (INT8) | | | | |

All results are reported as mean ± std over the five official MatBench folds.

## Data

- Dataset: `matbench_expt_gap`
- Task: predict experimental band gap (eV)
- Splits: official MatBench 5-fold splits

## Setup

```bash
conda create -n ai4s python=3.10 -y
conda activate ai4s
pip install -r requirements.txt
```

## Reproduce

```bash
python src/baseline.py
python src/train_gnn.py --folds 5
python src/quantize_eval.py
python src/error_analysis.py
```

## Repository structure

```
src/         training and evaluation scripts
results/     CSV results for baselines, GNN and efficiency
figures/     accuracy–latency and error-analysis figures
report/      two-page report
```

## Limitations

- One dataset and one target (experimental band gap).
- CPU-only latency measurements.
- Dynamic quantization only affects linear layers; message passing is not quantized.

## Citation

Dunn, A., Wang, Q., Ganose, A. et al. Benchmarking materials property prediction methods: the Matbench test set and Automat miner reference algorithm. *npj Comput Mater* 6, 138 (2020).

