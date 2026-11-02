# CGCNN for MatBench dielectric prediction

This repository is a small materials-informatics project built around two
questions:

1. How much does crystal structure help when predicting a material property
   compared with using composition alone?
2. Does post-training INT8 quantization make a compact graph neural network
   cheaper to store and run without damaging its predictions?

I used `matbench_dielectric`, which contains 4,764 crystal structures and
asks the model to predict refractive index. All experiments use the official
MatBench five-fold splits.

## Results

The values below are from seed 42 and a 30-epoch run. The scripts default to
longer, multi-seed runs, but I stopped at 30 epochs after two consistent runs
gave the same test MAE. Fold means use sample standard deviation.

| Model | MAE | R² | Size (MB) | CPU latency (ms) |
|---|---:|---:|---:|---:|
| Training-set mean | 0.809 ± 0.080 | −0.004 ± 0.008 | — | — |
| Composition + Ridge | 0.520 ± 0.070 | 0.214 ± 0.171 | — | — |
| CGCNN, FP32 (five folds) | **0.353 ± 0.083** | **0.272 ± 0.203** | — | — |
| CGCNN, FP32 (fold 0) | 0.244 | 0.565 | 0.172 | 1.83 |
| CGCNN, dynamic INT8 (fold 0) | 0.244 | 0.565 | **0.054** | 3.95 |

The main result is that adding crystal structure lowers MAE by about 32%
relative to the composition-only baseline. Quantization reduces the saved
model size by 3.2× with almost no change in MAE, but it does not improve
single-sample latency. Most of the runtime is in graph construction and
message passing, while `torch.quantization.quantize_dynamic` only changes
the linear layers.

The fold-level MAEs are 0.244, 0.330, 0.477, 0.354, and 0.361. The spread is
large enough that I report all five folds rather than selecting a favorable
split.

## Training setup

The model is a three-layer CGCNN with 64 hidden channels, a 4 Å neighbor
cutoff, 40 Gaussian radial-basis edge features, LayerNorm, residual
connections, and mean pooling. Training uses AdamW, a five-epoch warmup
followed by cosine decay, gradient clipping, and exponential moving
average weights. AMP is enabled when CUDA is available.

I also tested a gated convolution with dropout 0.1. Its five-fold MAE was
0.366, so the default configuration keeps the gate disabled and dropout at
zero. The relevant options remain available through `--gated` and
`--dropout`.

## Reproduce

Create the environment on Linux or WSL:

```bash
bash setup.sh cpu        # or cu121 / cu124 for a CUDA machine
source .venv/bin/activate
```

Run the baselines, five folds, quantization, and plotting steps:

```bash
python src/run_baseline.py
python src/run_gnn.py --folds 5 --epochs 30
python src/run_quantization.py --fold 0
python src/make_plots.py
```

For a one-fold smoke test, use:

```bash
python src/run_gnn.py --folds 1 --epochs 20 --out results/gnn_smoke.csv
```

The first MatBench run downloads and caches the dataset. On this machine the
five-fold CPU run took about 12–15 minutes.

## Repository layout

```text
src/          data handling, baselines, CGCNN training, quantization, plots
results/      committed CSV results for all reported numbers
figures/      accuracy and latency comparison
report/       short project report
setup.sh      environment setup
run_all.sh    end-to-end pipeline
```

## Limitations

This is a single dataset and target, and the latency measurement is CPU-only.
Dynamic quantization does not cover graph operations or message passing.
The next useful experiment would be a chemical-family error analysis and a
second MatBench task to check whether the structural gain transfers beyond
dielectric prediction.
