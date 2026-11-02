# Instructions for Codex

This repository is a self-contained AI-for-Science project. When asked to run it:

1. Use Python 3.11 and the local virtual environment:
   ```bash
   bash setup.sh cu121   # or cu124 / cpu
   source .venv/bin/activate
   ```
   If the environment already exists, just activate `.venv`.

2. Verify imports before training:
   ```bash
   python -c "import torch, torch_geometric, matbench, pymatgen; print('ok')"
   ```

3. Run a smoke test first:
   ```bash
   python src/run_gnn.py --folds 1 --epochs 20
   ```
   Do not start the 5-fold run until the smoke test produces `results/gnn.csv`.

4. Run the full pipeline (default: 5 folds, 80 epochs, seeds 42 and 43):
   ```bash
   bash run_all.sh 5 80
   ```
   On a GPU server this should take roughly 1-2 hours depending on the GPU. For a quicker
   first pass use `bash run_all.sh 5 40 42` (one seed).

5. Outputs:
   - `results/baselines.csv`
   - `results/gnn.csv`
   - `results/efficiency.csv`
   - `figures/accuracy_vs_latency.png`
   - `models/fold*.pt`

6. Known warnings that are safe:
   - matbench 0.6 reports dependency version conflicts (we install with `--no-deps`;
     the runtime works with the pinned scientific stack).
   - `torch.ao.quantization` deprecation warnings.
   - matplotlib font-cache warnings if `MPLCONFIGDIR` is not writable.

7. Do not overwrite `results/` or `models/` from a previous run without asking.
   For a new experiment, pass a new `--out` and `--models` directory.
