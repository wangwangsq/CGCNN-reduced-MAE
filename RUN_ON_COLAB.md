# Run this project on Colab

The current Windows machine has 32-bit Python 3.7, which cannot install PyTorch or PyTorch Geometric. Use Google Colab (free GPU/CPU) instead.

## Steps

1. Zip the folder `AI4S_MatBench_Kit` (or use the provided zip).
2. Open https://colab.research.google.com and create a new notebook.
3. Install dependencies:

```python
!pip install -q matbench pymatgen torch-geometric pandas matplotlib
```

4. Upload and unzip the project:

```python
from google.colab import files
files.upload()   # choose AI4S_MatBench_Kit.zip
!unzip -q -o AI4S_MatBench_Kit.zip
%cd AI4S_MatBench_Kit
```

5. Run the baseline:

```python
!python src/run_baseline.py
```

6. Verify the GNN on one fold first:

```python
!python src/run_gnn.py --folds 1 --epochs 40
```

7. If it works, run all five folds:

```python
!python src/run_gnn.py --folds 5 --epochs 80
```

8. Quantization and figure:

```python
!python src/run_quantization.py --fold 0
!python src/make_plots.py
```

9. Download `results/` and `figures/` to your computer.

## Notes

- The first run downloads the MatBench dataset; this may take a few minutes.
- If `torch-geometric` fails to install, run `!pip install torch-geometric` again after the first install; Colab usually matches the preinstalled torch.
- If the MatBench API differs in your installed version, send the traceback and adjust the data-loading function accordingly.
