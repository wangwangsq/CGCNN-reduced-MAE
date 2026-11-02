#!/usr/bin/env bash
# Create an isolated environment for the AI4S MatBench project.
# Usage: bash setup.sh [cu121|cu124|cpu]
set -e

TORCH_INDEX="${1:-cu121}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using Python: $($PYTHON_BIN --version)"
$PYTHON_BIN -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

case "$TORCH_INDEX" in
  cpu)
    echo "Installing CPU PyTorch..."
    pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
    ;;
  cu124)
    echo "Installing CUDA 12.4 PyTorch..."
    pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu124
    ;;
  *)
    echo "Installing CUDA 12.1 PyTorch..."
    pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu121
    ;;
esac

# matbench 0.6 has outdated pins in its metadata; install it without dependencies,
# then install the working scientific stack explicitly.
pip install torch-geometric==2.8.1 pymatgen==2026.10.24 \
  pandas==2.3.3 numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1 \
  matplotlib==3.11.2 matminer==0.10.1
pip install matbench==0.6 --no-deps

python -c "import torch, torch_geometric, matbench, pymatgen; print('torch', torch.__version__); print('PyG', torch_geometric.__version__); print('matbench', matbench.__version__ if hasattr(matbench,'__version__') else '0.6'); print('pymatgen', pymatgen.__version__ if hasattr(pymatgen,'__version__') else 'ok')"
echo "Environment ready. Activate with: source .venv/bin/activate"
