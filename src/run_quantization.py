"""Evaluate FP32 vs dynamic INT8 quantization for a trained CGCNN."""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark
from torch_geometric.loader import DataLoader

from ai4s_common import CGCNN, build_dataset, evaluate, mae, r2


def load_task(dataset: str):
    mb = MatbenchBenchmark(autoload=False, subset=[dataset])
    task = list(mb.tasks)[0]
    task.load()
    return task


def get_test(task, fold):
    structures, targets = task.get_test_data(fold, include_target=True)
    return list(structures), np.asarray(targets, dtype=float)


def model_size_mb(model) -> float:
    path = "_tmp_model.pt"
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size


def latency_ms(model, sample, n_warmup: int = 20, n_runs: int = 100) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample)
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="matbench_dielectric")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--model", default="models/seed42/fold0.pt")
    parser.add_argument("--out", default="results/efficiency.csv")
    args = parser.parse_args()

    task = load_task(args.dataset)
    test_structures, y_test = get_test(task, args.fold)
    test_data = build_dataset(test_structures, y_test, args.cutoff)
    loader = DataLoader(test_data, batch_size=64)

    model = CGCNN(hidden=args.hidden, layers=args.layers)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    fp_mae, fp_r2, _, _ = evaluate(model, loader, "cpu")
    qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    q_mae, q_r2, _, _ = evaluate(qmodel, loader, "cpu")

    sample = test_data[0]
    sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long)
    rows = [
        dict(model="cgcnn_fp32", mae=fp_mae, r2=fp_r2,
             size_mb=model_size_mb(model), latency_ms=latency_ms(model, sample)),
        dict(model="cgcnn_int8", mae=q_mae, r2=q_r2,
             size_mb=model_size_mb(qmodel), latency_ms=latency_ms(qmodel, sample)),
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
