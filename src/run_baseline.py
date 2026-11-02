"""Run mean and composition-only ridge baselines on MatBench."""
import argparse
import os

import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark

from ai4s_common import composition_features, fit_ridge, mae, predict_ridge, r2


def load_task(dataset: str):
    mb = MatbenchBenchmark(autoload=False, subset=[dataset])
    task = list(mb.tasks)[0]
    task.load()
    return task


def get_test(task, fold):
    structures, targets = task.get_test_data(fold, include_target=True)
    return list(structures), np.asarray(targets, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="matbench_dielectric")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out", default="results/baselines.csv")
    args = parser.parse_args()

    task = load_task(args.dataset)
    rows = []

    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        train_inputs = list(train_inputs)
        y_train = np.asarray(train_outputs, dtype=float)
        test_structures, y_test = get_test(task, fold)

        # 1) mean predictor
        pred_mean = np.full(len(y_test), y_train.mean())
        rows.append(dict(model="mean", fold=fold, mae=mae(y_test, pred_mean),
                         r2=r2(y_test, pred_mean)))

        # 2) composition features + ridge regression
        X_train = np.stack([composition_features(s) for s in train_inputs])
        X_test = np.stack([composition_features(s) for s in test_structures])
        w, mu, sd = fit_ridge(X_train, y_train, alpha=args.alpha)
        pred_ridge = predict_ridge(X_test, w, mu, sd)
        rows.append(dict(model="composition_ridge", fold=fold,
                         mae=mae(y_test, pred_ridge),
                         r2=r2(y_test, pred_ridge)))

        print(f"fold {fold}: mean MAE={rows[-2]['mae']:.4f}, "
              f"ridge MAE={rows[-1]['mae']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby("model")[["mae", "r2"]].agg(["mean", "std"]))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
