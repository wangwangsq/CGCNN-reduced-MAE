"""Train a CGCNN on MatBench with the official folds."""
import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark
from torch_geometric.loader import DataLoader

from ai4s_common import CGCNN, build_dataset, evaluate, mae, r2, set_seed


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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/gnn.csv")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    set_seed(args.seed)
    task = load_task(args.dataset)
    os.makedirs(args.models, exist_ok=True)
    rows = []

    for fold in list(task.folds)[: args.folds]:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        train_inputs = list(train_inputs)
        test_structures, y_test = get_test(task, fold)

        train_data = build_dataset(train_inputs, np.asarray(train_outputs), args.cutoff)
        test_data = build_dataset(test_structures, y_test, args.cutoff)

        rng = np.random.default_rng(args.seed + fold)
        idx = rng.permutation(len(train_data))
        n_val = max(1, int(0.1 * len(idx)))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        train_loader = DataLoader([train_data[i] for i in train_idx],
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader([train_data[i] for i in val_idx], batch_size=args.batch_size)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)

        model = CGCNN(hidden=args.hidden, layers=args.layers).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        loss_fn = nn.L1Loss()

        best_state, best_val, patience = None, float("inf"), 0
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for batch in train_loader:
                batch = batch.to(args.device)
                opt.zero_grad()
                loss = loss_fn(model(batch), batch.y.view(-1))
                loss.backward()
                opt.step()
                total += float(loss) * batch.num_graphs

            val_mae, _, _, _ = evaluate(model, val_loader, args.device)
            if val_mae < best_val - 1e-4:
                best_val, best_state, patience = val_mae, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
            if epoch % 10 == 0 or patience == 0:
                print(f"fold {fold} epoch {epoch:03d} train MAE {total/len(train_loader.dataset):.4f} val MAE {val_mae:.4f}")
            if patience >= 20:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        test_mae, test_r2, y_true, y_pred = evaluate(model, test_loader, args.device)
        rows.append(dict(model="cgcnn", fold=fold, mae=test_mae, r2=test_r2))
        torch.save(model.state_dict(), os.path.join(args.models, f"fold{fold}.pt"))
        np.savez(os.path.join(args.models, f"pred_fold{fold}.npz"), y_true=y_true, y_pred=y_pred)
        print(f"fold {fold}: test MAE={test_mae:.4f}, R2={test_r2:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df[["mae", "r2"]].agg(["mean", "std"]))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
