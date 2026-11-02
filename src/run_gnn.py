"""Train CGCNN on the official MatBench folds."""
import argparse
import copy
import csv
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark
from torch_geometric.loader import DataLoader

from ai4s_common import CGCNN, build_dataset, evaluate, set_seed


def load_task(dataset: str):
    mb = MatbenchBenchmark(autoload=False, subset=[dataset])
    task = list(mb.tasks)[0]
    task.load()
    return task


def get_test(task, fold):
    structures, targets = task.get_test_data(fold, include_target=True)
    return list(structures), np.asarray(targets, dtype=float)


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)
        return model


def make_scheduler(opt, epochs, warmup):
    def lr_lambda(epoch):
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = float(epoch - warmup) / float(max(1, epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_one(seed, fold, train_data, val_data, test_data, args, device):
    set_seed(seed + fold)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    model = CGCNN(
        hidden=args.hidden, layers=args.layers, dropout=args.dropout, gated=args.gated
    ).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(opt, args.epochs, args.warmup)
    loss_fn = nn.SmoothL1Loss()

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val, best_state, best_epoch, patience = float("inf"), None, -1, 0
    log_rows = []
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(batch)
                loss = loss_fn(pred, batch.y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            total += float(loss.detach()) * batch.num_graphs

        scheduler.step()
        val_mae = evaluate(model, val_loader, device)[0]
        lr = opt.param_groups[0]["lr"]
        log_rows.append([epoch, total / len(train_data), val_mae, lr])

        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == best_epoch:
            print(f"seed {seed} fold {fold} epoch {epoch:03d} "
                  f"train {total / len(train_data):.4f} val {val_mae:.4f} lr {lr:.2e}")
        if patience >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_val = evaluate(model, val_loader, device)[0]
    best_test = evaluate(model, test_loader, device)

    ema_model = copy.deepcopy(model).to(device)
    ema.copy_to(ema_model)
    ema_val = evaluate(ema_model, val_loader, device)[0]
    ema_test = evaluate(ema_model, test_loader, device)

    if ema_val < best_val:
        chosen = "ema"
        chosen_model = ema_model
        val_mae, (test_mae, test_r2, _, _) = ema_val, ema_test
    else:
        chosen = "best"
        chosen_model = model
        val_mae, (test_mae, test_r2, _, _) = best_val, best_test

    seconds = time.time() - started
    print(f"seed {seed} fold {fold}: {chosen}, val MAE {val_mae:.4f}, "
          f"test MAE {test_mae:.4f}, R2 {test_r2:.4f}, {seconds:.1f}s")

    model_dir = os.path.join(args.models, f"seed{seed}")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(chosen_model.state_dict(), os.path.join(model_dir, f"fold{fold}.pt"))

    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"seed{seed}_fold{fold}.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mae", "val_mae", "lr"])
        writer.writerows(log_rows)

    return dict(seed=seed, fold=fold, mae=test_mae, r2=test_r2,
                val_mae=val_mae, chosen=chosen, epochs=len(log_rows), seconds=seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="matbench_dielectric")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None,
                        help="run a single fold (overrides --folds)")
    parser.add_argument("--seeds", default="42", help="comma separated, e.g. 42,43,44")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gated", action="store_true",
                        help="enable the sigmoid gate in the convolution")
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--amp", action="store_true", default=torch.cuda.is_available(),
                        help="mixed precision on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/gnn.csv")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    task = load_task(args.dataset)
    rows = []

    fold_list = [args.fold] if args.fold is not None else list(task.folds)[: args.folds]
    for fold in fold_list:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        train_inputs = list(train_inputs)
        y_train = np.asarray(train_outputs, dtype=float)
        test_structures, y_test = get_test(task, fold)

        all_train = build_dataset(train_inputs, y_train, args.cutoff)
        test_data = build_dataset(test_structures, y_test, args.cutoff)

        for seed in seeds:
            rng = np.random.default_rng(seed + fold)
            idx = rng.permutation(len(all_train))
            n_val = max(1, int(0.1 * len(idx)))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            train_data = [all_train[i] for i in train_idx]
            val_data = [all_train[i] for i in val_idx]
            rows.append(train_one(seed, fold, train_data, val_data, test_data, args, device))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby("seed")[["mae", "r2"]].agg(["mean", "std"]))
    print("overall:", df["mae"].mean(), "+/-", df["mae"].std())
    print("saved:", args.out)


if __name__ == "__main__":
    main()
