"""Create a two-panel figure: MAE comparison and accuracy-latency trade-off."""
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    base = pd.read_csv("results/baselines.csv")
    gnn = pd.read_csv("results/gnn.csv")
    eff = pd.read_csv("results/efficiency.csv")
    os.makedirs("figures", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: MAE comparison (mean +/- std over folds)
    names, means, stds = [], [], []
    for model, label in [("mean", "Mean"), ("composition_ridge", "Composition + Ridge")]:
        vals = base[base["model"] == model]["mae"].values
        names.append(label)
        means.append(vals.mean())
        stds.append(vals.std())
    names.append("CGCNN")
    means.append(gnn["mae"].mean())
    stds.append(gnn["mae"].std())

    axes[0].bar(names, means, yerr=stds, capsize=4, color=["#9aa5b1", "#4c78a8", "#f58518"])
    axes[0].set_ylabel("MAE (refractive index, unitless)")
    axes[0].set_title("Test accuracy (5-fold mean +/- std)")
    axes[0].tick_params(axis="x", rotation=12)

    # Right: accuracy-latency scatter
    for _, row in eff.iterrows():
        axes[1].scatter(row["latency_ms"], row["mae"], s=70)
        axes[1].annotate(row["model"], (row["latency_ms"], row["mae"]),
                         textcoords="offset points", xytext=(6, 6), fontsize=9)
    axes[1].set_xlabel("CPU latency per sample (ms)")
    axes[1].set_ylabel("Fold-0 MAE (unitless)")
    axes[1].set_title("Accuracy-efficiency trade-off")
    axes[1].margins(x=0.2, y=0.2)

    fig.tight_layout()
    out = "figures/accuracy_vs_latency.png"
    plt.savefig(out, dpi=300)
    print("saved:", out)


if __name__ == "__main__":
    main()
