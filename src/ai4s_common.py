"""Shared utilities for the MatBench AI-for-Science project."""
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymatgen.core import Element
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter

Z_MAX = 100


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _safe_float(value):
    try:
        value = float(value)
        return None if math.isnan(value) else value
    except Exception:
        return None


def composition_features(structure) -> np.ndarray:
    """Composition-only features: element fractions + weighted property statistics."""
    comp = structure.composition
    total = float(comp.num_atoms)
    fractions = np.zeros(Z_MAX, dtype=np.float32)
    for element, amount in comp.items():
        z = int(element.Z)
        if 0 < z < Z_MAX:
            fractions[z] = float(amount) / total

    stats = []
    for prop in ("X", "atomic_radius", "atomic_mass", "group", "row"):
        values, weights = [], []
        for element, amount in comp.items():
            value = _safe_float(getattr(element, prop, None))
            if value is not None:
                values.append(value)
                weights.append(float(amount))
        if values:
            values = np.asarray(values, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / weights.sum()
            mean = float((values * weights).sum())
            std = float(np.sqrt(((values - mean) ** 2 * weights).sum()))
            stats.extend([mean, std])
        else:
            stats.extend([0.0, 0.0])

    return np.concatenate([fractions, np.asarray(stats, dtype=np.float32),
                           np.asarray([len(comp), total], dtype=np.float32)])


def fit_ridge(X, y, alpha: float = 1.0):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    Xb = np.c_[np.ones(len(Xs)), Xs]
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    A[0, 0] -= alpha  # do not regularise the intercept
    w = np.linalg.solve(A, Xb.T @ y)
    return w, mu, sd


def predict_ridge(X, w, mu, sd):
    Xs = (X - mu) / sd
    Xb = np.c_[np.ones(len(Xs)), Xs]
    return Xb @ w


def rbf(distances, dmin: float = 0.0, dmax: float = 8.0, n_bins: int = 40) -> np.ndarray:
    centers = np.linspace(dmin, dmax, n_bins)
    width = (dmax - dmin) / n_bins
    distances = np.asarray(distances, dtype=np.float32).reshape(-1, 1)
    return np.exp(-((distances - centers) ** 2) / (2 * width ** 2)).astype(np.float32)


def structure_to_data(structure, target=None, cutoff: float = 4.0) -> Data:
    zs, positions = [], []
    for site in structure:
        elements = site.species.elements
        zs.append(int(elements[0].Z) if elements else 0)
        positions.append(site.coords)

    src, dst, distances = [], [], []
    for i, site in enumerate(structure):
        for neighbor in structure.get_neighbors(site, cutoff):
            src.append(i)
            dst.append(int(neighbor.index))
            distances.append(float(neighbor.nn_distance))

    x = torch.tensor(zs, dtype=torch.long)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(rbf(distances), dtype=torch.float)
    if edge_attr.numel() == 0:
        edge_attr = torch.zeros((0, 40), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if target is not None:
        data.y = torch.tensor([float(target)], dtype=torch.float)
    return data


def build_dataset(structures, targets, cutoff: float = 4.0):
    return [structure_to_data(s, t, cutoff) for s, t in zip(structures, targets)]


class ConvLayer(nn.Module):
    """Gated crystal graph convolution with a residual connection."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(2 * node_dim + edge_dim, out_dim)
        self.gate = nn.Linear(2 * node_dim + edge_dim, out_dim)
        self.bn = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        z = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        message = F.softplus(self.fc(z)) * torch.sigmoid(self.gate(z))
        out = scatter(message, dst, dim=0, dim_size=x.size(0), reduce="mean")
        out = self.dropout(out)
        return F.softplus(self.bn(out + x))


class CGCNN(nn.Module):
    def __init__(self, edge_dim: int = 40, hidden: int = 64, layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(Z_MAX, hidden)
        self.convs = nn.ModuleList(
            [ConvLayer(hidden, edge_dim, hidden, dropout=dropout) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, data):
        x = F.one_hot(data.x.clamp(min=0, max=Z_MAX - 1), num_classes=Z_MAX).float()
        x = self.embed(x)
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
        graph = global_mean_pool(x, data.batch)
        return self.head(graph).view(-1)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).detach().cpu())
            targets.append(batch.y.detach().cpu().view(-1))
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    return mae(y_true, y_pred), r2(y_true, y_pred), y_true, y_pred


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
