"""
Node classification on Cora using PyTorch Geometric.

Approach: 2-layer GCN with dropout, trained with Adam + weight decay on the
standard public Planetoid split. We pick the best epoch by validation accuracy
and report the corresponding test accuracy.

We also train a small ensemble across seeds and average the logits to reduce
variance from random initialization, then report ensemble val/test accuracy.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T

DATA_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/cora/data/"
WORKING_DIR = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/runs/H_0/cora__free/")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_one(seed: int, data, num_features: int, num_classes: int,
              hidden: int = 16, lr: float = 0.01, weight_decay: float = 5e-4,
              dropout: float = 0.5, epochs: int = 200, patience: int = 50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = GCN(num_features, hidden, num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_logits = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=-1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_logits = logits.detach().cpu().numpy()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_val_acc, best_test_acc, best_logits


def main():
    t0 = time.time()
    # NormalizeFeatures is the standard preprocessing for Cora/Planetoid
    dataset = Planetoid(root=DATA_DIR, name="Cora", transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    n_train = int(data.train_mask.sum().item())
    n_val = int(data.val_mask.sum().item())
    n_test = int(data.test_mask.sum().item())
    print(f"Cora: nodes={data.num_nodes}, edges={data.num_edges}, "
          f"feat={num_features}, classes={num_classes}, "
          f"train/val/test={n_train}/{n_val}/{n_test}")

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    per_seed = []
    logits_sum = None
    for s in seeds:
        val_acc, test_acc, logits = train_one(s, data, num_features, num_classes)
        per_seed.append((s, val_acc, test_acc))
        print(f"  seed={s} val={val_acc:.4f} test={test_acc:.4f}")
        if logits_sum is None:
            logits_sum = logits.copy()
        else:
            logits_sum += logits

    # Ensemble: average best-val logits across seeds
    ens_pred = logits_sum.argmax(axis=-1)
    y = data.y.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    ens_val_acc = float((ens_pred[val_mask] == y[val_mask]).mean())
    ens_test_acc = float((ens_pred[test_mask] == y[test_mask]).mean())

    mean_val = float(np.mean([v for _, v, _ in per_seed]))
    mean_test = float(np.mean([t for _, _, t in per_seed]))
    std_test = float(np.std([t for _, _, t in per_seed]))

    print(f"Per-seed mean val={mean_val:.4f}, mean test={mean_test:.4f} +/- {std_test:.4f}")
    print(f"Ensemble  val={ens_val_acc:.4f}  test={ens_test_acc:.4f}")
    print(f"Elapsed: {time.time() - t0:.1f}s")

    # Headline = ensemble test accuracy (best single-model picker via val)
    result = {
        "target_column_chosen": "y",
        "headline_metric": "accuracy",
        "headline_value": ens_test_acc,
        "all_metrics": {
            "val_acc": ens_val_acc,
            "test_acc": ens_test_acc,
            "mean_val_acc_per_seed": mean_val,
            "mean_test_acc_per_seed": mean_test,
            "std_test_acc_per_seed": std_test,
        },
        "skills_applied": [],
        "interpretation_notes": (
            "2-layer GCN (hidden=16, dropout=0.5, Adam lr=0.01, wd=5e-4) on "
            "standard Planetoid Cora public split with row-normalized features; "
            "early-stop on val accuracy; ensembled best-val logits over 10 seeds."
        ),
    }

    with open(WORKING_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {WORKING_DIR / 'result.json'}")


if __name__ == "__main__":
    main()
