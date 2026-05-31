"""Cora node classification via 2-layer GCN (PyG) on GPU.

Standard Planetoid public split: 140 train / 500 val / 1000 test.
We train a GCN, select best epoch by val accuracy, then predict on test_mask nodes
and write submission.csv (node_id, predicted_class) + result.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T

HERE = Path(__file__).resolve().parent
DATA_ROOT = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/cora/data/"

SEED = 42
HIDDEN = 64
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 300
PATIENCE = 50


def set_seed(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_class: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden, n_class, cached=True, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def run_seed(seed: int, data, num_features: int, num_classes: int, device):
    set_seed(seed)
    model = GCN(num_features, HIDDEN, num_classes, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=-1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return logits.cpu(), best_val_acc, best_epoch


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset = Planetoid(
        root=DATA_ROOT,
        name="Cora",
        transform=T.NormalizeFeatures(),
    )
    data = dataset[0].to(device)
    num_features = dataset.num_node_features
    num_classes = dataset.num_classes
    print(
        f"nodes={data.num_nodes} edges={data.num_edges} feat={num_features} "
        f"classes={num_classes} train={int(data.train_mask.sum())} "
        f"val={int(data.val_mask.sum())} test={int(data.test_mask.sum())}"
    )

    # Ensemble across 5 seeds for stability (still well within budget).
    seeds = [42, 0, 1, 2, 3]
    all_logits = []
    val_accs = []
    for s in seeds:
        logits, val_acc, best_ep = run_seed(s, data, num_features, num_classes, device)
        all_logits.append(F.softmax(logits, dim=-1).numpy())
        val_accs.append(val_acc)
        print(f"  seed={s} best_val_acc={val_acc:.4f} best_epoch={best_ep}")

    mean_probs = np.mean(all_logits, axis=0)
    y_pred = mean_probs.argmax(axis=1)
    y_true = data.y.cpu().numpy()

    test_mask = data.test_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()

    test_acc = float((y_pred[test_mask] == y_true[test_mask]).mean())
    val_acc_ens = float((y_pred[val_mask] == y_true[val_mask]).mean())
    train_acc = float((y_pred[train_mask] == y_true[train_mask]).mean())

    print(f"ensemble val_acc={val_acc_ens:.4f} test_acc={test_acc:.4f}")

    # Submission: predictions on the canonical test_mask nodes
    test_node_ids = np.where(test_mask)[0]
    sub_rows = []
    for nid in test_node_ids:
        sub_rows.append((int(nid), int(y_pred[nid])))

    sub_path = HERE / "submission.csv"
    with open(sub_path, "w") as f:
        f.write("node_id,category\n")
        for nid, c in sub_rows:
            f.write(f"{nid},{c}\n")

    # Per-class accuracy on test
    per_class = {}
    for c in range(num_classes):
        m = (y_true == c) & test_mask
        if m.sum() > 0:
            per_class[str(c)] = float((y_pred[m] == y_true[m]).mean())

    elapsed = time.time() - t0
    result = {
        "target_column_chosen": "category",
        "headline_metric": "test_accuracy",
        "headline_value": test_acc,
        "all_metrics": {
            "test_accuracy": test_acc,
            "val_accuracy_ensemble": val_acc_ens,
            "train_accuracy": train_acc,
            "per_seed_val_accuracy": val_accs,
            "per_class_test_accuracy": per_class,
        },
        "skills_applied": [],
        "interpretation_notes": (
            "2-layer GCN (hidden=64, dropout=0.5) on Cora with Planetoid public split. "
            "Trained on GPU with Adam (lr=0.01, wd=5e-4), early stopping on val acc, "
            "ensembled over 5 seeds. None of the v4 tabular skills applied (this is "
            "a graph task with a fixed train/val/test mask)."
        ),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {sub_path} ({len(sub_rows)} rows)")
    print(f"wrote {HERE / 'result.json'}")
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
