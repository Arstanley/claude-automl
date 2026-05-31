"""GCN on Cora — target >0.90 test accuracy.

Approach:
- 2-layer GCN (PyG GCNConv, normalize=True, cached) — the "GCN algorithm" the spec names.
- Row-normalize features (NormalizeFeatures).
- Train on train+val masks (spec only constrains test accuracy; val labels are
  not held out by the task spec).
- Per seed: track best-val checkpoint logits (val_mask of Planetoid serves as
  an internal selection signal even though it is also in train_mask — its
  per-epoch noise still differentiates well/poorly-fit checkpoints).
- Ensemble best-val-checkpoint softmax across many seeds.
- Self-training round: high-confidence pseudo-labels on unlabeled nodes,
  retrain ensemble, average with round-1.

Skill library v4 is tabular-focused; no precondition matches a graph-NN task,
so skills_applied is [].
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures

WORK_DIR = Path(__file__).resolve().parent
DATA_ROOT = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/cora/data/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, n_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden, n_classes, cached=True, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_seed_best_val(x, edge_index, y, train_idx, val_idx, num_classes, seed,
                        epochs=300, hidden=64, lr=0.01, weight_decay=5e-4, dropout=0.5):
    """Train one GCN; return logits from epoch with best val-mask accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GCN(x.size(1), hidden, num_classes, dropout=dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_logits = None
    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optim.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
                pred = logits.argmax(dim=1)
                val_acc = (pred[val_idx] == y[val_idx]).float().mean().item()
                if val_acc > best_val:
                    best_val = val_acc
                    best_logits = logits.detach().cpu()
    return best_logits, best_val


def ensemble_seeds(x, edge_index, y, train_mask, val_mask, num_classes, seeds, **kw):
    probs_list = []
    val_accs = []
    for s in seeds:
        logits, v = train_seed_best_val(x, edge_index, y, train_mask, val_mask, num_classes, seed=s, **kw)
        probs_list.append(F.softmax(logits, dim=1))
        val_accs.append(v)
    return torch.stack(probs_list).mean(dim=0), val_accs


def main():
    t0 = time.time()
    dataset = Planetoid(root=DATA_ROOT, name="Cora", transform=NormalizeFeatures())
    data = dataset[0]
    num_classes = dataset.num_classes
    print(f"device={device} n_nodes={data.num_nodes} n_feat={data.num_features} n_classes={num_classes}")
    print(f"train={int(data.train_mask.sum())} val={int(data.val_mask.sum())} test={int(data.test_mask.sum())}")

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask_only = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    trainval_mask = (data.train_mask | data.val_mask).to(device)

    # Round 1: train on train+val, select best by val (noisy but breaks ties), ensemble 20 seeds
    seeds_r1 = list(range(20))
    avg_probs1, vaccs1 = ensemble_seeds(x, edge_index, y, trainval_mask, val_mask, num_classes, seeds_r1)
    pred1 = avg_probs1.argmax(dim=1).to(device)
    test_acc1 = (pred1[test_mask] == y[test_mask]).float().mean().item()
    print(f"ROUND1 test_acc={test_acc1:.4f}  mean_val_sel={np.mean(vaccs1):.4f}  ({time.time()-t0:.1f}s)")

    # Round 2: self-train with high-conf pseudo-labels on unlabeled
    labeled_or_test = (data.train_mask | data.val_mask | data.test_mask).to(device)
    unlabeled = ~labeled_or_test
    max_conf, pseudo = avg_probs1.max(dim=1)
    max_conf = max_conf.to(device); pseudo = pseudo.to(device)
    conf_thresh = 0.85
    pseudo_mask = unlabeled & (max_conf >= conf_thresh)
    print(f"pseudo-labeled {int(pseudo_mask.sum())} of {int(unlabeled.sum())} unlabeled (conf>={conf_thresh})")

    y2 = y.clone(); y2[pseudo_mask] = pseudo[pseudo_mask]
    train_mask_r2 = trainval_mask | pseudo_mask
    seeds_r2 = list(range(20, 40))
    avg_probs2, vaccs2 = ensemble_seeds(x, edge_index, y2, train_mask_r2, val_mask, num_classes, seeds_r2)
    pred2 = avg_probs2.argmax(dim=1).to(device)
    test_acc2 = (pred2[test_mask] == y[test_mask]).float().mean().item()
    print(f"ROUND2 test_acc={test_acc2:.4f}  mean_val_sel={np.mean(vaccs2):.4f}  ({time.time()-t0:.1f}s)")

    # Blend
    final_probs = 0.5 * avg_probs1 + 0.5 * avg_probs2
    pred = final_probs.argmax(dim=1).to(device)
    test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
    val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
    train_acc = (pred[train_mask_only] == y[train_mask_only]).float().mean().item()
    elapsed = time.time() - t0
    print(f"BLEND  test_acc={test_acc:.4f}  val={val_acc:.4f}  train={train_acc:.4f}  ({elapsed:.1f}s)")

    # submission.csv
    test_idx_list = data.test_mask.nonzero(as_tuple=False).view(-1).tolist()
    import csv
    sub_path = WORK_DIR / "submission.csv"
    pred_cpu = pred.detach().cpu()
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "predicted_class"])
        for ni in test_idx_list:
            w.writerow([ni, int(pred_cpu[ni].item())])

    result = {
        "target_column_chosen": "predicted_class",
        "headline_metric": "test_accuracy",
        "headline_value": float(test_acc),
        "all_metrics": {
            "test_accuracy": float(test_acc),
            "test_accuracy_round1": float(test_acc1),
            "test_accuracy_round2": float(test_acc2),
            "val_accuracy": float(val_acc),
            "train_accuracy": float(train_acc),
            "n_pseudo_labeled": int(pseudo_mask.sum().item()),
            "n_seeds_per_round": len(seeds_r1),
            "runtime_sec": float(elapsed),
        },
        "skills_applied": [],
        "interpretation_notes": (
            "2-layer GCN (PyG GCNConv, row-normalized features). Trained on train+val "
            "(spec only constrains test acc), best-val-checkpoint ensembled over 20 seeds. "
            "Self-training round: high-confidence pseudo-labels (>=0.85) on unlabeled nodes, "
            "retrained 20-seed ensemble, blended with round 1. "
            "v4 skill library is tabular-focused; no preconditions match a graph-NN task."
        ),
    }
    with open(WORK_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", sub_path, "and result.json")


if __name__ == "__main__":
    main()
