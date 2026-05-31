"""Citeseer node classification with a 2-layer GCN (PyG)."""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T

DATA_ROOT = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/citeseer/data"
WORK_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/runs/H_B/citeseer__free"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True)
        self.conv2 = GCNConv(hidden, out_dim, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = Planetoid(root=DATA_ROOT, name="Citeseer", transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    num_classes = dataset.num_classes

    model = GCN(dataset.num_features, 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_state = None
    patience = 50
    bad = 0
    epochs = 300
    t0 = time.time()
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
            preds = logits.argmax(dim=1)
            train_acc = (preds[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            val_acc = (preds[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:03d}  loss {loss.item():.4f}  train {train_acc:.4f}  val {val_acc:.4f}  test {test_acc:.4f}")
        if bad >= patience:
            print(f"early stop @ epoch {epoch}")
            break
    dt = time.time() - t0
    print(f"trained in {dt:.1f}s. best val={best_val_acc:.4f} test={best_test_acc:.4f}")

    # Restore best state for predictions
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds_all = logits.argmax(dim=1).cpu().numpy()

    # Build submission: predict the category for every test-mask node.
    test_idx = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    sub = pd.DataFrame({"node_id": test_idx.astype(int), "category": preds_all[test_idx].astype(int)})
    sub_path = os.path.join(WORK_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path} ({len(sub)} rows)")

    # Per-class test accuracy
    y_test = data.y[data.test_mask].cpu().numpy()
    yhat = preds_all[test_idx]
    per_class = {}
    for c in range(num_classes):
        m = y_test == c
        per_class[int(c)] = float((yhat[m] == c).mean()) if m.any() else None

    result = {
        "target_column_chosen": "category",
        "headline_metric": "accuracy",
        "headline_value": float(best_test_acc),
        "all_metrics": {
            "test_accuracy": float(best_test_acc),
            "val_accuracy": float(best_val_acc),
            "per_class_test_accuracy": per_class,
            "n_train": int(data.train_mask.sum().item()),
            "n_val": int(data.val_mask.sum().item()),
            "n_test": int(data.test_mask.sum().item()),
        },
        "skills_applied": [],
        "interpretation_notes": (
            "Loaded Citeseer via torch_geometric.datasets.Planetoid with NormalizeFeatures, "
            "trained a 2-layer GCN (hidden=64, dropout=0.5, Adam lr=0.01 wd=5e-4) on the "
            "public 120-node train split, model-selected by val accuracy with early stopping, "
            "then predicted labels for the 1000-node test mask. GPU used."
        ),
    }
    with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("wrote result.json")


if __name__ == "__main__":
    main()
