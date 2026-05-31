"""Solution for playground-series-s3e18 (predict EC1, EC2 probabilities, mean ROC AUC)."""
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e18/data"

t0 = time.time()

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

TARGETS = ["EC1", "EC2"]
DROP_COLS = ["id", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]
FEATURES = [c for c in train.columns if c not in DROP_COLS]

X_train = train[FEATURES].values
X_test = test[FEATURES].values

# Light grid of HGB configs (skill: tune_when_n_train_supports_it; n=11870, no size/latency constraint)
CONFIGS = [
    dict(max_depth=None, max_iter=400, learning_rate=0.05, min_samples_leaf=20, l2_regularization=0.0),
    dict(max_depth=6,    max_iter=500, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0),
    dict(max_depth=8,    max_iter=400, learning_rate=0.05, min_samples_leaf=20, l2_regularization=0.5),
]

SEEDS = [0, 1]  # seed-bag final ensemble
N_SPLITS = 5

cv_scores = {t: [] for t in TARGETS}
test_pred = {t: np.zeros(len(test)) for t in TARGETS}

for tgt in TARGETS:
    y = train[tgt].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    # OOF for tuning: pick best config by mean OOF AUC, then fit full+bag for test
    cfg_scores = []
    for cfg in CONFIGS:
        oof = np.zeros(len(train))
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
            model = HistGradientBoostingClassifier(random_state=0, early_stopping=False, **cfg)
            model.fit(X_train[tr_idx], y[tr_idx])
            oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
        score = roc_auc_score(y, oof)
        cfg_scores.append((score, cfg))
        print(f"[{tgt}] cfg={cfg} OOF AUC={score:.5f}", flush=True)
        if time.time() - t0 > 90:
            break
    cfg_scores.sort(key=lambda x: -x[0])
    best_score, best_cfg = cfg_scores[0]
    cv_scores[tgt] = best_score
    print(f"[{tgt}] best={best_score:.5f} cfg={best_cfg}", flush=True)

    # Seed-bag final fit on full train
    preds = []
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(random_state=sd, early_stopping=False, **best_cfg)
        m.fit(X_train, y)
        preds.append(m.predict_proba(X_test)[:, 1])
    test_pred[tgt] = np.mean(preds, axis=0)

mean_auc = float(np.mean(list(cv_scores.values())))
print(f"Mean OOF AUC = {mean_auc:.5f}", flush=True)

sub = pd.DataFrame({"id": test["id"].values, "EC1": test_pred["EC1"], "EC2": test_pred["EC2"]})
sub.to_csv(os.path.join(HERE, "submission.csv"), index=False)

result = {
    "target_column_chosen": "EC1,EC2",
    "headline_metric": "mean_roc_auc_oof",
    "headline_value": mean_auc,
    "all_metrics": {f"oof_auc_{t}": float(cv_scores[t]) for t in TARGETS} | {"mean_oof_auc": mean_auc},
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
        "skill_seed_bag_final_regressor.md",
    ],
    "interpretation_notes": (
        "Multi-label binary (EC1, EC2) on 31 numeric molecular descriptors. "
        "No missing values, no group structure, no listed auxiliary file path. "
        "Per-target HistGradientBoostingClassifier, small grid tuned by 5-fold OOF AUC, "
        "best config refit on full train with 2-seed bag for stability."
    ),
    "elapsed_sec": time.time() - t0,
}
with open(os.path.join(HERE, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
print("Wrote submission.csv and result.json", flush=True)
