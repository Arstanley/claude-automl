"""
Mohs Hardness Regression (playground-series-s3e25)
Metric: Median Absolute Error (MedAE)
Strategy:
- HistGradientBoostingRegressor with `loss='absolute_error'` (LAD aligns with MedAE better than squared loss).
- Small explicit hyper-parameter grid via 5-fold CV (skill_tune_when_n_train_supports_it; n_train=8325 >> 500).
- Seed-bag the final fit over 3 seeds for cheap variance reduction (skill_seed_bag_final_regressor).
"""
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import median_absolute_error, mean_absolute_error

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e25/data"

t0 = time.time()
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

TARGET = "Hardness"
ID = "id"
feature_cols = [c for c in train.columns if c not in (TARGET, ID)]
X = train[feature_cols].values
y = train[TARGET].values
X_test = test[feature_cols].values
test_ids = test[ID].values

print(f"n_train={len(train)}, n_test={len(test)}, n_features={len(feature_cols)}")
print(f"target nunique={train[TARGET].nunique()}, range=[{y.min():.3f}, {y.max():.3f}]")

# --- 5-fold CV, small explicit grid ---
grid = [
    dict(loss="absolute_error", max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
    dict(loss="absolute_error", max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=0.0),
    dict(loss="absolute_error", max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.0),
    dict(loss="absolute_error", max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.0),
]

cv = KFold(n_splits=5, shuffle=True, random_state=0)
results = []
for i, params in enumerate(grid):
    fold_medae = []
    fold_mae = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X)):
        m = HistGradientBoostingRegressor(**params, random_state=0, early_stopping=False)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[va_idx])
        fold_medae.append(median_absolute_error(y[va_idx], pred))
        fold_mae.append(mean_absolute_error(y[va_idx], pred))
    mean_medae = float(np.mean(fold_medae))
    mean_mae = float(np.mean(fold_mae))
    print(f"  cfg{i}: MedAE={mean_medae:.4f} ± {np.std(fold_medae):.4f}  MAE={mean_mae:.4f}  params={params}")
    results.append((mean_medae, mean_mae, params))

results.sort(key=lambda r: r[0])
best_medae, best_mae, best_params = results[0]
print(f"\nBest cfg by CV MedAE: {best_medae:.4f}, params={best_params}")

# --- Seed-bag final fit ---
seeds = [0, 1, 2]
test_preds = []
for s in seeds:
    m = HistGradientBoostingRegressor(**best_params, random_state=s, early_stopping=False)
    m.fit(X, y)
    test_preds.append(m.predict(X_test))
final_pred = np.mean(test_preds, axis=0)

# --- Compute OOF MedAE under best config for headline (seed 0) ---
oof = np.zeros(len(y))
for fold, (tr_idx, va_idx) in enumerate(cv.split(X)):
    m = HistGradientBoostingRegressor(**best_params, random_state=0, early_stopping=False)
    m.fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = m.predict(X[va_idx])
oof_medae = float(median_absolute_error(y, oof))
oof_mae = float(mean_absolute_error(y, oof))
print(f"OOF (seed 0, best cfg): MedAE={oof_medae:.4f}  MAE={oof_mae:.4f}")

# --- Write submission ---
sub = pd.DataFrame({ID: test_ids, TARGET: final_pred})
sub_path = os.path.join(HERE, "submission.csv")
sub.to_csv(sub_path, index=False)

# --- Write result.json ---
result = {
    "target_column_chosen": TARGET,
    "headline_metric": "MedAE",
    "headline_value": oof_medae,
    "all_metrics": {
        "cv_medae_best": best_medae,
        "cv_mae_best": best_mae,
        "oof_medae_seed0": oof_medae,
        "oof_mae_seed0": oof_mae,
    },
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
        "skill_seed_bag_final_regressor.md",
    ],
    "interpretation_notes": (
        "HistGradientBoostingRegressor with absolute_error loss (LAD aligns with MedAE). "
        "Picked best of 4 configs via 5-fold CV on MedAE, then seed-bagged (seeds 0,1,2) the "
        "full-data refit. All-numeric features, no missingness, n_train=8325 supports tuning."
    ),
    "audit": {
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(feature_cols),
        "hyperparameter_grid": grid,
        "chosen_config": best_params,
        "seed_bag": seeds,
        "wall_seconds": round(time.time() - t0, 2),
    },
}
with open(os.path.join(HERE, "result.json"), "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"\nWrote {sub_path} ({len(sub)} rows)")
print(f"Headline MedAE (OOF, seed 0): {oof_medae:.4f}")
print(f"Wall: {time.time() - t0:.1f}s")
