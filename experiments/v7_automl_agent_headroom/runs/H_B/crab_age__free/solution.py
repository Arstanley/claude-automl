"""
Crab Age regression — RMSLE.

Skills applied:
- skill_tune_when_n_train_supports_it (n_train=59240 >> 500, no size/latency tokens)
- skill_verify_split_claim_before_holdout (no claim; standard KFold)
- skill_probe_group_structure_in_ids (probed; no id overlap, features are continuous physical
  measurements — no group structure, standard KFold is fine)

Approach: HistGradientBoostingRegressor on log1p(Age) so MSE on log-target == RMSLE on Age.
Small 3-config grid + 4-fold CV. Two best configs averaged for the submission.
"""
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
T0 = time.time()

DATA = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/crab_age/data")
WORK = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/runs/H_B/crab_age__free")
WORK.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

TARGET = "Age"
ID_COL = "id"

# Features: physical measurements + Sex (one-hot)
feat_num = ["Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight"]
X_all = pd.concat([train[feat_num + ["Sex"]], test[feat_num + ["Sex"]]], axis=0, ignore_index=True)
X_all = pd.get_dummies(X_all, columns=["Sex"], drop_first=False)

# A couple of cheap, well-motivated engineered features (volume + density ratios)
X_all["Volume"] = X_all["Length"] * X_all["Diameter"] * X_all["Height"]
X_all["MeatRatio"] = X_all["Shucked Weight"] / (X_all["Weight"] + 1e-9)
X_all["ShellRatio"] = X_all["Shell Weight"] / (X_all["Weight"] + 1e-9)

X = X_all.iloc[: len(train)].reset_index(drop=True)
X_test = X_all.iloc[len(train):].reset_index(drop=True)
y = train[TARGET].astype(float).values
y_log = np.log1p(y)

# --- skill_tune_when_n_train_supports_it: small 3-config grid ---
configs = [
    dict(learning_rate=0.05, max_iter=600, max_depth=None, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=0.0),
    dict(learning_rate=0.05, max_iter=800, max_depth=None, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0),
    dict(learning_rate=0.03, max_iter=1000, max_depth=None, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
]

N_SPLITS = 4
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cv_scores = []
oof_per_cfg = []
test_per_cfg = []

for i, cfg in enumerate(configs):
    oof = np.zeros(len(X))
    tpred = np.zeros(len(X_test))
    fold_rmsle = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        m = HistGradientBoostingRegressor(
            loss="squared_error",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=42 + fold,
            **cfg,
        )
        m.fit(X.iloc[tr_idx], y_log[tr_idx])
        p_log_va = m.predict(X.iloc[va_idx])
        oof[va_idx] = p_log_va
        tpred += m.predict(X_test) / N_SPLITS
        # RMSLE on original scale = RMSE on log1p scale
        rmsle_fold = float(np.sqrt(mean_squared_error(y_log[va_idx], p_log_va)))
        fold_rmsle.append(rmsle_fold)
    cv_rmsle = float(np.sqrt(mean_squared_error(y_log, oof)))
    print(f"cfg {i}: CV RMSLE={cv_rmsle:.5f}  folds={['%.5f'%x for x in fold_rmsle]}  elapsed={time.time()-T0:.1f}s")
    cv_scores.append(cv_rmsle)
    oof_per_cfg.append(oof)
    test_per_cfg.append(tpred)
    if time.time() - T0 > 100:  # leave buffer in 120s budget
        print("time budget guard hit — stopping grid")
        break

# Pick best 2 configs, average their test predictions
order = np.argsort(cv_scores)
top_k = min(2, len(cv_scores))
top_idx = order[:top_k]
final_log = np.mean([test_per_cfg[i] for i in top_idx], axis=0)
final_oof_log = np.mean([oof_per_cfg[i] for i in top_idx], axis=0)
ens_rmsle = float(np.sqrt(mean_squared_error(y_log, final_oof_log)))

final_pred = np.expm1(final_log)
final_pred = np.clip(final_pred, 1.0, 40.0)  # Age range 1..29 in train

sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: final_pred})
sub.to_csv(WORK / "submission.csv", index=False)

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "RMSLE",
    "headline_value": ens_rmsle,
    "all_metrics": {
        "cv_rmsle_per_config": cv_scores,
        "ensemble_oof_rmsle": ens_rmsle,
        "top_configs": [int(i) for i in top_idx],
        "n_train": int(len(X)),
        "n_test": int(len(X_test)),
        "n_splits": N_SPLITS,
        "elapsed_sec": round(time.time() - T0, 2),
    },
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
        "skill_verify_split_claim_before_holdout.md",
        "skill_probe_group_structure_in_ids.md",
    ],
    "interpretation_notes": (
        "HistGradientBoostingRegressor on log1p(Age) so MSE-on-log equals RMSLE-on-Age. "
        "3-config small grid x 4-fold KFold (no group structure probed in id or features); "
        "top 2 configs averaged for the submission. Added Volume and meat/shell ratio features."
    ),
}
with open(WORK / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"DONE ensemble RMSLE={ens_rmsle:.5f} elapsed={time.time()-T0:.1f}s")
print(f"submission rows={len(sub)}  first:\n{sub.head()}")
