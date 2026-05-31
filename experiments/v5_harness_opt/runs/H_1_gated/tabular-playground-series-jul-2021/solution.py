"""
Tabular Playground July 2021 — air pollution regression (3 targets).
Metric: mean column-wise RMSLE over 3 targets.

Skills applied:
- skill_verify_split_claim_before_holdout: probed train/test date ranges — they
  fully overlap (both span 2010-03..2011-01), so use random KFold, not temporal.
- skill_tune_when_n_train_supports_it: n_train=5688 >> 500, no constraint tokens
  in prompt; tune a small explicit grid for HistGradientBoostingRegressor.
- skill_seed_bag_final_regressor: average final predictions over 3 random seeds.
"""

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
DATA = Path(
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/"
    "tabular-playground-series-jul-2021/data"
)
TARGETS = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["date_time"])
    out = df.drop(columns=["date_time"]).copy()
    out["hour"] = dt.dt.hour
    out["day"] = dt.dt.day
    out["month"] = dt.dt.month
    out["dayofweek"] = dt.dt.dayofweek
    out["dayofyear"] = dt.dt.dayofyear
    # cyclical encoding
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"train={train.shape} test={test.shape}")

    # ---- Skill: verify split claim ----
    tr_dt = pd.to_datetime(train["date_time"])
    te_dt = pd.to_datetime(test["date_time"])
    split_audit = {
        "train_range": [str(tr_dt.min()), str(tr_dt.max())],
        "test_range": [str(te_dt.min()), str(te_dt.max())],
        "ranges_overlap": bool(tr_dt.max() >= te_dt.min() and te_dt.max() >= tr_dt.min()),
    }
    print("Split audit:", split_audit)
    # → ranges overlap → random KFold is appropriate.

    X = add_time_feats(train.drop(columns=TARGETS))
    X_test = add_time_feats(test)
    feat_cols = X.columns.tolist()
    print(f"features: {feat_cols}")

    # ---- Skill: tune (small grid). Use early-stopping to keep budget tight. ----
    grid = [
        dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
             min_samples_leaf=20, l2_regularization=0.0),
        dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
             min_samples_leaf=20, l2_regularization=1.0),
        dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
             min_samples_leaf=10, l2_regularization=0.0),
    ]

    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    per_target_results = {}
    best_params_per_target = {}

    for tgt in TARGETS:
        y = train[tgt].values
        y_log = np.log1p(y)  # train on log1p to match RMSLE
        cfg_scores = []
        for cfg in grid:
            oof = np.zeros(len(X))
            for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X)):
                m = HistGradientBoostingRegressor(
                    **cfg, random_state=0,
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                )
                m.fit(X.iloc[tr_idx], y_log[tr_idx])
                oof[va_idx] = m.predict(X.iloc[va_idx])
            oof_exp = np.clip(np.expm1(oof), 0, None)
            score = rmsle(y, oof_exp)
            cfg_scores.append((score, cfg))
            print(f"  {tgt} cfg={cfg} -> RMSLE={score:.5f}")
        cfg_scores.sort(key=lambda x: x[0])
        best_score, best_cfg = cfg_scores[0]
        per_target_results[tgt] = {
            "best_oof_rmsle": best_score,
            "all_cfgs": [{"rmsle": s, "cfg": c} for s, c in cfg_scores],
        }
        best_params_per_target[tgt] = best_cfg
        print(f"  -> best for {tgt}: {best_cfg} (RMSLE={best_score:.5f})")

    # ---- Skill: seed-bag final fits, 3 seeds, average in log space ----
    seeds = [0, 1, 2]
    preds_per_target = {}
    for tgt in TARGETS:
        y_log = np.log1p(train[tgt].values)
        cfg = best_params_per_target[tgt]
        bag = []
        for s in seeds:
            m = HistGradientBoostingRegressor(
                **cfg, random_state=s,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            )
            m.fit(X, y_log)
            bag.append(m.predict(X_test))
        mean_log_pred = np.mean(bag, axis=0)
        preds_per_target[tgt] = np.clip(np.expm1(mean_log_pred), 0, None)

    sub = pd.DataFrame({"date_time": test["date_time"]})
    for tgt in TARGETS:
        sub[tgt] = preds_per_target[tgt]
    sub.to_csv(HERE / "submission.csv", index=False)
    print(f"Wrote submission.csv: {sub.shape}")

    mean_oof = float(np.mean([per_target_results[t]["best_oof_rmsle"] for t in TARGETS]))
    result = {
        "target_column_chosen": TARGETS,
        "headline_metric": "mean_column_wise_RMSLE",
        "headline_value": mean_oof,
        "all_metrics": {
            f"oof_rmsle_{t}": per_target_results[t]["best_oof_rmsle"] for t in TARGETS
        },
        "skills_applied": [
            "skill_verify_split_claim_before_holdout.md",
            "skill_tune_when_n_train_supports_it.md",
            "skill_seed_bag_final_regressor.md",
        ],
        "interpretation_notes": (
            "Train/test date ranges fully overlap, so random KFold (not temporal). "
            "Targets >0, so trained HistGBM on log1p(target) and expm1 the predictions "
            "(direct alignment with RMSLE). Per-target small-grid tuned, then averaged "
            "3 seeds for the final fit on full train."
        ),
        "audit": {
            "split": split_audit,
            "best_params_per_target": best_params_per_target,
            "per_target_oof": per_target_results,
            "seed_bag": seeds,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "wall_time_s": round(time.time() - t0, 2),
        },
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote result.json. mean OOF RMSLE = {mean_oof:.5f}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
