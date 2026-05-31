"""
Bike Sharing Demand — RMSLE.

Skills applied (see harness_H_1/skills/INDEX.md):
- skill_verify_split_claim_before_holdout: prompt claims days 1-19 train / 20+ test,
  but both train and test contain only days 1-19 with no datetime overlap. Random
  KFold (not temporal) is the correct CV.
- skill_log_target_for_skewed_nonnegative_regression: regression, y>=0, metric is
  RMSLE exactly, skew(y)=1.24. Fit on log1p(count), predict expm1 clipped to >=0.
- skill_blend_two_model_families: tabular, n=8708>=1000, GBM primary, only a
  training-time budget. Blend HistGradientBoostingRegressor + ExtraTreesRegressor;
  blend weight chosen by OOF RMSLE.
- skill_tune_when_n_train_supports_it (light variant): under the 2-min training
  budget, use a small explicit 3-config HGB grid (not full tuning).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold


DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/bike-sharing-demand/data/")
WORKING_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_1/bike-sharing-demand/")
N_JOBS = 2
N_SPLITS = 5
SEED = 42


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    # cyclical encodings — help linear/tree models still benefit from smoothness in hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # rush-hour indicators (interact hour x workingday)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["workingday_rush"] = df["is_rush_hour"] * df["workingday"]
    return df


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # Verify split claim: prompt says train=days 1-19, test=20-end. Reality: both have days 1-19.
    train_days = pd.to_datetime(train["datetime"]).dt.day
    test_days = pd.to_datetime(test["datetime"]).dt.day
    split_claim_ok = bool(train_days.max() <= 19 and test_days.min() >= 20)
    print(f"[verify_split] train_day_max={train_days.max()} test_day_min={test_days.min()} claim_holds={split_claim_ok}")
    print(f"[verify_split] -> using random KFold(n_splits={N_SPLITS}), NOT temporal CV")

    train_fe = add_datetime_features(train)
    test_fe = add_datetime_features(test)

    feature_cols = [
        "season", "holiday", "workingday", "weather",
        "temp", "atemp", "humidity", "windspeed",
        "year", "month", "day", "hour", "dayofweek", "dayofyear", "weekofyear",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_rush_hour", "workingday_rush",
    ]
    X = train_fe[feature_cols].values
    y = train_fe["count"].values.astype(float)
    y_log = np.log1p(y)
    X_test = test_fe[feature_cols].values

    # Skill 9: A/B raw vs log target on HGB baseline
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # --- Light tuning grid for HGB (skill 7, small under time budget) ---
    hgb_grid = [
        dict(max_iter=500, learning_rate=0.05, max_depth=None, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.0),
        dict(max_iter=700, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.1),
        dict(max_iter=400, learning_rate=0.08, max_depth=None, max_leaf_nodes=127, min_samples_leaf=10, l2_regularization=0.0),
    ]

    def cv_hgb_log(params):
        oof = np.zeros(len(y))
        for fold_idx, (tr, va) in enumerate(cv.split(X)):
            m = HistGradientBoostingRegressor(random_state=SEED, **params)
            m.fit(X[tr], y_log[tr])
            oof[va] = np.expm1(m.predict(X[va]))
        return rmsle(y, np.clip(oof, 0, None)), oof

    def cv_hgb_raw(params):
        oof = np.zeros(len(y))
        for fold_idx, (tr, va) in enumerate(cv.split(X)):
            m = HistGradientBoostingRegressor(random_state=SEED, **params)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
        return rmsle(y, np.clip(oof, 0, None)), oof

    # A/B test: raw vs log on the first (default-ish) config
    raw_score, _ = cv_hgb_raw(hgb_grid[0])
    log_score, _ = cv_hgb_log(hgb_grid[0])
    print(f"[ab_log_target] HGB raw={raw_score:.5f} log={log_score:.5f} -> {'log' if log_score < raw_score else 'raw'}")
    use_log = log_score < raw_score

    cv_fn = cv_hgb_log if use_log else cv_hgb_raw

    # Tune HGB
    hgb_results = []
    for i, p in enumerate(hgb_grid):
        s, oof_i = cv_fn(p)
        hgb_results.append((s, i, p, oof_i))
        print(f"[tune_hgb] cfg{i} score={s:.5f} params={p}")
    hgb_results.sort(key=lambda r: r[0])
    best_hgb_score, best_idx, best_hgb_params, best_hgb_oof = hgb_results[0]
    print(f"[tune_hgb] BEST cfg{best_idx} score={best_hgb_score:.5f}")

    # --- Second family: ExtraTreesRegressor (skill 10) ---
    et_params = dict(n_estimators=400, min_samples_leaf=2, max_features=0.7, n_jobs=N_JOBS, random_state=SEED)
    et_oof = np.zeros(len(y))
    for tr, va in cv.split(X):
        m = ExtraTreesRegressor(**et_params)
        m.fit(X[tr], y_log[tr] if use_log else y[tr])
        pred = m.predict(X[va])
        et_oof[va] = np.expm1(pred) if use_log else pred
    et_oof = np.clip(et_oof, 0, None)
    et_score = rmsle(y, et_oof)
    print(f"[second_family] ExtraTrees score={et_score:.5f}")

    # Blend weight search on OOF (skill 10)
    best_hgb_oof_clipped = np.clip(best_hgb_oof, 0, None)
    best_w, best_blend = 1.0, best_hgb_score
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * best_hgb_oof_clipped + (1 - w) * et_oof
        s = rmsle(y, np.clip(blend, 0, None))
        if s < best_blend:
            best_blend, best_w = s, float(w)
    print(f"[blend] best_weight_hgb={best_w:.2f} blend_score={best_blend:.5f}")
    use_blend = 0.0 < best_w < 1.0
    if not use_blend:
        print(f"[blend] degenerate weight ({best_w}) — shipping single best model")

    # --- Refit on full data and predict on test ---
    hgb_final = HistGradientBoostingRegressor(random_state=SEED, **best_hgb_params)
    hgb_final.fit(X, y_log if use_log else y)
    hgb_test_pred = hgb_final.predict(X_test)
    if use_log:
        hgb_test_pred = np.expm1(hgb_test_pred)
    hgb_test_pred = np.clip(hgb_test_pred, 0, None)

    if use_blend or best_w == 0.0:  # need ET only if we'll use it
        et_final = ExtraTreesRegressor(**et_params)
        et_final.fit(X, y_log if use_log else y)
        et_test_pred = et_final.predict(X_test)
        if use_log:
            et_test_pred = np.expm1(et_test_pred)
        et_test_pred = np.clip(et_test_pred, 0, None)
    else:
        et_test_pred = None

    if use_blend:
        final_pred = best_w * hgb_test_pred + (1 - best_w) * et_test_pred
        headline = best_blend
    elif best_w == 1.0:
        final_pred = hgb_test_pred
        headline = best_hgb_score
    else:  # best_w == 0.0
        final_pred = et_test_pred
        headline = et_score
    final_pred = np.clip(final_pred, 0, None)

    sub = pd.DataFrame({
        "datetime": test["datetime"].values,
        "count": final_pred,
    })
    sub.to_csv(WORKING_DIR / "submission.csv", index=False)
    print(f"[submit] wrote submission.csv ({len(sub)} rows)")

    skills_applied = [
        "skill_verify_split_claim_before_holdout.md",
        "skill_log_target_for_skewed_nonnegative_regression.md",
        "skill_tune_when_n_train_supports_it.md",
        "skill_blend_two_model_families.md" if use_blend else "skill_blend_two_model_families.md (evaluated, blend degenerate)",
    ]
    skills_evaluated_not_applied = [
        "skill_probe_group_structure_in_ids.md: no compound IDs, no value overlap (datetime overlap=0)",
        "skill_train_on_listed_auxiliary_files.md: prompt names no auxiliary file",
        "skill_stringify_low_cardinality_int_codes.md: model is tree-based (HGB/ET), not scaler/linear/distance",
        "skill_multi_axis_constraint_budgets.md: only one constraint axis (training time)",
        "skill_missing_indicator_when_missingness_signals_class.md: zero NaNs in train and test",
        "skill_tune_decision_threshold_for_f1.md: regression task, not binary classification",
    ]

    result = {
        "target_column_chosen": "count",
        "headline_metric": "rmsle",
        "headline_value": float(headline),
        "all_metrics": {
            "cv_rmsle_hgb_raw_target": float(raw_score),
            "cv_rmsle_hgb_log_target": float(log_score),
            "cv_rmsle_hgb_best": float(best_hgb_score),
            "cv_rmsle_extratrees": float(et_score),
            "cv_rmsle_blend": float(best_blend),
            "blend_weight_hgb": float(best_w),
            "used_log_target": bool(use_log),
            "split_claim_in_prompt_held": bool(split_claim_ok),
        },
        "skills_applied": skills_applied,
        "skills_evaluated_not_applied": skills_evaluated_not_applied,
        "interpretation_notes": (
            "Verified the prompt's day-1-19/day-20+ split claim is FALSE on this data — both "
            "train and test contain only days 1-19 with no datetime overlap, so a random "
            "5-fold KFold is correct (not temporal CV). Fit HGB on log1p(count) (skew=1.24, "
            "RMSLE metric) with a small 3-config tuning grid, then blended with ExtraTrees on "
            "OOF; blend weight chosen by minimizing CV RMSLE."
        ),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(WORKING_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] headline_rmsle={headline:.5f} runtime={result['runtime_seconds']}s")


if __name__ == "__main__":
    main()
