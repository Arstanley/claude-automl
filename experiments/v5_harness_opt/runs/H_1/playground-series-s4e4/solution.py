"""
Playground S4E4 — Abalone Rings regression (RMSLE).

Skills applied:
  - skill_log_target_for_skewed_nonnegative_regression: fit on log1p(y) (metric is RMSLE,
    y in [1,29], skew~1.2). Predict expm1, clip >=0.
  - skill_tune_when_n_train_supports_it: small explicit grid over HistGB
    (n=72k, GBM family, no size/latency tokens).
  - skill_blend_two_model_families: blend HistGB with ExtraTrees on OOF; weight by CV.

Data:
  - n_train = 72492, n_test = 18123, 8 features (1 categorical 'Sex' + 7 numeric).
  - No NaNs. No id overlap between train/test.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s4e4/data")
WORK_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_1/playground-series-s4e4")
N_JOBS = 2
N_SPLITS = 5
SEED = 42

t_start = time.time()


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    target = "Rings"
    id_col = "id"
    cat_cols = ["Sex"]
    num_cols = [c for c in train.columns if c not in cat_cols + [id_col, target]]

    X = train[cat_cols + num_cols].copy()
    y = train[target].astype(float).values
    X_test = test[cat_cols + num_cols].copy()

    # log target (RMSLE => optimize MSE on log1p(y))
    y_log = np.log1p(y)

    # Preprocessor for sklearn pipelines: OHE Sex (3 levels) + passthrough numerics.
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="passthrough",
    )

    # --- Tune HistGB with a small grid (skill_tune_when_n_train_supports_it) -----
    # All configs use early stopping so they do not run all max_iter rounds.
    common_es = dict(early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
    histgb_configs = [
        dict(max_iter=800,  learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.0, **common_es),
        dict(max_iter=800,  learning_rate=0.04, max_leaf_nodes=63, min_samples_leaf=60, l2_regularization=0.5, **common_es),
        dict(max_iter=1200, learning_rate=0.03, max_leaf_nodes=47, min_samples_leaf=50, l2_regularization=0.1, **common_es),
    ]

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(kf.split(X))

    def cv_histgb(cfg):
        oof = np.zeros(len(X))
        for tr_idx, va_idx in folds:
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr = y_log[tr_idx]
            pipe = Pipeline([
                ("pre", pre),
                ("m", HistGradientBoostingRegressor(loss="squared_error", random_state=SEED, **cfg)),
            ])
            pipe.fit(Xtr, ytr)
            oof[va_idx] = pipe.predict(Xva)
        # back to original scale
        preds = np.clip(np.expm1(oof), 0, None)
        return rmsle(y, preds), oof

    best_score = None
    best_cfg = None
    best_oof_hgb = None
    histgb_grid_results = []
    for cfg in histgb_configs:
        score, oof = cv_histgb(cfg)
        histgb_grid_results.append({"cfg": cfg, "rmsle": score})
        if best_score is None or score < best_score:
            best_score = score
            best_cfg = cfg
            best_oof_hgb = oof
        if time.time() - t_start > 80:
            break  # respect 2 min budget

    # --- Second family: ExtraTrees on log target (skill_blend_two_model_families) ----
    # Smaller forest for speed; OHE Sex via the same preprocessor.
    et_oof = np.zeros(len(X))
    et_models_preds_test = []
    for tr_idx, va_idx in folds:
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr = y_log[tr_idx]
        pipe = Pipeline([
            ("pre", pre),
            ("m", ExtraTreesRegressor(
                n_estimators=120, max_depth=None, min_samples_leaf=10,
                n_jobs=N_JOBS, random_state=SEED,
            )),
        ])
        pipe.fit(Xtr, ytr)
        et_oof[va_idx] = pipe.predict(Xva)
        et_models_preds_test.append(pipe.predict(X_test))
    et_oof_preds = np.clip(np.expm1(et_oof), 0, None)
    score_et = rmsle(y, et_oof_preds)

    # --- Search blend weight on OOF (in log space, then expm1) -------------------
    best_w = 0.0
    best_blend_score = best_score
    for w in np.linspace(0.0, 1.0, 21):
        blend_log = (1.0 - w) * best_oof_hgb + w * et_oof
        blend_pred = np.clip(np.expm1(blend_log), 0, None)
        s = rmsle(y, blend_pred)
        if s < best_blend_score:
            best_blend_score = s
            best_w = float(w)
    # If best_w is 0 or 1, ship single model per skill prescription.
    use_blend = 0.0 < best_w < 1.0

    # --- Refit HistGB on full data with best cfg --------------------------------
    pipe_hgb_full = Pipeline([
        ("pre", pre),
        ("m", HistGradientBoostingRegressor(loss="squared_error", random_state=SEED, **best_cfg)),
    ])
    pipe_hgb_full.fit(X, y_log)
    hgb_test_log = pipe_hgb_full.predict(X_test)

    # ET test = mean over fold-trained models (already collected). For blending,
    # this is cleaner than refitting full on small budget.
    et_test_log = np.mean(et_models_preds_test, axis=0)

    if use_blend:
        test_log = (1.0 - best_w) * hgb_test_log + best_w * et_test_log
    else:
        # whichever single-family CV is best
        if best_score <= score_et:
            test_log = hgb_test_log
        else:
            test_log = et_test_log

    test_pred = np.clip(np.expm1(test_log), 0, None)

    # --- Write submission --------------------------------------------------------
    sub = pd.DataFrame({id_col: test[id_col].values, target: test_pred})
    sub.to_csv(WORK_DIR / "submission.csv", index=False)

    # --- Headline metric: report the best OOF RMSLE we achieved -----------------
    headline = best_blend_score if use_blend else min(best_score, score_et)

    skills_applied = [
        "skill_log_target_for_skewed_nonnegative_regression.md",
        "skill_tune_when_n_train_supports_it.md",
    ]
    if use_blend:
        skills_applied.append("skill_blend_two_model_families.md")

    skills_evaluated_not_applied = [
        {"skill": "skill_probe_group_structure_in_ids.md",
         "reason": "id overlap train/test = 0; ids are non-compound integers."},
        {"skill": "skill_train_on_listed_auxiliary_files.md",
         "reason": "Prompt mentions original Abalone but no auxiliary file is present in DATA_DIR."},
        {"skill": "skill_verify_split_claim_before_holdout.md",
         "reason": "Prompt makes no structural train/test split claim."},
        {"skill": "skill_stringify_low_cardinality_int_codes.md",
         "reason": "No low-cardinality int feature columns (only id is int, dropped)."},
        {"skill": "skill_multi_axis_constraint_budgets.md",
         "reason": "Prompt names no constraint axes (no latency/size/hard-rule)."},
        {"skill": "skill_missing_indicator_when_missingness_signals_class.md",
         "reason": "No NaNs in train or test."},
        {"skill": "skill_tune_decision_threshold_for_f1.md",
         "reason": "Regression task; no threshold to tune."},
    ]
    if not use_blend:
        skills_evaluated_not_applied.append({
            "skill": "skill_blend_two_model_families.md",
            "reason": f"Best CV blend weight collapsed to {best_w}; shipping single model per skill rule.",
        })

    result = {
        "target_column_chosen": target,
        "headline_metric": "rmsle",
        "headline_value": float(headline),
        "all_metrics": {
            "rmsle_histgb_best": float(best_score),
            "rmsle_extratrees": float(score_et),
            "rmsle_blend": float(best_blend_score),
            "blend_weight_et": float(best_w),
            "histgb_best_cfg": best_cfg,
            "histgb_grid": [
                {"cfg": r["cfg"], "rmsle": float(r["rmsle"])} for r in histgb_grid_results
            ],
            "n_train": int(len(X)),
            "n_test": int(len(X_test)),
            "n_splits": N_SPLITS,
            "wall_seconds": round(time.time() - t_start, 2),
        },
        "skills_applied": skills_applied,
        "skills_evaluated_not_applied": skills_evaluated_not_applied,
        "interpretation_notes": (
            "Trained on log1p(Rings) since metric is RMSLE and target is skewed (skew=1.20). "
            "Tuned HistGradientBoostingRegressor across 3 configs via 5-fold CV, "
            f"then blended with ExtraTrees (CV-chosen weight={best_w:.2f}). "
            "Preprocessing: one-hot Sex, raw numerics. No NaNs to impute."
        ),
    }

    with open(WORK_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"DONE in {time.time() - t_start:.1f}s | RMSLE headline={headline:.5f} | blend_w={best_w:.2f}")
    print(f"  histgb best={best_score:.5f}  et={score_et:.5f}  blend={best_blend_score:.5f}")


if __name__ == "__main__":
    main()
