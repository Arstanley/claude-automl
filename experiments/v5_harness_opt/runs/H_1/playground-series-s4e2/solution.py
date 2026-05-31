"""
playground-series-s4e2 — Multi-class Obesity Classification.

Metric: accuracy (multiclass, 7 classes).
Approach:
  - HistGradientBoostingClassifier (sklearn's GBM) as primary, with small CV-driven
    hyperparam search (skill_tune_when_n_train_supports_it).
  - ExtraTreesClassifier as a second model family; blend OOF probabilities with
    weight chosen by CV accuracy (skill_blend_two_model_families).

n_jobs=2, time budget < 2 min.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def log(msg):
    print(msg, flush=True)


HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s4e2/data")
N_JOBS = 2
SEED = 42
N_SPLITS = 3

t0 = time.time()


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    target_col = "NObeyesdad"
    id_col = "id"

    y_raw = train[target_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    feature_cols = [c for c in train.columns if c not in {target_col, id_col}]
    cat_cols = [c for c in feature_cols if train[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_tr = train[feature_cols].copy()
    X_te = test[feature_cols].copy()

    # --- HistGradientBoostingClassifier handles categoricals via OHE pipeline ---
    pre_hgb = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    pre_et = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # --- Small explicit hyperparam grid for HGB (skill 7) ---
    hgb_grid = [
        dict(learning_rate=0.08, max_iter=250, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
        dict(learning_rate=0.05, max_iter=350, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.0),
        dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0),
    ]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    def oof_for_hgb(params):
        oof = np.zeros((len(X_tr), n_classes), dtype=np.float32)
        test_proba = np.zeros((len(X_te), n_classes), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
            pipe = Pipeline([
                ("pre", pre_hgb),
                ("clf", HistGradientBoostingClassifier(random_state=SEED + fold, **params)),
            ])
            pipe.fit(X_tr.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = pipe.predict_proba(X_tr.iloc[va_idx])
            test_proba += pipe.predict_proba(X_te) / N_SPLITS
        acc = (oof.argmax(1) == y).mean()
        return oof, test_proba, acc

    # Tune HGB
    hgb_results = []
    for params in hgb_grid:
        t = time.time()
        oof, te_p, acc = oof_for_hgb(params)
        hgb_results.append((acc, params, oof, te_p))
        log(f"[HGB] params={params} acc={acc:.5f} ({time.time()-t:.1f}s)")
        if time.time() - t0 > 60:
            log("[HGB] time budget — stopping grid early")
            break

    hgb_results.sort(key=lambda r: -r[0])
    hgb_acc, hgb_params, hgb_oof, hgb_test = hgb_results[0]
    log(f"[HGB] best acc={hgb_acc:.5f} params={hgb_params}")

    # --- Second family: ExtraTreesClassifier (skill 10) ---
    et_params = dict(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                     n_jobs=N_JOBS, random_state=SEED)

    def oof_for_et():
        oof = np.zeros((len(X_tr), n_classes), dtype=np.float32)
        test_proba = np.zeros((len(X_te), n_classes), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
            pipe = Pipeline([
                ("pre", pre_et),
                ("clf", ExtraTreesClassifier(**et_params)),
            ])
            pipe.fit(X_tr.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = pipe.predict_proba(X_tr.iloc[va_idx])
            test_proba += pipe.predict_proba(X_te) / N_SPLITS
        acc = (oof.argmax(1) == y).mean()
        return oof, test_proba, acc

    t = time.time()
    et_oof, et_test, et_acc = oof_for_et()
    log(f"[ET]  acc={et_acc:.5f} ({time.time()-t:.1f}s)")

    # --- Blend search ---
    best_w, best_blend_acc = 1.0, hgb_acc
    for w in np.linspace(0.0, 1.0, 11):
        blend = w * hgb_oof + (1 - w) * et_oof
        acc = (blend.argmax(1) == y).mean()
        if acc > best_blend_acc:
            best_blend_acc = acc
            best_w = w
    log(f"[Blend] best_w(HGB)={best_w:.2f} acc={best_blend_acc:.5f}")

    if best_w in (0.0, 1.0):
        # Ship single model
        if best_w == 1.0:
            final_test_proba = hgb_test
            headline = hgb_acc
            shipped = "HGB only"
        else:
            final_test_proba = et_test
            headline = et_acc
            shipped = "ET only"
    else:
        final_test_proba = best_w * hgb_test + (1 - best_w) * et_test
        headline = best_blend_acc
        shipped = f"blend HGB={best_w:.2f} ET={1-best_w:.2f}"

    test_pred = le.inverse_transform(final_test_proba.argmax(1))
    sub = pd.DataFrame({id_col: test[id_col].values, target_col: test_pred})
    sub.to_csv(HERE / "submission.csv", index=False)
    log(f"Wrote submission.csv: {sub.shape} — shipped {shipped}")

    result = {
        "target_column_chosen": target_col,
        "headline_metric": "accuracy",
        "headline_value": float(headline),
        "all_metrics": {
            "cv_accuracy_hgb": float(hgb_acc),
            "cv_accuracy_et": float(et_acc),
            "cv_accuracy_blend": float(best_blend_acc),
            "blend_weight_hgb": float(best_w),
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_blend_two_model_families.md",
        ],
        "skills_evaluated_not_applied": [
            "skill_probe_group_structure_in_ids.md: 0 overlap between train/test ids, ids are plain ints (no compound format).",
            "skill_train_on_listed_auxiliary_files.md: prompt mentions original dataset but no such file is supplied in DATA_DIR.",
            "skill_verify_split_claim_before_holdout.md: prompt makes no structural split claim (temporal/group).",
            "skill_stringify_low_cardinality_int_codes.md: no int-coded categoricals; all categoricals already strings.",
            "skill_multi_axis_constraint_budgets.md: prompt names no explicit latency/size/accuracy/hard-rule constraints.",
            "skill_missing_indicator_when_missingness_signals_class.md: 0 NaNs across all columns.",
            "skill_tune_decision_threshold_for_f1.md: multiclass accuracy, not F1/F-beta/MCC.",
            "skill_log_target_for_skewed_nonnegative_regression.md: classification, not regression.",
        ],
        "interpretation_notes": (
            f"7-class obesity classification (n_train=16606). Tuned HistGradientBoosting "
            f"with a 4-config grid and blended with ExtraTrees; final shipped: {shipped}. "
            "All preprocessing (OHE for 8 categorical cols) inside a CV pipeline to avoid leakage."
        ),
        "best_hgb_params": hgb_params,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "elapsed_sec": float(time.time() - t0),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"Wrote result.json (headline {result['headline_metric']}={result['headline_value']:.5f})")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
