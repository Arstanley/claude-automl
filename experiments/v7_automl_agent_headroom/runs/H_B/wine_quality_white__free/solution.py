"""
Wine Quality White — multi-class classification (target=quality, integer 3..9).
Evaluation: macro F1.

Strategy (applies skill_ordinal_target_and_rare_class_macro_f1):
- Treat quality as ordinal: train BOTH a regression-rounded model and a class-balanced classifier.
- Class-balanced HistGradientBoostingClassifier via compute_sample_weight("balanced").
- Build a small ensemble (HGB-cls, HGB-reg-rounded, RF-balanced, ExtraTrees-balanced) — pick by val macro-F1.
- Also try probability-blend ensemble (avg cls probs with one-hotted regression rounding).
- Refit best pipeline on train+valid before predicting on test.
"""
from __future__ import annotations
import json, os, time, warnings, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

t0 = time.time()
DATA = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/wine_quality_white/data"
OUT  = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/runs/H_B/wine_quality_white__free"
TARGET = "quality"
N_JOBS = 2
SEED = 42

train = pd.read_csv(f"{DATA}/train.csv")
valid = pd.read_csv(f"{DATA}/valid.csv")
test  = pd.read_csv(f"{DATA}/test.csv")

feature_cols = [c for c in train.columns if c != TARGET]
X_tr, y_tr = train[feature_cols].values, train[TARGET].values.astype(int)
X_va, y_va = valid[feature_cols].values, valid[TARGET].values.astype(int)
X_te, y_te = test[feature_cols].values,  test[TARGET].values.astype(int)

ymin, ymax = int(y_tr.min()), int(y_tr.max())
all_classes = np.arange(ymin, ymax + 1)  # [3..9]
print(f"train shape {X_tr.shape}  valid {X_va.shape}  test {X_te.shape}")
print(f"class range {ymin}..{ymax}  train class counts: {np.bincount(y_tr)[ymin:ymax+1].tolist()}")

sw_tr = compute_sample_weight("balanced", y_tr)

# ---------- Candidate models ----------
candidates = {}

# 1) HistGradientBoostingClassifier, class-balanced via sample_weight
hgb_cls = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
    min_samples_leaf=20, l2_regularization=0.0,
    early_stopping=True, validation_fraction=0.15,
    random_state=SEED,
)
hgb_cls.fit(X_tr, y_tr, sample_weight=sw_tr)
candidates["hgb_cls_balanced"] = ("cls", hgb_cls)

# 2) HistGradientBoostingRegressor, rounded
hgb_reg = HistGradientBoostingRegressor(
    max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
    min_samples_leaf=20, l2_regularization=0.0,
    early_stopping=True, validation_fraction=0.15,
    random_state=SEED,
)
hgb_reg.fit(X_tr, y_tr.astype(float))
candidates["hgb_reg_rounded"] = ("reg", hgb_reg)

# 3) RandomForest, class_weight balanced_subsample (deep)
rf = RandomForestClassifier(
    n_estimators=600, max_features="sqrt",
    min_samples_leaf=1, class_weight="balanced_subsample",
    n_jobs=N_JOBS, random_state=SEED,
)
rf.fit(X_tr, y_tr)
candidates["rf_balanced"] = ("cls", rf)

# 4) ExtraTrees, class_weight balanced
et = ExtraTreesClassifier(
    n_estimators=800, max_features="sqrt",
    min_samples_leaf=1, class_weight="balanced",
    n_jobs=N_JOBS, random_state=SEED,
)
et.fit(X_tr, y_tr)
candidates["et_balanced"] = ("cls", et)

# 5) HGB classifier — larger trees variant
hgb_cls2 = HistGradientBoostingClassifier(
    max_iter=800, learning_rate=0.03, max_leaf_nodes=127,
    min_samples_leaf=10, early_stopping=True, validation_fraction=0.15,
    random_state=SEED + 1,
)
hgb_cls2.fit(X_tr, y_tr, sample_weight=sw_tr)
candidates["hgb_cls_balanced_deep"] = ("cls", hgb_cls2)

def predict_int(name, kind, model, X):
    if kind == "reg":
        p = model.predict(X)
        return np.clip(np.round(p), ymin, ymax).astype(int)
    return model.predict(X).astype(int)

# Evaluate each candidate on valid
val_results = {}
for name, (kind, model) in candidates.items():
    pred = predict_int(name, kind, model, X_va)
    f1 = f1_score(y_va, pred, average="macro", labels=all_classes, zero_division=0)
    val_results[name] = f1
    print(f"  val_macro_f1[{name}] = {f1:.4f}")

# ---------- Probability-blend ensemble ----------
# Get class-probability matrices for cls models on valid
def proba_matrix(model, X):
    """Return P over all_classes (3..9). Missing classes -> 0."""
    classes = model.classes_
    p = model.predict_proba(X)
    P = np.zeros((X.shape[0], len(all_classes)), dtype=float)
    for j, c in enumerate(classes):
        if c in all_classes:
            P[:, int(c) - ymin] = p[:, j]
    return P

def reg_to_proba(reg_model, X, sigma=0.7):
    """Convert regression preds to soft probs via Gaussian over class grid."""
    p = reg_model.predict(X)
    # gaussian over (ymin..ymax)
    grid = all_classes.astype(float).reshape(1, -1)
    P = np.exp(-((p.reshape(-1, 1) - grid) ** 2) / (2 * sigma * sigma))
    P /= P.sum(axis=1, keepdims=True) + 1e-12
    return P

P_va_hgb = proba_matrix(hgb_cls, X_va)
P_va_hgb2 = proba_matrix(hgb_cls2, X_va)
P_va_rf  = proba_matrix(rf, X_va)
P_va_et  = proba_matrix(et, X_va)
P_va_reg = reg_to_proba(hgb_reg, X_va, sigma=0.7)

ensembles = {
    "ens_cls_only":  (P_va_hgb + P_va_hgb2 + P_va_rf + P_va_et) / 4.0,
    "ens_cls_reg":   (P_va_hgb + P_va_hgb2 + P_va_rf + P_va_et + 2.0 * P_va_reg) / 6.0,
    "ens_hgb_reg":   (P_va_hgb + P_va_hgb2 + P_va_reg) / 3.0,
    "ens_hgb_reg_heavy": (0.5 * (P_va_hgb + P_va_hgb2) / 2.0 + 0.5 * P_va_reg),
}
for name, P in ensembles.items():
    pred = (P.argmax(axis=1) + ymin).astype(int)
    f1 = f1_score(y_va, pred, average="macro", labels=all_classes, zero_division=0)
    val_results[name] = f1
    print(f"  val_macro_f1[{name}] = {f1:.4f}")

best_name = max(val_results, key=val_results.get)
best_val = val_results[best_name]
print(f"BEST on valid: {best_name}  val_macro_f1={best_val:.4f}")

# ---------- Refit selected on train+valid, predict test ----------
X_all = np.vstack([X_tr, X_va])
y_all = np.concatenate([y_tr, y_va]).astype(int)
sw_all = compute_sample_weight("balanced", y_all)

def fit_cls(model_cls, kwargs, sw=None):
    m = model_cls(**kwargs)
    if sw is not None:
        try:
            m.fit(X_all, y_all, sample_weight=sw)
        except TypeError:
            m.fit(X_all, y_all)
    else:
        m.fit(X_all, y_all)
    return m

# Refit all base learners on all training data for the chosen strategy
if best_name in candidates:
    kind, _ = candidates[best_name]
    if best_name == "hgb_cls_balanced":
        final = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=20, early_stopping=True, validation_fraction=0.15,
            random_state=SEED,
        )
        final.fit(X_all, y_all, sample_weight=sw_all)
        test_pred = final.predict(X_te).astype(int)
    elif best_name == "hgb_cls_balanced_deep":
        final = HistGradientBoostingClassifier(
            max_iter=800, learning_rate=0.03, max_leaf_nodes=127,
            min_samples_leaf=10, early_stopping=True, validation_fraction=0.15,
            random_state=SEED + 1,
        )
        final.fit(X_all, y_all, sample_weight=sw_all)
        test_pred = final.predict(X_te).astype(int)
    elif best_name == "hgb_reg_rounded":
        final = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=20, early_stopping=True, validation_fraction=0.15,
            random_state=SEED,
        )
        final.fit(X_all, y_all.astype(float))
        test_pred = np.clip(np.round(final.predict(X_te)), ymin, ymax).astype(int)
    elif best_name == "rf_balanced":
        final = RandomForestClassifier(
            n_estimators=600, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=SEED,
        )
        final.fit(X_all, y_all)
        test_pred = final.predict(X_te).astype(int)
    elif best_name == "et_balanced":
        final = ExtraTreesClassifier(
            n_estimators=800, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced", n_jobs=N_JOBS, random_state=SEED,
        )
        final.fit(X_all, y_all)
        test_pred = final.predict(X_te).astype(int)
    else:
        raise RuntimeError("?")
else:
    # ensemble selected → refit all on train+valid then blend
    hgb_cls_f = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=20, early_stopping=True, validation_fraction=0.15,
        random_state=SEED,
    )
    hgb_cls_f.fit(X_all, y_all, sample_weight=sw_all)
    hgb_cls2_f = HistGradientBoostingClassifier(
        max_iter=800, learning_rate=0.03, max_leaf_nodes=127,
        min_samples_leaf=10, early_stopping=True, validation_fraction=0.15,
        random_state=SEED + 1,
    )
    hgb_cls2_f.fit(X_all, y_all, sample_weight=sw_all)
    rf_f = RandomForestClassifier(
        n_estimators=600, max_features="sqrt", min_samples_leaf=1,
        class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=SEED,
    )
    rf_f.fit(X_all, y_all)
    et_f = ExtraTreesClassifier(
        n_estimators=800, max_features="sqrt", min_samples_leaf=1,
        class_weight="balanced", n_jobs=N_JOBS, random_state=SEED,
    )
    et_f.fit(X_all, y_all)
    hgb_reg_f = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=20, early_stopping=True, validation_fraction=0.15,
        random_state=SEED,
    )
    hgb_reg_f.fit(X_all, y_all.astype(float))

    P_te_hgb  = proba_matrix(hgb_cls_f, X_te)
    P_te_hgb2 = proba_matrix(hgb_cls2_f, X_te)
    P_te_rf   = proba_matrix(rf_f, X_te)
    P_te_et   = proba_matrix(et_f, X_te)
    P_te_reg  = reg_to_proba(hgb_reg_f, X_te, sigma=0.7)

    if best_name == "ens_cls_only":
        P_te = (P_te_hgb + P_te_hgb2 + P_te_rf + P_te_et) / 4.0
    elif best_name == "ens_cls_reg":
        P_te = (P_te_hgb + P_te_hgb2 + P_te_rf + P_te_et + 2.0 * P_te_reg) / 6.0
    elif best_name == "ens_hgb_reg":
        P_te = (P_te_hgb + P_te_hgb2 + P_te_reg) / 3.0
    else:  # ens_hgb_reg_heavy
        P_te = 0.5 * (P_te_hgb + P_te_hgb2) / 2.0 + 0.5 * P_te_reg
    test_pred = (P_te.argmax(axis=1) + ymin).astype(int)

# ---------- Scoring on test (labels are available in test.csv per task) ----------
test_macro_f1 = f1_score(y_te, test_pred, average="macro", labels=all_classes, zero_division=0)
per_class_test = f1_score(y_te, test_pred, average=None, labels=all_classes, zero_division=0)
per_class_val_best = None
# recompute per-class val for the chosen
if best_name in candidates:
    kind, mdl = candidates[best_name]
    val_pred_best = predict_int(best_name, kind, mdl, X_va)
else:
    P = ensembles[best_name]
    val_pred_best = (P.argmax(axis=1) + ymin).astype(int)
per_class_val_best = f1_score(y_va, val_pred_best, average=None, labels=all_classes, zero_division=0)
print("VAL per-class F1:", dict(zip(all_classes.tolist(), [round(v,4) for v in per_class_val_best.tolist()])))
print("TEST macro F1:", round(test_macro_f1, 4))
print("TEST per-class F1:", dict(zip(all_classes.tolist(), [round(v,4) for v in per_class_test.tolist()])))
print(classification_report(y_te, test_pred, labels=all_classes, zero_division=0))

# ---------- Write submission ----------
sub = pd.DataFrame({TARGET: test_pred})
sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "macro_f1",
    "headline_value": float(test_macro_f1),
    "all_metrics": {
        "val_macro_f1_by_model": {k: float(v) for k, v in val_results.items()},
        "best_model": best_name,
        "best_val_macro_f1": float(best_val),
        "test_macro_f1": float(test_macro_f1),
        "test_per_class_f1": {int(c): float(v) for c, v in zip(all_classes, per_class_test)},
        "val_per_class_f1": {int(c): float(v) for c, v in zip(all_classes, per_class_val_best)},
    },
    "skills_applied": [
        "skill_ordinal_target_and_rare_class_macro_f1.md",
        "skill_tune_when_n_train_supports_it.md",
    ],
    "interpretation_notes": (
        "Treated quality as ordinal: trained HGB-regressor (rounded) alongside class-balanced HGB/RF/ExtraTrees "
        "classifiers (sample_weight='balanced'), then selected by val macro-F1 across single models AND "
        "probability-blend ensembles (cls + Gaussian-softened regression). lightgbm/xgboost unavailable so "
        "fell back to sklearn. Refit the chosen pipeline on train+valid before scoring test."
    ),
    "runtime_seconds": round(time.time() - t0, 2),
}
with open(os.path.join(OUT, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"Done in {result['runtime_seconds']}s.  best={best_name}  test_macro_f1={test_macro_f1:.4f}")
