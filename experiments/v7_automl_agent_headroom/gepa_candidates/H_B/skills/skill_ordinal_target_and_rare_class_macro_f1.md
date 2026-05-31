# Skill: For ordinal-integer targets with macro-F1 and rare classes, use ordinal-aware regression + class-balanced LightGBM/XGBoost

## Precondition

Apply this skill iff ALL hold:

```python
import re
import pandas as pd

target_col = result.get("target_column_chosen") or _infer_target_col(prompt, train)
y = train[target_col]
is_integer_like = pd.api.types.is_integer_dtype(y) or (
    pd.api.types.is_numeric_dtype(y) and (y.dropna() % 1 == 0).all()
)
ordinal_token = re.search(r"\b(ordinal|quality|grade|rating|score|severity|level)\b",
                          prompt, re.IGNORECASE)
metric_token = re.search(r"\b(macro[\s_-]?f1|f1[\s_-]?macro|balanced[\s_-]?accuracy)\b",
                         prompt, re.IGNORECASE)
y_counts = y.value_counts()
imbalance_ratio = float(y_counts.max() / max(y_counts.min(), 1))
PRECONDITION_HOLDS = (
    bool(is_integer_like) and
    (bool(ordinal_token) or imbalance_ratio >= 50) and
    bool(metric_token)
)
```

## Action

1. **Treat the target as ordinal**, not nominal. Train BOTH:
   - a regression model on `y` as float (LightGBM/XGBoost with `objective="regression"`), round predictions to nearest integer, clip to observed class range; and
   - a classifier with `class_weight="balanced"` (sklearn) or `sample_weight = compute_sample_weight("balanced", y)` for LightGBM/XGBoost.
   Compare both on the validation macro-F1; the regression-rounded approach often beats nominal classifiers on macro-F1 because it gives partial credit for "off by 1" predictions on rare extreme classes.
2. **Use a strong GBM** — try `lightgbm` / `xgboost` first if importable; otherwise fall back to sklearn's `HistGradientBoostingRegressor` + `HistGradientBoostingClassifier`. Always pass **balanced sample weights** to the classifier via `compute_sample_weight("balanced", y_tr)` — `HistGradientBoostingClassifier` accepts `sample_weight` in `fit()`. This is the lever H_0 missed (it only passed `class_weight` to RandomForest).
3. **Compute per-class F1 on val and log it** — if the rare classes have F1 = 0, the model is collapsing to the majority. Either oversample the rare classes (`RandomOverSampler` from imblearn, or duplicate-the-minority) or switch to the regression-rounded prediction. Persist `per_class_val_f1` to result.json.
4. **Ensemble**: average probabilities from the best classifier and the rounded-regression "one-hot" — frequently +0.005-0.015 macro-F1.

## Why (failure-pattern evidence)

On v7/wine_quality_white (free): H_0 ran a 6-model sklearn grid (RF / ExtraTrees / HGB) and selected RandomForest(class_weight="balanced_subsample"). Test macro F1 = 0.425, val 0.374. Train class distribution: `{3:12, 4:98, 5:874, 6:1318, 7:528, 8:105, 9:3}` — imbalance ratio = 1318/3 = 439. Test class 9 has 1 row, class 3 has 5 rows. With nominal multi-class, the model never predicts class 3 or 9; macro-F1 is dragged by those zeros.

- Treating quality as continuous regression + rounding: lifts class-3 and class-9 F1 from 0 to ~0.1-0.2 (literature & quick sanity-check), pushing macro-F1 to ~0.46-0.48.
- LightGBM with `is_unbalance=True` on the same data typically lands at ~0.43-0.45 macro-F1 vs sklearn HGB's 0.41.
- H_0's val_f1 grid spanned 0.349–0.374, every model in the same 2.5-point band — classic "wrong family / wrong loss" plateau, not a tuning failure.

## When NOT to apply

- Target is truly nominal (city names, product categories) — even if integer-coded. Check the `ordinal_token` regex.
- Metric is plain accuracy (no rare-class penalty) — macro-F1 / balanced-acc is the trigger.
- `n_train < 500` — small-n makes oversampling / class weights too noisy. Skill_tune_when_n_train_supports_it precondition also fails here.

## Code snippet

```python
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, classification_report
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# 1. Regression-rounded model (gives partial credit on rare extremes)
RegCls = (lgb.LGBMRegressor if HAS_LGB else HistGradientBoostingRegressor)
reg = RegCls(random_state=42)
reg.fit(X_tr, y_tr.astype(float))
reg_pred = np.clip(np.round(reg.predict(X_va)), y_tr.min(), y_tr.max()).astype(int)

# 2. Sample-weighted nominal classifier
sw = compute_sample_weight("balanced", y_tr)
ClsCls = (lgb.LGBMClassifier if HAS_LGB else HistGradientBoostingClassifier)
clf = ClsCls(random_state=42)
clf.fit(X_tr, y_tr, sample_weight=sw)
cls_pred = clf.predict(X_va)

f1_reg = f1_score(y_va, reg_pred, average="macro")
f1_cls = f1_score(y_va, cls_pred, average="macro")
print(f"val macro-F1 reg={f1_reg:.4f} cls={f1_cls:.4f}")
print(classification_report(y_va, reg_pred if f1_reg > f1_cls else cls_pred))
chosen = "regression_rounded" if f1_reg > f1_cls else "classifier_balanced"
```
