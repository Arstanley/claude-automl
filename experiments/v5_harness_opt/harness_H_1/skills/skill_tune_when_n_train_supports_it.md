# Skill: Tune GBM/RF hyperparameters when n_train is large enough to support it

## Precondition
Apply this skill if ALL of the following hold (this is the **with-precondition** version that v2's `skill_tune_dont_default` lacked):

1. The chosen primary model family is GBM-like (`GradientBoostingClassifier/Regressor`, `HistGradientBoosting*`, `XGBoost`, `LightGBM`, `RandomForest*`), AND
2. `n_train >= 500` (smaller datasets are dominated by CV variance and hyperparameter selection adds more noise than signal), AND
3. The task is NOT a "constraint-dominated" one — i.e. the prompt does NOT cap `model_size_mb` AND does NOT cap p99 latency at sub-100ms. (Sklearn's `GradientBoostingClassifier(n_estimators=100, max_depth=3, lr=0.1)` is much smaller and faster than `n_estimators=500, max_depth=5` — when size/latency are tight, the defaults may be the Pareto-correct choice.) Check by searching the prompt for tokens like `size`, `latency`, `MB`, `ms`, `production`, `mobile` — if matched, this skill does NOT apply automatically; verify the size/speed of any tuned config explicitly.

If precondition 2 fails (`n_train < 500`): use sklearn defaults but lock in a sensible config (`n_estimators=200`, `max_depth=3`, `learning_rate=0.05`) — too small to tune.
If precondition 3 fails: see `skill_multi_axis_constraint_budgets`.

## Action
- Try a SMALL EXPLICIT grid of 3-5 configs (not a 100-config random search; you don't have the budget):
  ```
  [
    (n_estimators=300, lr=0.05, max_depth=3, subsample=0.9),
    (n_estimators=500, lr=0.03, max_depth=4, subsample=0.9),
    (n_estimators=300, lr=0.05, max_depth=5, subsample=0.8),
  ]
  ```
- Evaluate each with the SAME CV protocol you picked (see `skill_probe_group_structure_in_ids`). Report mean ± std per config; pick by mean.
- For imbalanced binary classification, add `class_weight="balanced"` (RF) or `scale_pos_weight=neg/pos` (XGB/LGB) as one of the grid axes.
- Refit the chosen config on full train. Persist `result.json.audit.hyperparameter_grid: [...], chosen_config: {...}`.

## Why (failure pattern evidence)
On v1/playground-series-s3e3 (Employee Attrition, n_train=1341, ROC AUC), C_warned used sklearn's `GradientBoostingClassifier()` with all defaults: `n_estimators=100, max_depth=3, learning_rate=0.1`. A_vanilla and B_audit both used the tuned `(n_estimators=300, lr=0.05, max_depth=3, subsample=0.9)` config. C's true AUC: 0.8153. A's: 0.8319. B's: 0.8395. The 0.024 gap between C and B is essentially the tuning gap — same model family, same preprocessing, different `n_estimators` and `learning_rate`. C lost by 2pp because of this single decision.

The v2 skill `skill_tune_dont_default` (no precondition) would also have fired on `titanic` (n=712) where the gain is negligible and on `langid` (n=70k but multi-axis constrained) where tuning a bigger GBM would push past the size budget. The precondition `n_train >= 500 AND not constraint-dominated` correctly fires on s3e3, s3e22 (n=988, just over threshold), spaceship-titanic (n=8693), bike-sharing-demand (n=10886) and correctly does NOT fire on titanic (only n=712 but historically tuning helps marginally — borderline, applies) or langid (size-constrained).

## Code snippet
```python
import re
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def should_tune(n_train: int, prompt: str) -> bool:
    if n_train < 500:
        return False
    constraint_tokens = re.findall(r"\b(size|latency|MB|ms|production|mobile|p99)\b", prompt, re.IGNORECASE)
    if constraint_tokens:
        # constraint-dominated; let the multi-axis skill handle config selection
        return False
    return True

if should_tune(len(train), prompt_text):
    grid = [
        dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9),
        dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.9),
        dict(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8),
    ]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    results = []
    for params in grid:
        scores = cross_val_score(
            GradientBoostingClassifier(**params, random_state=0),
            X, y, cv=cv, scoring=metric_name, n_jobs=1,
        )
        results.append((scores.mean(), scores.std(), params))
        print(f"  {params}: {scores.mean():.4f} ± {scores.std():.4f}")
    results.sort(reverse=True)
    best_params = results[0][2]
    final_model = GradientBoostingClassifier(**best_params, random_state=0).fit(X, y)
else:
    # Below threshold or constraint-dominated: lock in a sensible single config
    final_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=0,
    ).fit(X, y)
```

## Cross-task evidence
- Past failure: v1/playground-series-s3e3 C_warned — used sklearn `GradientBoostingClassifier()` defaults, AUC 0.8153; tuned configs (A/B) scored 0.832/0.840. n_train=1341 well above the 500 threshold; no constraint mentions. Clean precondition match.
- Past success-by-skip: v3/langid — A used a tuned `LogisticRegression(C=4)` with HashingVectorizer(2^18) — explicitly *picked* the small/fast config for the multi-axis constraints. Tuning to `n_estimators=500` on a deep GBM would blow the size budget. The constraint-token check correctly suppresses the skill here.
