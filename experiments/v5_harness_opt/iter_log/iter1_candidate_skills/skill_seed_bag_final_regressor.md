# Skill: Seed-bag the final regressor for variance reduction

## Precondition
Apply if ALL hold (mechanically checkable):
1. Task is regression — `pd.api.types.is_numeric_dtype(train[target])` AND `train[target].nunique() > 20`.
2. `n_train >= 500`.
3. Chosen model family takes a `random_state` argument (HGB, GBR, RF, ExtraTrees, sklearn linear with `sag/saga` solver — true for the default GBM/HGB families this harness uses).
4. The prompt does NOT contain latency/size tokens (re-uses the `skill_tune_when_n_train_supports_it` token list: `size|latency|MB|ms|production|mobile|p99`).

```python
import re, pandas as pd
preco = (
    pd.api.types.is_numeric_dtype(train[target]) and train[target].nunique() > 20
    and len(train) >= 500
    and not re.search(r"\b(size|latency|MB|ms|production|mobile|p99)\b", prompt, re.I)
)
```

## Action
After picking the chosen config via CV, refit on full train with **3 different `random_state` seeds** (e.g. 0, 1, 2) and average their test-set predictions. Persist the per-seed predictions count in `result.json.audit.seed_bag = [0,1,2]`.

This is pure variance reduction — no hyperparameter is tuned against OOF. **Anti-overfit guard**: do NOT choose the bag size or seed set by inspecting test predictions or by maximizing OOF; use the fixed `[0, 1, 2]` set. If you want to verify the bag helps, compute OOF score for seed 0 alone vs the 3-seed average on a separate **held-out fold not used for hyperparameter selection** — report both numbers and only ship the bag if average ≤ seed-0 OOF (lower is better for RMSE/RMSLE/MAE).

```python
import numpy as np
seeds = [0, 1, 2]
test_preds = []
for s in seeds:
    m = ModelCls(**best_params, random_state=s).fit(X_full, y_full)
    test_preds.append(m.predict(X_test))
final_pred = np.mean(test_preds, axis=0)
```

## Why
H_0 baseline on the 20-task pool: only `playground-series-s3e9` (the lone task with explicit seed-bagging) used this trick. RMSE-style regression tasks (s3e6 = 170293, s3e8 = 579.7, s3e16 = 1.366 MAE, bike RMSLE 0.260, s3e11 RMSLE 0.296) all used a single-seed final fit. A free 3-seed bag is the cheapest known variance reduction for tree GBMs and never hurts in expectation when bagging-compatible (independent random_state). Expected gain is small (0.1–0.5% relative) but free, and the cost is 3x final-fit time only (no extra CV).

## Cross-task evidence
- v5/H_0/playground-series-s3e9 already did seed-bagging (5 seeds) and reported a clean RMSE of 12.82; the 4 other regression tasks in the train pool (s3e6, s3e8, s3e11, s3e16, bike) did not. This skill makes that practice systematic.
