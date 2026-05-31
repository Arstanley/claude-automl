# Skill: Tune the decision threshold on OOF probabilities when the metric is F1 / F-beta / MCC

## Precondition
Apply this skill if ALL of the following hold:

1. The task is binary classification (target has exactly 2 classes), AND
2. The headline metric is one of: `f1`, `f1_binary`, `f_beta`, `fbeta`, `mcc`, `matthews`, `balanced_accuracy` (any threshold-sensitive metric — NOT `roc_auc`, NOT `log_loss`, NOT `accuracy` if classes are balanced), AND
3. The chosen model has a `predict_proba` (or `decision_function`) method (sklearn classifiers do), AND
4. `n_train >= 500` (need enough rows for a stable threshold pick).

If the metric is `roc_auc` or `log_loss`, this skill does NOT apply — those are threshold-independent / proper-scoring-rule metrics and threshold tuning is irrelevant. If the metric is plain `accuracy` AND classes are balanced (40-60%), the optimal threshold is essentially 0.5 anyway; skill is optional.

## Action
- Generate out-of-fold (OOF) probability predictions using the same CV split you used for hyperparameter selection.
- Sweep thresholds in a small grid (e.g., `np.linspace(0.20, 0.80, 31)` — 31 candidates is plenty) and pick the threshold that maximizes the headline metric on OOF.
- Apply that threshold to test-set probabilities to produce the binary submission.
- Persist `result.json.audit.chosen_threshold`, `chosen_threshold_metric_value`, and the threshold grid.
- Sanity check: print `(threshold, oof_f1)` for a few values around the optimum so the choice is auditable.

## Why (failure pattern evidence)
On H_0/nlp-getting-started (F1 metric, n_train=6090, ~43% positives), the solver fit `LogisticRegression`, ran a C-grid, then called `.predict()` — which uses the implicit 0.5 cutoff. OOF F1 was 0.7592 with precision 0.81 / recall 0.71. The precision-recall imbalance is the tell: when precision >> recall on a balanced metric like F1, the threshold is too high — lowering it would trade some precision for more recall and push F1 up. Threshold tuning on the same OOF probs (no extra model training) typically recovers +0.01-0.03 F1 on tasks like this. This is mechanical, free, and works for any binary F1/MCC/F-beta task.

The same pattern applies to imbalanced binary classification with F1 (rare-positive medical/fraud tasks) where the optimal threshold is often well below 0.5 (e.g. 0.20-0.35).

## Code snippet
```python
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def tune_threshold(oof_proba: np.ndarray, y: np.ndarray, metric=f1_score,
                   grid=np.linspace(0.05, 0.95, 91)) -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = metric(y, (oof_proba >= t).astype(int))
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s

# After hyperparam selection, generate OOF probabilities with the chosen config
cv = StratifiedKFold(5, shuffle=True, random_state=0)
oof_proba = np.zeros(len(y))
for tr_idx, va_idx in cv.split(X, y):
    m = make_model(**best_params).fit(X[tr_idx], y[tr_idx])
    oof_proba[va_idx] = m.predict_proba(X[va_idx])[:, 1]

thr, oof_f1_at_thr = tune_threshold(oof_proba, y)
print(f"chosen threshold: {thr:.3f}  OOF F1: {oof_f1_at_thr:.4f}  (vs 0.5 default)")

# Refit on full data, predict probs on test, apply tuned threshold
final = make_model(**best_params).fit(X, y)
test_proba = final.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= thr).astype(int)
```

## Cross-task evidence
- H_0/nlp-getting-started: predict() at 0.5 yielded OOF F1=0.759 with P=0.81 / R=0.71 — high precision, low recall, classic sign that the threshold is too high for F1. Free fix.
- Generalizes to any binary task with `f1`, `mcc`, or `f_beta` as the headline.
