# Skill: Match submission format (header, dtype, value form) to the metric

## When this applies
Before writing `submission.csv`. The metric definition determines whether predictions must be hard labels, probabilities, integers, floats, strings, or booleans — and the header must match what the grader expects.

## What to do
- Read the metric spec carefully and pick the prediction form it requires:
  - **AUC / log-loss / ranking metrics**: emit `predict_proba(...)[:, 1]` (or class probabilities), NOT `predict()`. Hard-labels cap AUC well below the ranking ceiling.
  - **Accuracy / F1 / hard-classification**: emit the literal class label in the same dtype as `train[TARGET]` (int `0/1`, bool `True/False`, or string `"lived"/"died"`).
  - **RMSE / RMSLE / regression**: emit non-negative reals where the metric requires it (RMSLE needs `pred >= 0`; clip before writing). Don't round to int unless the metric or train target is integer-valued AND rounding was chosen on validation.
- Set the header to exactly what `sample_submission.csv` uses (treating sample_sub as a *format* example, not a row set). Match case, spacing, and column order.
- Match the prediction column dtype to `train[TARGET].dtype` (so bool stays bool, not 0/1; strings stay strings, not int-encoded).
- Verify with assertions before writing: no NaN, no inf, no negative when forbidden, value range is plausible for the metric.

## Why
A submission with the right model but the wrong format scores zero or far below ceiling. Hard labels under AUC silently sandbag. Float `0.0/1.0` under exact-match accuracy can grade as zero. Booleans written as `True`/`False` strings vs python bools change downstream parsing.

## Code snippet
```python
# Pick output form from metric
if METRIC in {"auc", "log_loss", "roc_auc"}:
    preds = model.predict_proba(X_test)[:, 1]
    assert preds.min() >= 0 and preds.max() <= 1
elif METRIC in {"rmsle"}:
    preds = np.clip(model.predict(X_test), 0, None)
elif METRIC in {"accuracy", "f1_micro", "f1_macro"}:
    preds = model.predict(X_test)
    # cast to train target dtype
    preds = pd.Series(preds).astype(train[TARGET].dtype).values

# Match sample header / dtype literally
sample = pd.read_csv("data/sample_submission.csv", nrows=1)
header = list(sample.columns)        # e.g. ['PassengerId','Survived']
sub = pd.DataFrame({header[0]: test[header[0]].values, header[1]: preds})
assert not sub.isna().any().any()
sub.to_csv("submission.csv", index=False)
```

## Cross-task evidence
- Saw on: titanic (int 0/1 vs float/bool/string risk), spaceship-titanic (bool True/False matching train dtype), bike-sharing-demand (non-negative floats for RMSLE; rounding optional), playground-series-s3e3 (probabilities for AUC, NOT hard labels — biggest sandbag risk), playground-series-s3e22 (string labels `lived/died/euthanized` for micro-F1).
