# Skill: Trust test.csv as authoritative; sample_submission may be stale

## When this applies
Any task that ships a `sample_submission.csv` (or `gender_submission.csv`, `sampleSubmission.csv`, etc.) alongside `test.csv`. Especially when the dataset prompt looks "Kaggle-shaped."

## What to do
- Compute `len(sample_submission)` vs `len(test)` and the intersection of their ID columns at load time. If they disagree, **the test.csv IDs win** — sample_submission is likely copied from the original public competition and was never regenerated for this experimental split.
- Build the submission by taking IDs directly from `test.csv` (preserving order unless told otherwise), not by joining onto / copying / overwriting `sample_submission.csv`.
- Use `sample_submission` only as a *format example*: header name, column ordering, value dtype. Never as the row set or as predicted values.
- Explicitly assert in code that your submission's ID set equals the test.csv ID set, with the same row count. Log the comparison.
- Beware example rows in the prompt itself (e.g. `892,<value>` or `1677,...`) — those may also be stale IDs from the original full dataset.

## Why
A solver who builds `submission = sample_submission.copy(); submission[target] = pred` will produce a file with completely wrong IDs and zero overlap with the actual grading set, scoring zero. This was the most common honeytrap across v1 tasks.

## Code snippet
```python
test_ids = test["id"]  # or "PassengerId", "datetime", etc.
sample = pd.read_csv("data/sample_submission.csv")
sample_ids = sample.iloc[:, 0]

overlap = set(test_ids) & set(sample_ids)
print(f"test rows: {len(test_ids)}  sample rows: {len(sample_ids)}  ID overlap: {len(overlap)}")
if len(sample_ids) != len(test_ids) or len(overlap) < len(test_ids):
    print("sample_submission IDs do NOT match test.csv. Using test.csv as the source of truth.")

submission = pd.DataFrame({
    test_ids.name: test_ids.values,
    TARGET:        preds,            # in test.csv row order
})
assert len(submission) == len(test_ids)
assert set(submission[test_ids.name]) == set(test_ids)
submission.to_csv("submission.csv", index=False)
```

## Cross-task evidence
- Saw on: titanic (gender_submission IDs 892..1309, test IDs 6..890, zero overlap), spaceship-titanic (sample 4,277 rows vs test 1,739, zero overlap), bike-sharing-demand (sample 6,493 days-20+ rows vs test 2,178 days-1-19), playground-series-s3e3 (sample IDs disjoint from test, prompt mentions `1677/1678` not in test), playground-series-s3e22 (sample 824 rows vs test 247).
