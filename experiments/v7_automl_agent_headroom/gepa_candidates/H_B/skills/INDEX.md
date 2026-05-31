# v4 Skill Library — Preconditioned Skills (Self-Improvement Round)

6 conditional skills distilled from v1 (5 DSBench tasks) and v3 (langid) failure cases. Every skill carries an explicit **PRECONDITION block** so a vanilla solver can mechanically check whether the skill applies — addressing v2's failure mode where unconditioned skills (e.g. `tune_dont_default`) misfired on small-n or constraint-dominated tasks.

Every skill below is backed by ≥1 concrete past failure with a measured gap.

## Skills

1. **[skill_probe_group_structure_in_ids](skill_probe_group_structure_in_ids.md)** — Precondition: ≥5% train/test value overlap on any non-target column, OR a parseable compound ID format. Action: GroupKFold + parse-the-substring groups. Evidence: spaceship-titanic A (+0.011pp optimistic CV).

2. **[skill_train_on_listed_auxiliary_files](skill_train_on_listed_auxiliary_files.md)** — Precondition: prompt names a non-train/test data file with phrases like "available for your use" and the file is consistent with a known target class. Action: concatenate into train. Evidence: langid A (91.98% abstain on Hi-Latn vs 98.7% correct for B/C).

3. **[skill_verify_split_claim_before_holdout](skill_verify_split_claim_before_holdout.md)** — Precondition: prompt makes a structural claim about train/test split (temporal/group) that is empirically checkable on the data. Action: probe before designing CV; trust data over prompt. Evidence: bike-sharing-demand A (believed "days 20+" lie, lost 0.008 RMSLE to C).

4. **[skill_stringify_low_cardinality_int_codes](skill_stringify_low_cardinality_int_codes.md)** — Precondition: int-dtype column with `nunique <= max(100, 0.05 * n_rows)` AND downstream pipeline uses scaler / linear / distance model. Action: cast to string + one-hot. Evidence: s3e22 A (`lesion_1` standardized as numeric, +0.024 micro-F1 gap).

5. **[skill_multi_axis_constraint_budgets](skill_multi_axis_constraint_budgets.md)** — Precondition: prompt names ≥2 constraint axes (latency/size/accuracy/hard-rule) and at least one has no explicit number. Action: commit to concrete numeric budgets per axis up front; do Pareto-aware model search. Evidence: langid A (3 unquantified axes, dominated point on the frontier).

6. **[skill_missing_indicator_when_missingness_signals_class](skill_missing_indicator_when_missingness_signals_class.md)** — Precondition: per-column NaN rate ≥ 5% AND silent imputer used. Action: add `*_missing` binary feature, NA-as-category for categoricals, convert sentinels (-1/9999) to NaN first. Evidence: s3e22 A (silent median imputation, +0.024 gap), spaceship-titanic A (silent most_frequent, +0.0098 gap).

7. **[skill_tune_when_n_train_supports_it](skill_tune_when_n_train_supports_it.md)** — Precondition: GBM/RF model family AND `n_train >= 500` AND prompt does not mention size/latency/production-budget tokens (the v2 misfire condition). Action: small explicit 3-5 config grid with the chosen CV. Evidence: s3e3 C_warned (sklearn defaults → 0.8153 vs tuned 0.840, −0.025 AUC). This is the v2 `skill_tune_dont_default` REWRITTEN with the precondition that would have suppressed it on the langid/small-n cases where it misfired.

8. **[skill_ordinal_target_and_rare_class_macro_f1](skill_ordinal_target_and_rare_class_macro_f1.md)** — Precondition: target is integer-like AND prompt names ordinal token (ordinal/quality/grade/rating/score/level) or class imbalance ratio ≥ 50, AND metric is macro-F1 / balanced-accuracy. Action: train both a regression-rounded model and a class-balanced LightGBM/XGBoost classifier; pick by val macro-F1; log per-class F1. Evidence: v7/wine_quality_white H_0 used sklearn RF/HGB nominal-classifier grid, all candidates val-F1 in 0.349–0.374 plateau; test F1 = 0.425 with class-3 and class-9 recall ≈ 0.

## How preconditions differ from v2

The v2 library had skills like `skill_tune_dont_default` with a soft "when this applies" block that was advisory, not mechanical. A vanilla solver reading "applies to non-trivial datasets" had to make a judgment call — and got it wrong on langid (size-constrained) and titanic-small-n.

v4 preconditions are written as **Python expressions over (prompt, train, test)** so the solver can compute them, not interpret them:

```python
# v4 skill applies iff:
n_train >= 500 and not re.search(r"\b(size|latency|MB|ms|production)\b", prompt, re.I)
```

This makes the skill library **executable as a decision procedure**, not a reading-comprehension exercise.
