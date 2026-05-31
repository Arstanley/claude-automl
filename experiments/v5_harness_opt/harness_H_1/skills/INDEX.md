# v5 Skill Library — Preconditioned Skills (Harness Optimization Round)

10 conditional skills. The 7 v4 skills are kept; v5 adds 3 modeling skills distilled from H_0 outcomes on 10 train tasks where the solver consistently shipped single-GBM models without threshold tuning or target transforms. Every skill carries an explicit **PRECONDITION block** so a vanilla solver can mechanically check whether the skill applies — addressing v2's failure mode where unconditioned skills (e.g. `tune_dont_default`) misfired on small-n or constraint-dominated tasks.

Every skill below is backed by ≥1 concrete past failure with a measured gap (or, for v5 additions, by a consistent H_0 pattern where the gap is recoverable mechanically).

## Skills

1. **[skill_probe_group_structure_in_ids](skill_probe_group_structure_in_ids.md)** — Precondition: ≥5% train/test value overlap on any non-target column, OR a parseable compound ID format. Action: GroupKFold + parse-the-substring groups. Evidence: spaceship-titanic A (+0.011pp optimistic CV).

2. **[skill_train_on_listed_auxiliary_files](skill_train_on_listed_auxiliary_files.md)** — Precondition: prompt names a non-train/test data file with phrases like "available for your use" and the file is consistent with a known target class. Action: concatenate into train. Evidence: langid A (91.98% abstain on Hi-Latn vs 98.7% correct for B/C).

3. **[skill_verify_split_claim_before_holdout](skill_verify_split_claim_before_holdout.md)** — Precondition: prompt makes a structural claim about train/test split (temporal/group) that is empirically checkable on the data. Action: probe before designing CV; trust data over prompt. Evidence: bike-sharing-demand A (believed "days 20+" lie, lost 0.008 RMSLE to C).

4. **[skill_stringify_low_cardinality_int_codes](skill_stringify_low_cardinality_int_codes.md)** — Precondition: int-dtype column with `nunique <= max(100, 0.05 * n_rows)` AND downstream pipeline uses scaler / linear / distance model. Action: cast to string + one-hot. Evidence: s3e22 A (`lesion_1` standardized as numeric, +0.024 micro-F1 gap).

5. **[skill_multi_axis_constraint_budgets](skill_multi_axis_constraint_budgets.md)** — Precondition: prompt names ≥2 constraint axes (latency/size/accuracy/hard-rule) and at least one has no explicit number. Action: commit to concrete numeric budgets per axis up front; do Pareto-aware model search. Evidence: langid A (3 unquantified axes, dominated point on the frontier).

6. **[skill_missing_indicator_when_missingness_signals_class](skill_missing_indicator_when_missingness_signals_class.md)** — Precondition: per-column NaN rate ≥ 5% AND silent imputer used. Action: add `*_missing` binary feature, NA-as-category for categoricals, convert sentinels (-1/9999) to NaN first. Evidence: s3e22 A (silent median imputation, +0.024 gap), spaceship-titanic A (silent most_frequent, +0.0098 gap).

7. **[skill_tune_when_n_train_supports_it](skill_tune_when_n_train_supports_it.md)** — Precondition: GBM/RF model family AND `n_train >= 500` AND prompt does not mention size/latency/production-budget tokens (the v2 misfire condition). Action: small explicit 3-5 config grid with the chosen CV. Evidence: s3e3 C_warned (sklearn defaults → 0.8153 vs tuned 0.840, −0.025 AUC). This is the v2 `skill_tune_dont_default` REWRITTEN with the precondition that would have suppressed it on the langid/small-n cases where it misfired.

8. **[skill_tune_decision_threshold_for_f1](skill_tune_decision_threshold_for_f1.md)** — Precondition: binary classification AND metric is F1/F-beta/MCC/balanced-accuracy AND `n_train >= 500`. Action: OOF probabilities → sweep thresholds → pick metric-maximizing threshold (NOT 0.5). Evidence: H_0/nlp-getting-started shipped predict() at 0.5, OOF P=0.81 / R=0.71 — recoverable +0.01-0.03 F1.

9. **[skill_log_target_for_skewed_nonnegative_regression](skill_log_target_for_skewed_nonnegative_regression.md)** — Precondition: regression AND `(y >= 0).all()` AND (metric is rmsle/msle OR `skew(y) >= 1.0` OR `max(y)/median(y) >= 20`). Action: fit on `log1p(y)`, predict `expm1` clipped to ≥0; A/B vs raw on CV. Evidence: H_0/bike-sharing-demand DID this and won (RMSLE); H_0/s3e6 (price) and s3e9 (strength) did not and shipped raw-y models on right-skewed targets.

10. **[skill_blend_two_model_families](skill_blend_two_model_families.md)** — Precondition: tabular (NOT pure-text) AND `n_train >= 1000` AND primary model is a GBM AND no latency/size constraint tokens. Action: fit a second family (RF / ExtraTrees / Ridge), blend OOF, pick weight by CV; ship single model if blend weight ∈ {0, 1}. Evidence: every H_0 tabular task shipped a single GBM — typical recoverable gain 0.2-0.8% on classification, 1-3% on regression.

## How preconditions differ from v2

The v2 library had skills like `skill_tune_dont_default` with a soft "when this applies" block that was advisory, not mechanical. A vanilla solver reading "applies to non-trivial datasets" had to make a judgment call — and got it wrong on langid (size-constrained) and titanic-small-n.

v4 preconditions are written as **Python expressions over (prompt, train, test)** so the solver can compute them, not interpret them:

```python
# v4 skill applies iff:
n_train >= 500 and not re.search(r"\b(size|latency|MB|ms|production)\b", prompt, re.I)
```

This makes the skill library **executable as a decision procedure**, not a reading-comprehension exercise.
