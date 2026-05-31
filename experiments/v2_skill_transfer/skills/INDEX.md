# Skill library — v2 transfer set

10 task-general ML skills distilled from 5 v1 DSBench tasks (titanic, spaceship-titanic, bike-sharing-demand, playground-series-s3e3, playground-series-s3e22). Each skill matched ≥2 v1 tasks and addresses a recurring gaming surface or failure mode that an adversarial challenger flagged. The skills are framed in problem-shape language ("when test/train were re-split", "when codes are stored as int") so they transfer to tasks with different metrics, domains, and target types.

## Skills

1. **[skill_target_via_schema_diff](skill_target_via_schema_diff.md)** — Use `set(train.columns) - set(test.columns)` to identify the target; don't trust prose, folklore, or the sample-submission header alone.

2. **[skill_trust_test_csv_not_sample_submission](skill_trust_test_csv_not_sample_submission.md)** — Sample-submission files are often stale (copied from the original public competition). Build the submission from `test.csv` IDs; use sample-sub only as a format example.

3. **[skill_match_submission_format_to_metric](skill_match_submission_format_to_metric.md)** — Pick prediction form (probabilities vs hard labels, clipped float vs int, bool vs string) from the metric definition. Mismatched format silently sandbags scores.

4. **[skill_detect_group_leakage](skill_detect_group_leakage.md)** — Probe for ID/group/time overlap between train and test; use `GroupKFold` (or time-aware splits) when overlap is material. Random CV on group-leaked data inflates headlines by the leakage rate.

5. **[skill_categorical_prep_hygiene](skill_categorical_prep_hygiene.md)** — Stringify integer code columns, decompose compound strings (`deck/num/side`), drop constants & ID columns from features, and collapse rare categorical levels into `__rare__`.

6. **[skill_model_missingness_as_signal](skill_model_missingness_as_signal.md)** — Add missing-indicator features instead of silent imputation; treat categorical NaN as its own level; watch for encoded missingness (sensor-floor zeros, "-1" placeholders).

7. **[skill_validation_matches_test_distribution](skill_validation_matches_test_distribution.md)** — Empirically diagnose the train/test split before picking CV. Don't over-defensively pick the strictest possible split — a pessimistic validation can push you toward a worse model.

8. **[skill_baselines_and_invariants](skill_baselines_and_invariants.md)** — Report majority-class / marginal-mean / single-feature baselines so the model's complexity is justified. Exploit deterministic invariants (sum-identities, "if A then B" rules) for imputation and decomposition.

9. **[skill_verify_spec_claims_against_data](skill_verify_spec_claims_against_data.md)** — At load time, verify every factual claim in the prompt (row counts, column names, ID ranges, split logic) against the actual data. When they disagree, the data wins.

10. **[skill_tune_dont_default](skill_tune_dont_default.md)** — Don't ship sklearn defaults on the final model. Tune at least `n_estimators`, `learning_rate`, `max_depth`, and class balance using the CV protocol you committed to. Defaults routinely lose 1-3pp.

## How to use this library

When starting a new task:
1. **Ground truth first** — run #9 (verify spec) + #1 (schema diff) + #7 (diagnose split) before any modeling.
2. **Submission template early** — apply #2 (trust test.csv) + #3 (format to metric) immediately, before training, so the end-to-end pipeline is exercised on a stub prediction.
3. **Feature engineering** — apply #5 (categorical prep) + #6 (missingness) + #8 (invariants).
4. **Validation** — apply #4 (group-aware where needed) + #7 (CV matches test distribution).
5. **Modeling** — apply #10 (tune, don't default) + #8 (baselines as sanity floor).
