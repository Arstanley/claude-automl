# v2 — Skill Transfer (Round 2)

Tests whether skills distilled from v1's 5 challenger audits help a fresh solver on **held-out tasks** with no per-task audit written for them.

## Setup

- **Skill library** (10 skills, distilled from 5 v1 audits) lives in `skills/`. Includes: `target_via_schema_diff`, `trust_test_csv_not_sample_submission`, `match_submission_format_to_metric`, `detect_group_leakage`, `categorical_prep_hygiene`, `model_missingness_as_signal`, `validation_matches_test_distribution`, `baselines_and_invariants`, `verify_spec_claims_against_data`, `tune_dont_default`.
- **2 held-out tasks** chosen with deliberately unusual eval shapes (no v1 task looks like these):
  - **playground-series-s3e5** — Wine Quality, ordinal target, **Quadratic Weighted Kappa** metric (ordinal-aware)
  - **playground-series-s3e13** — Vector Borne Disease, 11-class, **MPA@3** metric (space-separated top-3 predictions per row)
- **3 conditions per task**:
  - `A_no_skills` — vanilla solver, no audit, no skill library (baseline)
  - `A_with_skills` — vanilla solver + skill library prepended (test of transfer)
  - `B_audit_round2` — fresh per-task challenger audit + solver (ceiling)

## Headline numbers

| Task | Metric | Dir | A_no_skills | A_with_skills | B_audit_round2 |
|---|---|---|---:|---:|---:|
| s3e5 wine | QWK | ↑ | 0.4665 | **0.4816** | 0.4811 |
| s3e13 disease | hit@3 | ↑ | 0.5986 | 0.5775 | **0.6197** |

### The two deltas that matter

| Task | **A_with_skills − A_no_skills** (transfer signal) | **B_audit_round2 − A_with_skills** (ceiling gap) |
|---|---:|---:|
| s3e5 wine | **+0.0151** ✅ skills help | **−0.0006** ✅ skill library matches audit |
| s3e13 disease | **−0.0211** ❌ skills hurt | **+0.0423** ❌ audit clearly better |

**Mixed signal.** 1/2 tasks: skill library transfers and reaches per-task-audit ceiling. 1/2 tasks: skill library **regresses** vs vanilla and the per-task audit beats both.

## What happened on each task

### s3e5 — the success case
- A_no_skills: trained a `RandomForestRegressor` on continuous quality, rounded + clipped. CV-QWK 0.504, test-QWK 0.467.
- A_with_skills: applied `skill_match_submission_format_to_metric` → did regression + **post-hoc monotone threshold tuning** on OOF predictions to directly maximize QWK. CV-QWK 0.540, test-QWK 0.482.
- B_audit_round2: same regression + threshold-tuning recipe, with audit pushing toward GBR+RF+ET ensemble. CV-QWK 0.568, test-QWK 0.481.
- **The QWK-specific skill content (regression + threshold-tuning > argmax) came from the library, not from the per-task audit.** This is a clean win for transfer.

### s3e13 — the failure case
- A_no_skills: 3-model soft-vote (LR + RandomForest + BernoulliNB), simple, no tuning. test hit@3 = 0.599.
- A_with_skills: applied `skill_tune_dont_default` → grid-searched 10 configs with CV; picked **ExtraTrees(depth=8)** as best single. test hit@3 = 0.578. **Worse.**
- B_audit_round2: more conservative grid (LR + RF + voting), RF won. test hit@3 = 0.620. **Best.**
- **The "tune don't default" skill misfired on small-n data (565 train rows).** Aggressive single-model tuning beat a simple ensemble on CV but **overfit and lost on test**. A simple LR+RF+NB soft-vote (A_no_skills's strategy) generalizes better at this n.

## Interpretation

**1. Skills can transfer.** s3e5 is direct evidence: a skill written from a *titanic-accuracy / spaceship-accuracy / s3e22-F1* context successfully drove a *wine-QWK* solver toward the right modeling paradigm (regression + thresholds, not argmax). The skill articulated the *general* pattern ("match submission format to metric structure") concretely enough that it generalized to a metric type the v1 tasks never used.

**2. Skills can also misfire.** s3e13's failure is informative: `skill_tune_dont_default` was derived from cases where defaults *did* cost score (e.g., s3e3 attrition where tuned GBM beat default GBM). But applied unconditionally on n=565 with 11 classes, tuning becomes overfitting. The skill lacked *conditionality* — it said "tune" without saying "only when n is large enough relative to model complexity."

**3. The per-task audit (B) wins or ties on both tasks.** This was already the v1 story; v2 confirms it on the held-out tasks. The audit's value is that it's task-aware.

**4. The gap between "transfer" and "audit ceiling" varies by task type.** Where the skill matches the task's gaming surface (s3e5: format-to-metric), transfer ≈ ceiling. Where the skill misfires (s3e13: small-n tuning), transfer < no-skills. This suggests **skill activation needs to be conditional**, not blanket.

## Honest caveats

- **N=2 tasks.** Cannot generalize. Need 5+ held-out tasks to claim a pattern.
- **The skill library is a single revision.** No iteration over the skill set; no "remove the misfiring skill" loop.
- **Skills were dumped wholesale.** The A_with_skills solver got the *entire* library and decided which to apply. A retrieval-style protocol (only the relevant 2-3 skills shown) would be a different experiment — possibly with cleaner results because the solver wouldn't see misfiring skills.
- **No iteration of the challenger.** Skill quality is bounded by what v1's challengers happened to write. Co-evolution would refine skills based on which ones produced test-set gains.

## What this means for the paper

**Honest assessment**: This is a real but partial result. The "skills transfer across tasks" claim has evidence (s3e5), but it also has a counterexample (s3e13) that points at a missing piece: **conditional skill activation**.

The publishable story is *not* "skills transfer" (too clean, doesn't match data). It's:

> **"Adversarial-challenger-derived skills transfer for surface-level patterns (format-to-metric, schema-diff target, sample-submission traps) but require conditioning to avoid misfire on regimes (small-n, high-class-count) where the skill's underlying recommendation reverses sign. Per-task audits remain the ceiling because they are task-aware; the gap between accumulated skills and per-task audits is the open research question."**

That's a sharper, more honest framing than the round-1 results suggested. And the failure mode (skill_tune_dont_default on n=565) is a concrete enough mechanism to drive the next iteration: **skills should carry preconditions, not just actions**.

## Next experiments

In order of marginal value:

1. **Rewrite `skill_tune_dont_default` with a precondition** (e.g., "only when n_train > 1000 or n_classes ≤ 4") and re-run s3e13 A_with_skills. If transfer signal flips to positive, the failure was indeed missing conditionality — easy fix, large payoff.

2. **Add 3-5 more held-out tasks** to make the transfer-vs-misfire ratio statistically interpretable. Need diversity: more small-n tasks, more unusual metrics, some classical large-n tabular.

3. **Switch to retrieval**: rather than dumping all 10 skills, retrieve top-3 by similarity to the task spec. If the s3e13 failure is partly "solver got distracted by an irrelevant skill", retrieval should fix it. But this introduces a retriever-quality confound.

4. **Co-evolve the library**: after each task, the challenger updates / adds / refines skills based on what worked. v2 used a frozen library; the actual paper claim is that *the library evolves*. Run v3 with this loop.

## Files

- `/home/colligo/claude-automl/experiments/v2_skill_transfer/skills/` — 10 skill files + INDEX.md
- `/home/colligo/claude-automl/experiments/v2_skill_transfer/<task>/conditions/<cond>/` — per-condition artifacts
- `/home/colligo/claude-automl/experiments/v2_skill_transfer/<task>/results/eval.json` and `eval.md` — per-task scoring
