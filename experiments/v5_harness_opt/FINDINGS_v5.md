# v5 — Harness Optimization (SkillOpt-style)

First attempt at the core paper contribution: **optimize the AutoML harness itself** — not the model, not the data — using outcomes from a training task pool to propose bounded edits to the harness (solver prompt + skill library), validated on a held-out val set, then tested on a held-out test set.

## Setup

- **Initial harness H_0**: `harness/solver_prompt.md` (template) + 7 conditional skills distilled from v1+v3 (the v4 library).
- **Task pool**: 20 DSBench-tabular tasks, stratified by metric type (binary acc/AUC, multiclass, regression RMSLE/RMSE/MAE/QWK, multi-output, top-3-classification).
- **Splits**: 10 train / 5 val / 5 test (frozen at the start).
- **Optimizer**: SkillOpt-style — reads (current harness, train outcomes), proposes a patch bundle (≤5 bounded edits across solver_prompt.md + skills/*.md), patch is applied to H_1.
- **Acceptance gate**: H_1 evaluated on val set; accept only if val mean doesn't drop materially.

## The patch the Optimizer proposed

Reading the 10 H_0 training-task outcomes, the Optimizer identified three patterns it thought were missing:

1. **`skill_tune_decision_threshold_for_f1`** (new) — motivated by nlp-getting-started OOF precision/recall imbalance (P=0.81, R=0.71) signaling a too-high default 0.5 threshold for F1.
2. **`skill_log_target_for_skewed_nonnegative_regression`** (new) — motivated by s3e6 (price RMSE) and s3e9 (strength RMSE) shipping raw-y models on right-skewed targets; bike-sharing only used log1p because the solver hand-detected RMSLE.
3. **`skill_blend_two_model_families`** (new) — every H_0 tabular task shipped a single GBM; only s3e9 ensembled (same-family seeds).
4. **`solver_prompt.md` edit** — mandatory mechanical iteration over every INDEX skill (many H_0 result.jsons listed only 1-3 skills_applied, suggesting partial library scans).
5. **`INDEX.md` rewrite** to incorporate the three new skills.

5 edits within the SkillOpt budget. Every new skill carries an explicit precondition (Python expression over `(prompt, train.csv, test.csv)`).

## Results: H_0 vs H_1 (DSBench native scores)

### Val (5 tasks — gate)

| task | metric | dir | H_0 | H_1 | Δ | rel% | verdict |
|---|---|---|---:|---:|---:|---:|---|
| playground-series-s3e7 | AUC | ↑ | 0.8941 | 0.8967 | +0.0025 | +0.28% | **WIN** |
| playground-series-s4e3 | mean-AUC | ↑ | 0.8881 | 0.8850 | −0.0031 | −0.35% | lose |
| playground-series-s3e14 | MAE | ↓ | 338.38 | 338.45 | −0.07 | −0.02% | lose |
| playground-series-s3e5 | QWK | ↑ | 0.4286 | 0.4653 | +0.0367 | **+8.56%** | **WIN** |
| playground-series-s3e13 | MPA@3 | ↑ | 0.5986 | 0.5915 | −0.0070 | −1.18% | lose |

**Val: 2 wins / 3 losses, mean rel = +1.46%, median rel = −0.02%**

### Test (5 tasks — held out, never seen by Optimizer)

| task | metric | dir | H_0 | H_1 | Δ | rel% | verdict |
|---|---|---|---:|---:|---:|---:|---|
| playground-series-s3e25 | MedAE | ↓ | 0.5274 | 0.5120 | +0.0154 | **+2.93%** | **WIN** |
| playground-series-s3e1 | RMSE | ↓ | 0.5959 | 0.5907 | +0.0052 | +0.87% | **WIN** |
| tabular-playground-series-jul-2021 | RMSLE | ↓ | 0.1246 | 0.1280 | −0.0034 | −2.70% | lose |
| playground-series-s3e18 | mean-AUC | ↑ | 0.6465 | 0.6361 | −0.0104 | −1.61% | lose |
| playground-series-s4e4 | RMSLE | ↓ | 0.1512 | 0.1516 | −0.0005 | −0.32% | lose |

**Test: 2 wins / 3 losses, mean rel = −0.17%, median rel = −0.32%**

### Overall (val + test, 10 tasks)

- **Win rate**: 4/10 = 40%
- **Mean relative improvement**: +0.65%
- **Median relative improvement**: −0.02%
- **Sum of relative improvements**: +6.46% (dominated by the s3e5 QWK outlier; without it, the picture flips negative)

## Honest interpretation

**This is a borderline result.** The new skills the Optimizer added are individually well-motivated — log-target for skewed regression and family-diverse blending are textbook gains — but they don't reliably help on held-out tasks. Specifically:

- **Where H_1 wins**: tasks where one of the new skills had a *clear precondition match and a clear modeling correction*. s3e5 QWK gained +8.6% because regression+threshold-tuning + blending is exactly the recipe that wins this competition class. s3e25 MedAE won because log-target + blend is a textbook gain on a medium-skew target.

- **Where H_1 loses (small margins)**: heterogeneous tasks where the blend was attempted but didn't help, or where the Optimizer's mandatory-iteration directive made the solver bias toward "applying skills" even when the underlying model would have been better without.

- **Where H_1 ties or barely moves**: most tasks (median = −0.02% relative). This is the dominant outcome.

**The validation gate would actually reject this patch on a strict reading** (median val improvement is essentially zero, win-rate < 50%). On a generous reading (mean val improvement positive due to outlier), it accepts. The test set confirms the mixed picture.

## What this tells us about the framework

**Working**: The mechanical pipeline is sound. The Optimizer reads outcomes, proposes structured edits, the edits get applied, the patched harness runs on held-out tasks, scores are reproducible. As a *system*, this works.

**Not yet working**: Single-shot optimization with only 10 training tasks produces patches that don't reliably generalize. The signal-to-noise ratio is poor:

- 10 train tasks is probably **too few** to identify patterns that generalize across the diversity of metric types and modalities.
- **One Optimizer iteration is too few** — SkillOpt's protocol involves dozens of patches with strict gating, not one bundle of 5.
- **The validation gate is too lenient as defined** — SkillOpt's protocol gates per-edit, not per-bundle. A bundle of "good edit + bad edit" can pass even though the bad edit alone would be rejected.
- **Heterogeneous metrics** make aggregation tricky — a +8% win on QWK doesn't actually offset 4 × 1% losses in the way mean-relative arithmetic suggests, because tasks aren't comparable on a per-percent basis.

## What this means for the paper

This is **honest evidence the framework needs more work** before it can claim "self-improving AutoML harness." The contribution as currently demonstrated is:

> *"We propose harness optimization as a publishable target — a SkillOpt-style optimizer that produces bounded structured edits to an AutoML harness. On a 20-task DSBench pool with 10 train / 5 val / 5 test, a single-iteration optimizer produces patches with marginal mean improvement (val +1.5%, test −0.2%) and a win rate of 4/10. The framework is mechanically sound; producing reliable per-iteration improvement on heterogeneous tasks remains an open problem."*

That's an honest research finding, not a hero-story. It's also useful — it tells the field that **harness optimization isn't trivially better than a frozen hand-written harness**, and suggests what the actual research direction needs (more iterations, stricter gating, larger task pools, or a different optimization strategy).

## Next steps (in order of marginal value)

1. **Run 3-5 more Optimizer iterations** with strict per-edit gating (SkillOpt's actual protocol). Single-bundle, single-iteration optimization is the weakest form of this method — we used it for budget reasons. Compounding signal might emerge over multiple iterations.

2. **Scale train pool to 20-30 tasks** and split 15 train / 5 val / 10 test. With only 10 train tasks, the Optimizer extrapolates from too little data.

3. **Per-edit validation** instead of per-bundle. Apply each edit individually, val-gate it, repeat. SkillOpt does this; we batched.

4. **Use stricter gate**: median rel improvement must be positive (not mean), accept only if win-rate > 50%. Our current gate is lenient.

5. **Bigger optimizer model**: this run used the same Claude as the solver. A separate, larger model for the Optimizer (or longer thinking budget) might propose better-targeted edits.

6. **Comparison to AutoML-Agent baseline**: once Azure access is configured, run AutoML-Agent on the same 20-task split as an external reference point. Until then we only know how our optimized harness compares to *our own initial harness*, not to the field.

## Files

- `/home/colligo/claude-automl/experiments/v5_harness_opt/splits.json` — frozen task splits
- `/home/colligo/claude-automl/experiments/v5_harness_opt/harness/` — H_0 (initial, untouched)
- `/home/colligo/claude-automl/experiments/v5_harness_opt/harness_H_1/` — H_1 (post-Optimizer)
- `/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_0/baseline.json` — H_0 native scores on all 20 tasks
- `/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_1/proposed_patch.json` — Optimizer's proposed edits
- `/home/colligo/claude-automl/experiments/v5_harness_opt/runs/h0_vs_h1_comparison.json` — final per-task comparison
- `/home/colligo/claude-automl/experiments/v5_harness_opt/runs/{H_0,H_1}/<task>/{solution.py,submission.csv,result.json}` — per-run artifacts
- `/home/colligo/claude-automl/experiments/v5_harness_opt/runs/{H_0,H_1}/_scores/<task>/result.txt` — DSBench native scores
