# v5 Diagnostic — Is the Optimizer even learning?

After producing H_1 we ran the basic ML sanity check: **does H_1 beat H_0 on the training tasks the Optimizer saw?** If not, the framework isn't even fitting in-sample and we have a bigger problem than generalization.

## The three-way comparison

| Split | N | Wins | Mean rel% | Median rel% |
|---|---:|---:|---:|---:|
| **TRAIN** | 10 | **6** | **+0.06%** | **+0.24%** |
| VAL | 5 | 2 | +1.46% | −0.02% |
| TEST | 5 | 2 | −0.17% | −0.32% |

## Reading the numbers

**Good news**: training signal exists. 6/10 train tasks improve, median rel +0.24%. The Optimizer **is** learning something — it's not random noise.

**Bad news**: the training signal is tiny. Most train wins are <2% relative; the mean is barely above zero because the wins and losses nearly cancel. In ML terms, "training loss is decreasing, but barely."

**Worse news**: generalization is poor. Val and test win rates drop to 40%, and the test mean goes negative. **Whatever the Optimizer learned on train doesn't transfer.**

## The classic ML signature

This is exactly what you'd see for a model that:
1. Has *some* gradient signal (6/10 train wins, not 5/10)
2. But the signal is weak relative to noise (median +0.24% on train)
3. And what it *does* fit doesn't generalize (test rel% goes negative)

In normal ML, the prescription would be: more data, regularization, smaller model class, longer training. For us: more tasks, stricter gating, smaller-grained edits, more iterations.

## Per-task forensics — where did learning happen vs not?

### Wins on train (where the Optimizer found real patterns)

| task | metric | rel% | What skill drove it |
|---|---|---:|---|
| playground-series-s3e22 | F1 micro | +1.70% | missing-indicator + stringify-codes (carried from v4) |
| playground-series-s3e9 | RMSE | +1.53% | blend + log-target A/B (new) |
| playground-series-s3e3 | AUC | +1.19% | tune-when-supports-it + blend (new) |
| titanic | acc | +0.68% | missing-indicator (carried) |
| playground-series-s4e2 | acc | +0.24% | tune (carried) |
| playground-series-s3e2 | AUC | +0.18% | blend (new) |

### Losses on train (where the new skills hurt)

| task | metric | rel% | Likely cause |
|---|---|---:|---|
| bike-sharing-demand | RMSLE | **−2.77%** | blend hurt — second family didn't help two-head modeling |
| spaceship-titanic | acc | −0.78% | new complexity didn't pay off; H_0 was already optimal |
| playground-series-s3e6 | RMSE | −0.82% | log-target A/B chose identity (right call) but slight ship-time noise |
| nlp-getting-started | F1 | −0.60% | **threshold-tuning OVERFIT OOF**: OOF F1 0.7588→0.7660 but native test F1 went down 0.7687→0.7641 |

The nlp result is the most informative single failure mode: the F1 threshold skill optimizes the OOF threshold and ships it as the production threshold. On nlp, this **moved the OOF in the right direction but pushed the test in the wrong direction** — a textbook OOF-threshold-overfitting story. That's a fixable bug in the skill, but it ships because the patch bundle was accepted as a whole rather than per-edit.

## What this tells us about the framework

The diagnostic is much more useful than the val/test numbers alone. It tells us:

1. **The Optimizer's gradient signal is real but weak.** With 10 train tasks and 1 iteration, we got median +0.24% on train. To produce meaningfully larger gradients, we need either more tasks per iteration or more iterations to compound.

2. **There's an overfitting failure mode at the skill level.** `skill_tune_decision_threshold_for_f1` over-fit OOF F1 on nlp-getting-started. Per-bundle validation can't catch this — only per-edit ablation would. We need finer-grained gating.

3. **Some new skills are genuinely useful.** `skill_blend_two_model_families` produced clean wins on s3e9 (+1.5%), s3e3 (+1.2%), s3e2 (+0.2%), and s3e5 (val: +8.6%). The pattern: family-diverse blending works when n is moderate and the dataset doesn't have a dominant pre-existing single best model.

4. **Some skills are noise**: log-target A/B-checked correctly on s3e6 (chose identity = correct) but the act of A/B-checking itself introduced small overhead variance. Skills with "A/B and ship the winner" patterns add variance even when correct.

## Concrete next moves (in order of expected payoff)

1. **Per-edit gating instead of per-bundle.** Apply each Optimizer edit individually, val-gate it, keep or discard. This is SkillOpt's actual protocol. With current setup we'd have killed the F1 threshold edit and kept the blend skill.

2. **More training tasks.** Scale to 20-30 train tasks. Pattern signal scales with N.

3. **More Optimizer iterations.** Single-shot saw +0.06% mean. Compounding signal across iterations is the canonical SkillOpt outcome; we didn't test it.

4. **Independent validation set for skill threshold tuning.** Skills that tune thresholds (like F1 skill) must use a held-out fold for the threshold, not the same OOF used for model selection.

5. **Per-task-type skill specialization.** Several losses are "skill applied to wrong task type" (blend on bike-sharing where two-head was already optimal; threshold tune on text-classification where the underlying model was the bottleneck not the threshold). Skills could carry task-type preconditions, not just data-shape preconditions.

## Bottom line for the paper

The honest one-line summary is:

> *"A single-iteration harness optimizer with N=10 train tasks produces small but positive training-set improvements (6/10 wins, median +0.24%) that fail to generalize (test mean −0.17%). Diagnostic per-task analysis reveals two distinct failure modes — per-bundle acceptance hiding individual bad edits, and OOF-threshold overfitting — both of which point to per-edit gating and larger train pools as the obvious next experiments."*

That's a real research finding. It's not the dramatic win story, but it's defensible, methodologically honest, and points concretely at what to fix.
