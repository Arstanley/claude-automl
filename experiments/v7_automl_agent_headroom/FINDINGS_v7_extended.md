# v7 Extended — Adding Software Defects + Crab Age (10 settings total)

After the original 4 tasks (8 settings, 6 comparable) we added 2 more AutoML-Agent tasks: **Software Defects** (binary classification, F1) and **Crab Age** (regression, RMSLE). Both use DSBench's mirror of the same Kaggle competitions, with slightly different splits than AutoML-Agent's reported numbers.

Banana Quality and Textual Entailment were attempted but their public mirrors couldn't be located without Kaggle credentials — flagged for future acquisition.

## New task numbers (H_0 baseline only — GEPA not yet applied)

### Software Defects (binary F1)

The AutoML-Agent paper says "the evaluation metric is the F1 score" without specifying which variant. Three variants for the same H_0 predictions:

| Setting | binary F1 | macro F1 | weighted F1 | AutoML-Agent NPS |
|---|---:|---:|---:|---:|
| Free | 0.547 | **0.695** | 0.777 | 0.664 |
| Constraint | 0.548 | **0.701** | 0.786 | 0.573 |

- **If AutoML-Agent reports binary F1** (positive class only): we LOSE both settings (−0.117 free, −0.025 constraint).
- **If they report macro F1**: we WIN both (+0.031 free, +0.128 constraint).
- **If they report weighted F1**: we WIN both (+0.113 free, +0.213 constraint).

The constraint-setting drop in AutoML-Agent (0.664 → 0.573) is unusual — typically constraints don't cause large degradation. Our model is essentially flat (0.547 → 0.548 binary, 0.695 → 0.701 macro), suggesting more robust constraint handling.

### Crab Age (RMSLE → NPS = 1/(1+RMSLE))

| Setting | RMSLE | NPS | AutoML-Agent NPS | Δ |
|---|---:|---:|---:|---:|
| Free | 0.162 | 0.861 | 0.859 | **+0.002** (tie/marginal win) |
| Constraint | 0.163 | 0.860 | 0.861 | −0.001 (tie/marginal loss) |

Essentially equivalent on Crab Age — both us and AutoML-Agent are near the RMSLE-floor for this task.

## Full extended comparison table

Using the most natural reading per AutoML-Agent's prompt — **binary F1** for Software Defects, **RMSLE** for Crab Age (NPS-converted):

| Task | Setting | Metric | Our H_0 NPS | AutoML-Agent NPS | Δ | Verdict |
|---|---|---|---:|---:|---:|---|
| Cora | Free | acc | 0.820 | 0.831 | −0.011 | lose (narrow) |
| Cora | Constraint | acc | 0.904 | 0.843 | +0.061 | WIN |
| Citeseer | Free | acc | 0.715 | 0.592 | +0.123 | WIN |
| Citeseer | Constraint | acc | 0.795 | 0.632 | +0.163 | WIN |
| Higher Ed Students | Free | RI | 0.800 | 0.760 | +0.040 | WIN |
| Higher Ed Students | Constraint | RI | 0.810 | 0.769 | +0.041 | WIN |
| **Software Defects** | **Free** | F1-binary | 0.547 | 0.664 | **−0.117** | **lose** |
| **Software Defects** | **Constraint** | F1-binary | 0.548 | 0.573 | −0.025 | lose (small) |
| **Crab Age** | **Free** | RMSLE-NPS | 0.861 | 0.859 | +0.002 | tie |
| **Crab Age** | **Constraint** | RMSLE-NPS | 0.860 | 0.861 | −0.001 | tie |

**Win rate (H_0, binary F1 interpretation): 5 wins / 2 ties / 3 losses out of 10 settings**.

If macro F1 reading is right: **7 wins / 2 ties / 1 loss out of 10**.

## What this changes

The original 6/6 win story with GEPA was on tasks where AutoML-Agent had clear headroom. The added 4 settings show:

1. **Software Defects is where AutoML-Agent does better** — at least by binary F1. This is a real and interesting case where the older AutoML-Agent harness apparently exceeds Claude Opus 4.7 + minimal scaffolding. Or: it's an F1-variant artifact, where AutoML-Agent's reported number is a different F1.

2. **Crab Age is a near-tie** — both H_0 and AutoML-Agent saturate the same RMSLE floor (~0.16). No meaningful difference.

3. **The Cora-Free loss persists at H_0** — already known, fixed by GEPA's H_A.

## How GEPA could close the gaps

Two clearly-actionable patterns from the new losses:

- **Software Defects F1 binary**: requires explicit decision-threshold tuning on val (the threshold-tuning failure mode we identified back in v5/v6). A careful version of `skill_tune_decision_threshold_for_f1` — using a held-out fold (not the same OOF used for model selection) — should close most of the gap.
- **Reporting clarity**: we should output BOTH binary and macro F1 in our result.json so the comparison isn't variant-dependent.

This would be the next GEPA iteration: candidate **H_C** specializing in F1-threshold-tuning for binary classification with imbalanced classes.

## Honest claims based on all 10 settings

**Stronger claim** (binary F1 reading on Software Defects):
> *"With H_0 alone, Claude Opus 4.7 + minimal harness wins 5 of 10 settings against AutoML-Agent's published NPS, ties on 2, and loses on 3 (notably Software Defects binary F1). GEPA-style optimization closes the original Cora-Free loss and adds Wine Quality gains. The Software Defects gap is the largest remaining headroom for further harness optimization."*

**Generous claim** (macro F1 reading on Software Defects):
> *"With H_0 alone, our harness wins 7 of 10 settings, ties on 2, loses on 1 (Cora-Free, which GEPA fixes). The framework dominates AutoML-Agent across diverse task types."*

The truth is between these. The honest move in a paper is to **report both F1 variants explicitly** and let reviewers decide. We've now made that data available.

## What to do next

Three options, in increasing ambition:

1. **Add Banana Quality + Textual Entailment** (need Kaggle creds or different mirror). Both AutoML-Agent reports near ceiling (0.987 / 0.803), so unlikely to flip the overall pattern but extends N to 14 settings.

2. **GEPA Iteration 2: add H_C for F1-threshold tuning** to close the Software Defects gap. Specifically test on Software Defects (both settings) — if H_C lifts binary F1 from 0.547 to 0.66+, we re-establish the win.

3. **Both #1 and #2** — gives a complete 14-setting comparison with the strongest possible methodology story.

## Files

- `runs/H_0/software_defects__{free,constraint}/` — new H_0 artifacts
- `runs/H_0/crab_age__{free,constraint}/` — new H_0 artifacts
- `runs/extended_v7_scores.json` — machine-readable scores with all F1 variants
- `FINDINGS_v7_extended.md` — this file
- `FINDINGS_v7_phaseB.md` / `FINDINGS_v7_final.md` — earlier writeups (Phase B headline + GEPA-final on original 4 tasks)
