# Improved AutoML harness (`improved-harness` branch)

This branch is the baseline `/automl` skill with **distilled tabular-ML lessons** baked into two
agents. It is a drop-in replacement for the agents on `master` — same scaffold, same Web UI.

## What changed vs `master`

Only two files differ (13 lines total):

- **`agents/automl-trainer.md`** — adds a *tabular accuracy playbook* (feature engineering →
  k-fold out-of-fold model selection → ensemble of ≥2 diverse families → seed-bagging),
  a *small-data guard* (restraint < ~2000 rows), and *task-type adaptation* (regression /
  multiclass / linear-model specifics, metric-aware effort).
- **`agents/automl-planner.md`** — adds a *tabular task playbook* so the plan calls for FE,
  CV-OOF selection, and an ensemble attempt rather than a single untuned model.

All other agents, `SKILL.md`, the Web UI, and schemas are unchanged.

## Why (evidence)

These lessons were distilled by a reflection loop and then validated on held-out tasks:

- **Tabular (8 held-out tasks):** improved harness beats baseline by **+0.0128 mean** primary
  metric, no significant regressions in this split.
- **Real text task — search-query language ID** (out of domain): improved harness beats baseline
  by **+0.0413 macro-F1** (0.905 → 0.946), 95% CI [+0.036, +0.047], reproducible across replicates.
  The gain is *methodology transfer* — deployment-matched cross-validation, optimizing the real
  metric (macro-F1), and ensembling — even though the tabular feature tricks don't apply to text.

See `experiments/reflection_distill/FINDINGS_langid.md` and `FINDINGS_selfroute.md` for details.

## Known tradeoff to watch when testing

On the language-ID task the stronger classifier became **more eager to assign out-of-distribution
text to a supported class** — e.g. romanized Hindi mislabeled as English rose 1.5% → 5.0%. If your
task has an OOD/"unsupported" guard or a per-slice constraint, encode it as a planner `hard_rule`;
the harness optimizes the aggregate metric and won't protect that slice on its own.

## How to test

```bash
git fetch origin && git checkout improved-harness
bash install.sh                 # symlinks agents/*.md into ~/.claude/agents/, installs Web UI deps
# ensure the skill is discoverable: ~/.claude/skills/automl should point at this checkout
```

Then in Claude Code:

```
/automl <your natural-language ML task>      # e.g. the prompt in examples/langid/prompt.md
```

Runs land in `./automl_runs/<run_id>/`; the dashboard (auto-launched, or
`python webui/server.py --runs-dir ./automl_runs --port 7860`) shows live progress.

To A/B against baseline, run the same task on `master` vs this branch and compare the run reports.
