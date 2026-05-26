# AutoML — Claude Code as an ML researcher

`/automl` is a [Claude Code](https://docs.claude.com/en/docs/claude-code) skill
that turns a natural-language ML task description into a full research run.
It mirrors how a real ML engineer works — parse the task, scan the literature,
acquire data, train several model variants in parallel, evaluate them against
the user's actual constraints, and write up the result — and shows everything
live in a local Web UI.

## What it does

```
You: /automl Build me a CPU-fast language identifier for Adobe search.
     Tier-1 langs are EN/FR/DE/JA; tier-2 SP/IT/KO/PT-BR; tier-3 NL/SV/... .
     ≤100ms p99 CPU, ≤10GB model, must NOT misclassify romanized Hindi as English.

AutoML: ► UI: http://localhost:7860
        ► Planning…
        ► Plan ready — 3 attempts proposed (fastText, char-ngram+LR, distilled XLM-R).
          Approve in chat or in the UI.
        [approve]
        ► Data engineer + researcher running in parallel…
        ► 3 trainers running in parallel…
        ► Eval: fastText 91% macro-F1 / 2.1ms p99 / 300MB / all constraints pass
                xlmr    94% / 850ms p99 / 480MB / latency fail
                ngram   86% / 0.4ms / 80MB / pass
        ► Best: fastText.
        ► Report: automl_runs/20260525-143012-langid-search-tiers/report.md
```

## Architecture

```
/automl <prompt>
     │
     ▼ (orchestrator in SKILL.md)
   ┌─────────────┐
   │ state.json  │ ◄── single blackboard, written atomically by every agent
   └──────┬──────┘
          │ watched by
          ▼
   ┌─────────────┐
   │   Web UI    │   FastAPI + SSE, http://localhost:7860
   └─────────────┘

Subagents (Claude Code Agent tool):
  automl-planner       prompt → constraints + experiment plan
  automl-researcher    brief lit scan, gotchas, baselines
  automl-data-engineer download + process + split datasets
  automl-trainer       one per attempt; parallel; streams metrics
  automl-evaluator     latency bench + slice analysis + constraint checks
  automl-reporter      report.md + model_card.md
```

The orchestrator is just markdown (`SKILL.md`); each subagent is just
markdown (`agents/automl-*.md`). The only real code is the Web UI server
and the per-run training scripts (which are themselves written *by* the
trainer subagent at run time, not shipped in this repo).

## Install

```bash
git clone https://github.com/your-org/claude-automl ~/.claude/skills/automl
bash ~/.claude/skills/automl/install.sh
```

Requirements:
- Claude Code installed
- Python 3.10+ with pip
- Optionally `AUTOML_VENV=~/.venvs/automl` env var to install into a venv

`install.sh` symlinks the six subagent files into `~/.claude/agents/` and
installs `fastapi`, `uvicorn`, `pydantic` for the Web UI.

## Use

In Claude Code:

```
/automl <task description>
```

Variants:
- `/automl --yolo <task>` — skip all approval gates (run fully autonomously)
- `/automl --resume <run_id>` — resume an interrupted run
- `/automl --ui-only` — just start the dashboard, no run

On first invocation the UI starts in the background on port 7860. Open
`http://localhost:7860` to follow along.

## Run directory layout

Each run lives in `./automl_runs/<run_id>/` (relative to where you invoked
`/automl`):

```
<run_id>/
  state.json            ← the blackboard (UI watches this)
  attempts/<id>/
    train.py            ← written by the trainer
    train_log.jsonl     ← streaming metrics (UI plots this live)
    config.json
    stdout.log
  models/<id>/          ← saved model artifacts
  datasets/
    raw/                ← downloads
    processed/
      train.jsonl, val.jsonl, test.jsonl
      test_slices/<slice>.jsonl
    dataset_card.md
    stats.json
  artifacts/            ← shared eval scripts, plots, etc.
  logs/                 ← orchestrator log
  report.md             ← final research report
  model_card.md         ← model card for the best attempt
```

## Constraints, gates, autonomy

The harness reads quantitative constraints (latency, model size, accuracy
targets) and hard rules ("must not misclassify romanized Hindi as English")
from your prompt and **enforces them at eval time** — every attempt gets a
pass/fail table for each constraint. If no attempt meets all constraints,
the report says so plainly.

By default the harness pauses for approval at three gates: after planning,
after data preparation, and (optionally) after the first attempt completes.
You approve from chat (`yes`/`no`/`edit`) or by clicking in the UI. Use
`--yolo` to skip gates entirely.

## Subagent contracts

Each subagent reads `state.json` and writes specific fields back. The full
schema lives in `schemas/state.schema.json` and `schemas/attempt.schema.json`.
Subagents never touch each other's fields, which is what lets the trainers
run safely in parallel.

## Extending

Adding a new task type just means: pick datasets your `automl-data-engineer`
can find, and the `automl-planner` already adapts its attempts to the
constraints you give it. No code changes needed for most ML problems
(classification, regression, retrieval). For generation tasks or anything
that needs custom eval (e.g., factuality, fluency, image quality), you may
want to add a task-specific addendum to `automl-evaluator.md`.

## Status

This skill is intentionally small and self-contained so it can be lifted
out of `~/.claude/skills/` into its own GitHub repo. There is no package,
no entry point — Claude Code reads the markdown directly.

## License

MIT.
