# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Claude Code **skill**, not an application. `/automl <task>` turns a natural-language ML task description into a multi-phase research run (plan → research → data → train → eval → report). The orchestrator is `SKILL.md`; each phase is a subagent under `agents/automl-*.md`. The only executable code is the FastAPI Web UI in `webui/`.

Runs land in `./automl_runs/<run_id>/` relative to the user's cwd — never inside this repo (see `.gitignore`).

## Architecture: the state.json blackboard

Everything coordinates through a single file: `automl_runs/<run_id>/state.json`. Subagents and the Web UI all read and write it.

- Schema lives in `schemas/state.schema.json` (run-level) and `schemas/attempt.schema.json` (per-attempt). Any new field added to state must be reflected in the schema.
- **Writes must be atomic**: write `state.json.tmp`, then `mv` to `state.json`. The UI's SSE watcher polls mtime every 1s and will read torn JSON otherwise. `webui/server.py:save_state_atomic` is the reference implementation.
- **Always read-modify-write** — another subagent or the UI may have updated state since you last looked.
- Subagents never touch each other's fields. That field-level ownership is what allows the trainer subagents to run safely in parallel (`MAX_PARALLEL_TRAINERS = 3` from `SKILL.md`).
- `state.thoughts[]` is bounded to the last 200 entries; trim when appending.

## Repo layout

- `SKILL.md` — the orchestrator skill, frontmatter declares allowed tools. Loaded when the user types `/automl`.
- `agents/automl-*.md` — six subagent definitions (planner, researcher, data-engineer, trainer, evaluator, reporter). Each has its own frontmatter (`name`, `description`, `tools`, `model`).
- `webui/server.py` — FastAPI + SSE dashboard. Single-file server, no DB, reads state from filesystem on every request.
- `webui/static/{index.html,app.js,style.css}` — vanilla JS dashboard that parses state.json into structured components and subscribes to `/api/runs/{id}/events` for live updates.
- `schemas/` — JSON Schema sources of truth for the state blackboard.
- `install.sh` — symlinks `agents/*.md` into `~/.claude/agents/` and `pip install`s the webui deps. Idempotent.
- `examples/langid/prompt.md` — canonical example task prompt.

## Common commands

```bash
# Install (symlinks agents into ~/.claude/agents/, installs UI deps)
bash install.sh

# Optionally install UI deps into a venv
AUTOML_VENV=~/.venvs/automl bash install.sh

# Run the Web UI standalone (skill normally launches this in background)
python webui/server.py --runs-dir ./automl_runs --port 7860

# Health check
curl -s http://localhost:7860/api/health
```

There is no test suite, no linter config, no build step. Changes to `SKILL.md` or `agents/*.md` take effect on the next `/automl` invocation; changes to `webui/server.py` require restarting the UI.

## Gotchas to remember

- **FastAPI + `from __future__ import annotations`**: Pydantic body models referenced via `Body(...)` must be defined at module scope, not inside `make_app`. With postponed annotations, FastAPI can't resolve a class captured in a closure. `GateAction` is module-level in `server.py:48` for this reason — don't move it back.
- **Path traversal**: `safe_run_dir` and `safe_run_file` in `webui/server.py` are the only sanctioned ways to resolve a run/path from a URL. Anything new that takes a `run_id` or relative `path` from the client must go through them.
- **Skill discovery**: The skill is read from `~/.claude/skills/automl/` (typically a clone/symlink of this repo). `install.sh` does **not** copy `SKILL.md` itself — Claude Code is expected to find the repo at that path. If editing `SKILL.md` doesn't seem to take effect, check that `~/.claude/skills/automl` actually points here.
- **Agents are markdown, not code**: extending a phase means editing the relevant `agents/automl-*.md`. The frontmatter `tools:` line is the agent's tool allowlist — keep it minimal.
- **Run dir is relative to user cwd, not skill root**: `RUNS_DIR = ./automl_runs/`. Don't ever hardcode an absolute path inside `SKILL.md` or the agents.

## When extending

- **New task type (classification/regression/retrieval)**: usually no code changes — the planner adapts attempts to the constraints. Generation or custom-eval tasks (factuality, fluency, image quality) need an addendum in `agents/automl-evaluator.md`.
- **New state field**: update `schemas/state.schema.json` and document the owning subagent in that agent's `.md`. The UI will surface unknown top-level fields under a generic panel but won't render them well.
- **New API endpoint**: add to `make_app` in `webui/server.py`, use `safe_run_dir`/`safe_run_file` for any client-supplied path component, and emit through SSE if the dashboard should react live.
