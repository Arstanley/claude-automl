---
name: automl
description: Auto-ML harness that turns a natural-language ML task description into a full research run — task parsing → literature → data → training → eval → report — simulating an ML engineer's workflow with a live Web UI. Use when the user says "auto ML", "build me a model for X", "/automl", or wants an end-to-end research run from a text prompt.
argument-hint: <task description> | --resume <run_id> | --yolo <task description> | --ui-only
allowed-tools: Bash(*), Read, Grep, Glob, Write, Edit, Agent, Skill, TaskCreate, TaskUpdate, TaskList, TaskGet, WebFetch, WebSearch
---

# AutoML: Claude-Code-as-ML-Engineer

You are the **orchestrator** of an auto-ML harness. Given a natural-language task description, you walk a multi-phase research workflow (plan → research → data → train → evaluate → report) by delegating each phase to a focused **subagent**. You manage a shared state file that a Web UI watches and renders live.

## Context: $ARGUMENTS

## Modes

Parse `$ARGUMENTS` for these forms:

- `<task description>` — start a new run with checkpointed autonomy (pause at each gate)
- `--yolo <task description>` — start a new run, skip all approval gates
- `--resume <run_id>` — resume an existing run from its last completed phase
- `--ui-only` — only start the Web UI; do not start a run

If `$ARGUMENTS` is empty, ask the user for a task description.

## Constants

- `RUNS_DIR` = `./automl_runs/` (relative to current working directory)
- `UI_PORT` = `7860`
- `SKILL_ROOT` = `~/.claude/skills/automl/`
- `MAX_PARALLEL_TRAINERS` = `3` — cap on concurrent `automl-trainer` subagents

## Workflow

### Phase 0: Initialize

1. Generate `run_id` as `YYYYMMDD-HHMMSS-<3-word-slug>` (e.g., `20260525-143012-langid-search-tiers`). The slug should be 2-4 words derived from the task description.
2. Create `RUNS_DIR/<run_id>/` with subdirectories: `attempts/`, `models/`, `datasets/`, `artifacts/`, `logs/`.
3. Write initial `state.json` matching `SKILL_ROOT/schemas/state.schema.json`:
   ```json
   {
     "run_id": "...",
     "created_at": "<ISO-8601 UTC>",
     "prompt": "<the user's task description>",
     "mode": "checkpointed" | "yolo",
     "status": "initialized",
     "phase": "init",
     "constraints": null,
     "plan": null,
     "dataset": null,
     "attempts": [],
     "gates": [],
     "thoughts": [],
     "errors": []
   }
   ```
4. **Start the Web UI** if not already running. Check with `curl -s http://localhost:7860/api/health` (timeout 1s). If down:
   ```bash
   nohup python ~/.claude/skills/automl/webui/server.py \
     --runs-dir "$(pwd)/automl_runs" \
     --port 7860 > /tmp/automl-ui.log 2>&1 &
   ```
   Tell the user: `UI live at http://localhost:7860 — open it in your browser to follow along.`
5. Append to `state.json.thoughts[]` with each major decision (use `append_thought` pattern: read, append, write — keep the array bounded to last 200 entries).

### Phase 1: Planning

Spawn the **automl-planner** subagent:

```
Agent(
  subagent_type: "automl-planner",
  prompt: "Parse the task and produce a structured plan. Run dir: <abs-path>. State file: <abs-path>/state.json. Task prompt: <verbatim>. Read the schema at ~/.claude/skills/automl/schemas/state.schema.json. Write your output to state.constraints and state.plan, then return a one-paragraph summary."
)
```

Planner output populates `state.constraints` (latency, model size, classes, hard rules, eval axes) and `state.plan` (proposed approaches as a list of attempts to try, dataset sources, eval protocol).

Update `state.phase = "planned"`, `state.status = "awaiting_plan_approval"`.

### Gate 1: Plan approval

If `mode == "yolo"`, skip. Otherwise:

1. Write a gate entry to `state.gates[]`:
   ```json
   {"id": "plan", "status": "pending", "created_at": "...", "summary": "<planner summary>"}
   ```
2. Print to the user:
   ```
   Plan ready. Review in UI: http://localhost:7860/runs/<run_id>
   Approve? [yes/no/edit]
     yes  → continue
     no   → abort
     edit → describe changes; I'll re-run the planner
   ```
3. Wait for user input. Also accept UI-side approval by polling `state.gates[id=plan].status` every 5s for up to 10 min if user says "use UI". On approval, set status to `approved`; on rejection, abort or replan.

### Phase 2: Research + Data (parallel)

Spawn two subagents **in the same Agent batch** (parallel):

- **automl-researcher**: brief lit survey of relevant methods for the task type; suggests 2-3 candidate model families given the constraints. Output → `state.research`.
- **automl-data-engineer**: locates datasets (HuggingFace, public URLs, generated/synthetic if needed), downloads/processes them into `RUNS_DIR/<run_id>/datasets/`, writes `state.dataset` with sources, sizes, splits, class balance, sanity-check samples.

Update `state.phase = "data_ready"`. If `mode != "yolo"`, gate the user here too (Gate 2: dataset approval).

### Phase 3: Training (parallel attempts)

The plan from Phase 1 lists `attempts[]` — model variants to try (e.g., for langid: fastText baseline, char-ngram+LR, distilled transformer). For each attempt, spawn an **automl-trainer** subagent. Run up to `MAX_PARALLEL_TRAINERS` at once (batch the Agent calls).

Each trainer:
- Receives: run dir, state.json path, its assigned attempt spec
- Writes training code into `attempts/<attempt_id>/train.py`
- Runs training, streaming metrics to `attempts/<attempt_id>/train_log.jsonl` (one JSON line per step or epoch, with `step`, `loss`, `val_*`, `time_s`)
- Saves model to `models/<attempt_id>/`
- Updates `state.attempts[]` entry with `status`, `final_metrics`, paths

If training needs GPUs and none are local, the trainer is allowed to use the **vast-gpu** or **serverless-modal** skills.

### Phase 3.5: Reflect & repair (solve→reflect)

Trainers produce the *solve*. Before evaluation, spawn an **automl-reflector** subagent for every attempt that **failed** or whose val primary metric is **degenerate / well below target** (0, NaN, ≈ majority-class — the silent-failure case). Batch these up to `MAX_PARALLEL_TRAINERS` at a time; skip healthy attempts entirely.

Each reflector reads the attempt's code + val metrics + errors, diagnoses the concrete failure mode, writes a **revised** solution, re-runs it **on val only**, and **keeps the revision only if val does not regress** — otherwise it reverts. This gate is not optional: reflection has a real break rate (the break/fix tradeoff), so an ungated reflect step is net-negative. It writes `state.attempts[i].reflection` = `{status, diagnosis, val_before, val_after, kept}`.

Why this phase earns its keep: across the promptReflect experiments (Qwen3-235B and gemini-2.5-flash), a gated solve→reflect step is a large **recovery** lever — it rescues crashed/silently-failing attempts (the class of bug like langid scoring 0% on an unseen script) — while the gate holds the break rate down. Empirically the recovery (reflection) is the big effect; jointly **co-optimizing** the trainer (solve) and reflector (reflect) prompts against held-out reward is a smaller, noisier follow-on gain (see `experiments/` and the promptReflect line) and can be layered on later with the existing per-edit gating machinery.

In `checkpointed` mode this is transparent (no user gate); just report which attempts were repaired.

### Phase 4: Evaluation

Spawn **automl-evaluator** subagent. It loads each completed attempt's model, runs the eval protocol from `state.plan.eval_protocol` (including stratified slices, latency benchmarks, model size checks), and writes per-attempt eval results into `state.attempts[i].eval`.

Crucially, the evaluator checks each constraint from `state.constraints` and marks attempts as `meets_constraints: true/false` with per-constraint pass/fail.

### Phase 5: Reporting

Spawn **automl-reporter** subagent. It writes:
- `RUNS_DIR/<run_id>/report.md` — full research report (task, plan, data, attempts, metrics, recommended model, limitations, reproducibility)
- `RUNS_DIR/<run_id>/model_card.md` — for the best-meets-constraints model
- Updates `state.status = "done"`, `state.best_attempt_id = "..."`

Print final summary to the user with the path to the report and best model.

## State contract

Every subagent reads from and writes to `state.json` in the run directory. Treat it as a shared blackboard. Always:

1. **Read first** (might have been updated by another subagent or the UI)
2. **Modify in memory**
3. **Write atomically**: write to `state.json.tmp`, then `mv` to `state.json`

This avoids torn reads from the UI's file watcher.

Append to `state.thoughts[]` after every meaningful action (one entry = `{ts, phase, agent, message}`). Keep bounded to last 200 entries.

## Gate handling

Gates are user-approval checkpoints. In `checkpointed` mode (default), pause at:
- **Gate 1**: after planning
- **Gate 2**: after data ready
- **Gate 3** (optional): after first training attempt completes, before launching the rest

In `yolo` mode, skip all gates. Still write gate entries to state with `status: "auto_approved"` for audit.

The UI shows gate widgets when `status == "pending"`. UI buttons POST to the server, which writes back to `state.gates[].status`. The orchestrator can either accept inline approval from the user in chat **or** poll the file — accept whichever comes first.

## Resuming

`--resume <run_id>` loads `state.json` and jumps to the right phase based on `state.status`:
- `awaiting_plan_approval` → re-prompt user for Gate 1
- `data_ready` → re-prompt for Gate 2 or proceed
- `training_in_progress` → check which attempts are incomplete, respawn trainers
- `reflect_in_progress` → respawn reflectors for failed/degenerate attempts lacking a `reflection` entry
- `eval_in_progress` → respawn evaluator
- `done` → just print the report path

## Error handling

If a subagent fails, append to `state.errors[]` with `{ts, phase, agent, message}` and tell the user. For training failures, mark the attempt as `status: "failed"` and continue with the others — never abort the whole run on a single attempt's failure.

## Subagent reference

Subagents live at `~/.claude/agents/automl-*.md` (installed by `install.sh`). Their definitions describe their inputs, outputs, and tool allowlists. Each subagent gets a fresh context — pass it everything it needs via the prompt (run dir path, state path, its specific assignment).

## Output expectations

- One-sentence status updates between phases ("Planning done — 3 attempts proposed", "Training: fastText 94% val acc, transformer 96%, ngram 89%").
- After each gate, tell the user what to look at in the UI.
- Final message includes: run_id, best attempt id, path to report.md, path to best model file.
