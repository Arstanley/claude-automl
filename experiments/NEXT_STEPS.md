# Next steps after v7

## Staged GEPA-style refinement (user's idea — note for future work)

**Reference**: [arxiv:2410.13509](https://arxiv.org/abs/2410.13509) — iterative refinement of staged systems where each stage's output is the next stage's input.

**The problem with our current single-shot Optimizer**: We patch the whole harness at once. But the harness has *stages*:

```
planning  →  data acquisition  →  execution  →  verification
```

Stages are interdependent. Changing the planning harness can produce plans the data harness can't consume; changing the data harness can produce schemas the execution harness doesn't expect. A single-shot Optimizer that touches multiple stages may produce a harness that's individually-improved per stage but broken in composition.

**GEPA-staged proposal**:

1. **Specialize reflection per stage**: separate Reflector agents for planning / data / execution / verification, each diagnosing failures specific to their stage.
2. **Stage-wise iterative update**: improve one stage at a time, holding others fixed. After updating the planning stage, run the full pipeline again and observe whether the downstream stages still consume correctly. If they break, back out the change.
3. **Inter-stage compatibility checks**: explicit "contract" between stages (e.g., what schema does planning produce? what does data expect?). When patching one stage, verify the contract isn't violated.
4. **Pareto frontier per stage**: GEPA's archive, but per-stage. The "best planner" + "best data engineer" + "best trainer" can be different variants combined at runtime.

**Why this matters**: our v5/v6 results showed that single-iteration multi-edit bundles either ship bad edits (v5) or get too conservative when gated (v6). Staged refinement could let us accept good planning edits and good execution edits *separately*, with composition validation in between.

**Concrete experiment to run later**: 
- Decompose our current harness into 4 explicit stages (we already have 6 subagent files — group them as planner/data/execution/verifier).
- Run separate Reflector agents per stage.
- Update one stage at a time; verify composition.
- Compare to single-shot Optimizer baseline.

## Other deferred ideas

- **AutoML-Agent baseline head-to-head** (this is v7, currently in progress).
- **MLE-bench evaluation** — external standard benchmark, needs GPU + Kaggle data.
- **Multi-iteration on the v6 setup** — Phases C-E from v6 plan.
- **Reflective NL diagnosis step** before propose (the core GEPA innovation we haven't tried).
