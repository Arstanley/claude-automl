# Skill: When the prompt names multiple unquantified constraints, fix concrete numeric budgets up front

## Precondition
Apply this skill if BOTH of the following hold:

1. The prompt mentions AT LEAST TWO of these constraint axes:
   - latency / inference time / "fast" / "real-time" / p99 / throughput
   - model size / "small" / "fit in memory" / disk size / MB
   - accuracy / F1 / AUC / RMSE (always present, the implicit axis)
   - a hard rule / "must not be" / "should not be" / abstention / "confidently wrong"
   - fairness / sensitive attribute / "must not discriminate"
2. AT LEAST ONE of those axes is named without an explicit numeric budget (e.g. "latency matters" without saying "p99 < X ms").

When met, the spec is multi-objective and the optimum is a Pareto frontier point, not a single number. You MUST commit to concrete numbers for each axis or you cannot defend your modeling choices.

## Action
- For each axis named in the prompt, choose a defensible target NUMBER, derived from the domain context the prompt gives (production service, search backend, mobile, etc.). Examples:
  - "latency matters" + "production search component" → p99 < 100 ms CPU-single-thread (search standard).
  - "model size matters" + no explicit number → ≤ 50 MB on disk (a service typically gets a few hundred MB total, language ID is one of many components).
  - "must not be confidently wrong" → top-prob threshold tuned to ≤ 1% confident-error rate on a probe set.
- Write these numbers to `result.json` under `constraint_budgets: {axis: number}` BEFORE training.
- Measure every model candidate on EVERY axis; reject candidates that violate any budget; pick the accuracy-maximizing candidate among survivors.
- Use a **Pareto-aware model search**: try at least 2 model families with different size/latency profiles (e.g. char-ngram + linear vs subword + small NN; HGB shallow vs LightGBM deep). Don't ship the first model that meets accuracy — show you considered the trade-off.

## Why (failure pattern evidence)
On v3/langid, the Ambig prompt named three axes ("latency matters", "model size matters", "must not be confidently wrong") with ZERO numbers. A_vanilla picked TF-IDF char-ngram + LogisticRegression and reported its numbers — p99=1.22 ms, size=37.7 MB, abstain-tuned threshold — but never wrote down a TARGET for any axis. As a result A could not see that its abstention threshold was too high: 92% abstain rate on real romanized Hindi (the load-bearing input class) was treated as "successfully declined" rather than "model is useless on this slice". C_warned committed to model-size ≤ 50 MB and produced a 22 MB model — half the size, similar accuracy, similar latency. B_audit committed to "must not be confidently wrong" as a NUMBER (≤1% confident-en errors) and achieved 0.12% confident-en errors with explicit Unicode-block routing. Without budgets, A could not see that its operating point was a local optimum within "ship a 20-class classifier" rather than a global optimum within "solve the production routing task".

## Code snippet
```python
# At the top of solution.py, before any modeling:
CONSTRAINT_BUDGETS = {
    "p99_latency_ms": 100,   # search backend SLA
    "model_size_mb": 50,     # service memory budget
    "headline_accuracy_floor": 0.97,  # papluca_test floor
    "hard_rule_max_confident_error_rate": 0.01,  # "must not be confidently wrong"
}

def check_budgets(metrics: dict) -> dict:
    """Return per-axis pass/fail; reject candidate if any axis fails."""
    return {
        axis: metrics.get(axis, float("nan")) <= budget if "max" in axis or "latency" in axis or "size" in axis
        else metrics.get(axis, float("-inf")) >= budget
        for axis, budget in CONSTRAINT_BUDGETS.items()
    }

# After fitting each candidate
m_metrics = evaluate(model)  # returns {p99_latency_ms, model_size_mb, accuracy, ...}
passes = check_budgets(m_metrics)
if not all(passes.values()):
    print(f"  REJECTED: violations = {[k for k, v in passes.items() if not v]}")
    continue
candidates.append((m_metrics["accuracy"], model, m_metrics))

# Persist
result["constraint_budgets"] = CONSTRAINT_BUDGETS
result["budget_compliance"] = passes
```

## Cross-task evidence
- Past failure: v3/langid A_vanilla — three named axes, zero numbers, 92% abstain on Hi-Latn treated as success. C and B both committed to per-axis numbers and produced Pareto-superior models. A was the only "dominated" point on the frontier.
- Partial: v1/bike-sharing-demand — RMSLE was the only explicit axis, but the prompt also mentioned "spec lies about the split" as an implicit robustness axis. Solvers that committed to "report headline on the actual test split distribution" (C) beat solvers that committed to "report headline on the strictest plausible holdout" (B).
