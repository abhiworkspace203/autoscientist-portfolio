# Critic Agent (Week 3) - 08/11/2026

Prepared by: Sachin (Academic Critic Agent & QA Lead)
Code: `src/agents/critic_agent.py`  |  Tests: `tests/test_critic.py`

The Critic reviews one finished experiment like a referee for a top venue (Nature / NeurIPS)
and returns a structured verdict with critique bullets.

## Inputs

| Input | Source |
|---|---|
| Hypothesis | `selected_hypothesis` in the graph state |
| Experiment code and result | `generated_code`, `execution_result` (sandbox / Code Executor) |
| Statistics | dict from `src/utils/stats_validator.py` (Member 4) |

Statistics format (from the Week 3 plan):
`{"p_value", "is_significant", "confidence_interval_95", "effect_size_cohens_d", "summary"}`.
For the sample-size check the validator should also include `n_baseline` and `n_treatment`
(group sizes). Without them the Critic cannot confirm the sample size and will not ACCEPT.

## Output (`CriticReview`, a Pydantic model)

```
VERDICT: ACCEPTED | REJECTED | NEEDS_REPLICATION
- critique bullet
- critique bullet ...
```

`review.verdict`, `review.checks`, `review.critique`, `review.p_value`, `review.as_text()`,
`review.model_dump()` (plain dict for the graph state).

## How the verdict is decided

| Situation | Verdict |
|---|---|
| No p-value (statistics missing) | NEEDS_REPLICATION |
| Experiment does not test the hypothesis (Check 1) | REJECTED |
| p >= 0.05 and sample size adequate or unknown | REJECTED (inconclusive) |
| p >= 0.05 and sample too small (n < 30 per group) | NEEDS_REPLICATION |
| p < 0.05 but small or unreported sample, negligible effect (abs(d) < 0.2), or interval containing 0 | NEEDS_REPLICATION |
| p < 0.05, adequate sample, meaningful effect, hypothesis tested | ACCEPTED |

The statistics decide the verdict. An optional LLM (any object with `complete(prompt) -> str`,
the same interface as `llm_client_from_env()` in `hypothesis_agent.py`) may judge Check 1 and add
critique text, but it can never turn a non-significant result into ACCEPTED. Without an LLM
the Critic runs fully offline.

## Wiring it into the orchestrator (for `critic_node`)

```python
from src.agents.critic_agent import critique_experiment

review = critique_experiment(
    state["selected_hypothesis"],
    state["execution_result"],
    stats,                              # from stats_validator
    code=state.get("generated_code", ""),
)
return {"critic_verdict": review.verdict, "critic_review": review.model_dump()}
```

Route on `review.verdict`: `ACCEPTED` -> END, `REJECTED` -> next iteration or parameter tuning,
`NEEDS_REPLICATION` -> rerun with more samples.

## Limitations

- Check 1 offline is a keyword heuristic (share of the hypothesis' key terms found in the
  experiment). It is a sanity check, not a proof. The Week 2 Code Executor only simulates random
  numbers, so the Critic correctly REJECTS it as not testing the hypothesis until the Code Executor
  generates real experiments.
- Direction of the effect is not checked (a significant result in the wrong direction is not detected).
- On synthetic data (100 seeded runs, n = 40 per group): real effects accepted 100/100, pure noise
  accepted 3/100. Noise passes only when it is significant by chance, at about the 5% rate p < 0.05 allows.
