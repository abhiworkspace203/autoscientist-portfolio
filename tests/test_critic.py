"""
Tests for the Critic Agent (Member 5).

The Critic must approve significant findings and reject random noise.
All tests are offline: no LLM, no network, no Docker.
Statistics for the noise-vs-signal test come from scipy directly (not from
Member 4's validator), so this file does not depend on anyone else's code.
"""
import json

import pytest
from pydantic import ValidationError

from src.agents.critic_agent import (
    CriticReview,
    build_critic_prompt,
    critique_experiment,
    hypothesis_coverage,
)

HYP = "Hypothesis: Dynamic token pruning reduces inference memory by 30% with <1% perplexity loss."
RELEVANT_RESULT = "Measured inference memory and perplexity for token pruning vs the full model."
UNRELATED_RESULT = "Baseline mean: 100.1\nMethod mean: 102.0\nMetric delta: 1.9"

SIGNIFICANT = {
    "p_value": 0.003,
    "is_significant": True,
    "confidence_interval_95": [2.1, 6.4],
    "effect_size_cohens_d": 0.82,
    "n_baseline": 40,
    "n_treatment": 40,
    "summary": "Statistically significant improvement over baseline with large effect size.",
}
NOISE = {
    "p_value": 0.62,
    "is_significant": False,
    "confidence_interval_95": [-1.5, 2.4],
    "effect_size_cohens_d": 0.05,
    "n_baseline": 40,
    "n_treatment": 40,
    "summary": "No significant difference.",
}


class FakeLLM:
    """Stands in for a real LLM and records the prompts it receives."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def review(stats=None, result=RELEVANT_RESULT, **kwargs):
    return critique_experiment(HYP, result, stats, **kwargs)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
def test_accepts_a_significant_finding():
    assert review(SIGNIFICANT).verdict == "ACCEPTED"


def test_rejects_a_non_significant_finding():
    result = review(NOISE)
    assert result.verdict == "REJECTED"
    assert result.checks.significant is False


def test_p_value_exactly_0_05_is_not_significant():
    assert review({**NOISE, "p_value": 0.05}).verdict == "REJECTED"


def test_rejects_when_the_experiment_does_not_test_the_hypothesis():
    result = review(SIGNIFICANT, result=UNRELATED_RESULT)
    assert result.verdict == "REJECTED"
    assert result.checks.hypothesis_tested is False


def test_significant_but_small_sample_needs_replication():
    result = review({**SIGNIFICANT, "n_baseline": 6, "n_treatment": 6})
    assert result.verdict == "NEEDS_REPLICATION"
    assert result.checks.sample_size_adequate is False


def test_not_significant_with_tiny_sample_needs_replication():
    result = review({**NOISE, "n_baseline": 5, "n_treatment": 5})
    assert result.verdict == "NEEDS_REPLICATION"


def test_unreported_sample_size_blocks_acceptance():
    stats = {k: v for k, v in SIGNIFICANT.items() if not k.startswith("n_")}
    result = review(stats)
    assert result.verdict == "NEEDS_REPLICATION"
    assert result.checks.sample_size_adequate is None


def test_negligible_effect_needs_replication():
    result = review({**SIGNIFICANT, "effect_size_cohens_d": 0.05})
    assert result.verdict == "NEEDS_REPLICATION"


def test_interval_containing_zero_contradicts_significance():
    result = review({**SIGNIFICANT, "confidence_interval_95": [-0.5, 3.0]})
    assert result.verdict == "NEEDS_REPLICATION"


def test_missing_statistics_needs_replication():
    assert review(None).verdict == "NEEDS_REPLICATION"
    assert review({}).verdict == "NEEDS_REPLICATION"


def test_p_value_wins_over_a_wrong_is_significant_flag():
    result = review({**NOISE, "is_significant": True})
    assert result.verdict == "REJECTED"
    assert any("disagrees" in line for line in result.critique)


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------
def test_text_output_starts_with_verdict_line_then_bullets():
    lines = review(SIGNIFICANT).as_text().splitlines()
    assert lines[0] == "VERDICT: ACCEPTED"
    assert all(line.startswith("- ") for line in lines[1:])
    assert len(lines) > 3


def test_critique_is_never_empty():
    for stats in (SIGNIFICANT, NOISE, None, {}):
        assert review(stats).critique


def test_schema_rejects_an_unknown_verdict():
    with pytest.raises(ValidationError):
        CriticReview(verdict="MAYBE", checks={}, critique=["x"])


def test_schema_requires_critique_bullets():
    with pytest.raises(ValidationError):
        CriticReview(verdict="ACCEPTED", checks={}, critique=[])


def test_review_converts_to_a_plain_dict_for_graph_state():
    data = review(SIGNIFICANT).model_dump()
    assert data["verdict"] == "ACCEPTED"
    json.dumps(data)  # must be JSON-serialisable


# ---------------------------------------------------------------------------
# Hypothesis coverage (Check 1)
# ---------------------------------------------------------------------------
def test_coverage_high_for_a_relevant_experiment():
    assert hypothesis_coverage(HYP, RELEVANT_RESULT) >= 0.3


def test_coverage_low_for_an_unrelated_experiment():
    assert hypothesis_coverage(HYP, UNRELATED_RESULT) < 0.3


def test_printing_the_hypothesis_does_not_count_as_testing_it():
    assert hypothesis_coverage(HYP, f"print({HYP!r})\n{HYP}") < 0.3


def test_week2_placeholder_simulation_is_not_a_real_test():
    """The skeleton Code Executor simulates random numbers, so the Critic must not trust it."""
    from src.agents.code_executor_agent import generate_and_run_code

    code, output = generate_and_run_code(HYP)
    assert critique_experiment(HYP, output, SIGNIFICANT, code=code).verdict == "REJECTED"


# ---------------------------------------------------------------------------
# Signal vs noise on synthetic data
# ---------------------------------------------------------------------------
def _stats_from_samples(baseline, treatment):
    """Welch t-test, 95% interval for the mean difference and Cohen's d, via scipy/numpy."""
    from scipy import stats as sp

    diff = float(treatment.mean() - baseline.mean())
    std_error = (baseline.var(ddof=1) / len(baseline) + treatment.var(ddof=1) / len(treatment)) ** 0.5
    pooled_sd = ((baseline.var(ddof=1) + treatment.var(ddof=1)) / 2) ** 0.5
    return {
        "p_value": float(sp.ttest_ind(treatment, baseline, equal_var=False).pvalue),
        "confidence_interval_95": [diff - 1.96 * float(std_error), diff + 1.96 * float(std_error)],
        "effect_size_cohens_d": diff / float(pooled_sd),
        "n_baseline": len(baseline),
        "n_treatment": len(treatment),
    }


def test_reliably_accepts_real_effects_and_rejects_random_noise():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    runs = 100
    accepted_real = accepted_noise = 0
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        baseline = rng.normal(100, 10, 40)

        real = rng.normal(112, 10, 40)    # a genuine, large improvement
        noise = rng.normal(100, 10, 40)   # same distribution: any difference is chance

        if review(_stats_from_samples(baseline, real)).verdict == "ACCEPTED":
            accepted_real += 1
        if review(_stats_from_samples(baseline, noise)).verdict == "ACCEPTED":
            accepted_noise += 1

    assert accepted_real >= 0.95 * runs   # real findings are approved
    assert accepted_noise <= 0.10 * runs  # noise is (almost always) not approved


# ---------------------------------------------------------------------------
# Prompt and LLM path
# ---------------------------------------------------------------------------
def test_prompt_contains_persona_inputs_and_checks():
    prompt = build_critic_prompt(HYP, "some output", SIGNIFICANT, code="print(1)")
    assert "Nature / NeurIPS" in prompt
    assert HYP in prompt and "some output" in prompt and "print(1)" in prompt
    assert "0.003" in prompt  # the statistics are included
    assert "sample size" in prompt and "core claim" in prompt


def test_llm_critique_text_is_included():
    llm = FakeLLM(json.dumps({"hypothesis_tested": True, "critique": ["Report memory in MB."]}))
    result = review(SIGNIFICANT, llm=llm)
    assert result.verdict == "ACCEPTED"
    assert "Report memory in MB." in result.critique
    assert llm.prompts


def test_llm_can_reject_an_experiment_that_misses_the_claim():
    llm = FakeLLM('```json\n{"hypothesis_tested": false, "critique": ["Wrong metric."]}\n```')
    assert review(SIGNIFICANT, llm=llm).verdict == "REJECTED"


def test_llm_can_not_turn_noise_into_acceptance():
    llm = FakeLLM(json.dumps({"hypothesis_tested": True, "critique": ["Looks great!"]}))
    assert review(NOISE, llm=llm).verdict == "REJECTED"


@pytest.mark.parametrize("reply", ["not json at all", '{"critique": []}', RuntimeError("API down")])
def test_bad_llm_reply_falls_back_to_offline_rules(reply):
    assert review(SIGNIFICANT, llm=FakeLLM(reply)).verdict == "ACCEPTED"


# ---------------------------------------------------------------------------
# Contract with the orchestrator's critic_node (Lead). Protects the hand-off.
# ---------------------------------------------------------------------------
def test_contract_used_by_the_orchestrator_critic_node():
    """src/orchestrator.py calls critique_experiment with these keyword names and reads these fields."""
    result = critique_experiment(
        hypothesis=HYP,
        execution_result=RELEVANT_RESULT,
        stats=SIGNIFICANT,
        code="print('pruning memory perplexity')",
    )
    assert result.verdict in {"ACCEPTED", "REJECTED", "NEEDS_REPLICATION"}
    assert isinstance(result.critique, list) and all(isinstance(line, str) for line in result.critique)
    assert result.p_value == SIGNIFICANT["p_value"]


def _critic_node_or_skip():
    orchestrator = pytest.importorskip("src.orchestrator")
    if not hasattr(orchestrator, "critic_node"):
        pytest.skip("critic_node is not in this branch yet (Lead)")
    return orchestrator.critic_node


def test_critic_node_accepts_a_real_significant_result():
    state = {
        "research_topic": "Dynamic Pruning",
        "selected_hypothesis": HYP,
        "execution_result": RELEVANT_RESULT,
        "statistical_metrics": SIGNIFICANT,
        "experiment_iteration": 0,
        "logs": [],
    }
    out = _critic_node_or_skip()(state)
    assert out["critic_verdict"] == "ACCEPTED"
    assert out["experiment_iteration"] == 1


def test_critic_node_asks_for_replication_when_no_statistics_exist():
    state = {
        "research_topic": "Dynamic Pruning",
        "selected_hypothesis": HYP,
        "execution_result": RELEVANT_RESULT,
        "experiment_iteration": 0,
        "logs": [],
    }
    out = _critic_node_or_skip()(state)
    assert out["critic_verdict"] == "NEEDS_REPLICATION"


# ---------------------------------------------------------------------------
# Check 1 must not be fooled by a script that merely embeds the claim
# ---------------------------------------------------------------------------
LONG_HYP = "Dynamic token pruning reduces inference memory.\nSecond sentence: it's tested with 'quotes' on long contexts."


def test_claim_embedded_with_repr_does_not_count_as_testing_it():
    """repr() changes newlines and quotes, so a plain text match misses the copy."""
    code = f"HYPOTHESIS = {LONG_HYP!r}\nprint(HYPOTHESIS)\nprint('Baseline mean: 100')"
    assert hypothesis_coverage(LONG_HYP, code + "\n" + LONG_HYP) < 0.3

    result = critique_experiment(LONG_HYP, LONG_HYP + "\nBaseline mean: 100", SIGNIFICANT, code=code)
    assert result.verdict == "REJECTED"
    assert result.checks.hypothesis_tested is False


def test_a_real_result_line_that_mentions_the_claim_still_counts():
    claim = "Dynamic token pruning reduces inference memory."
    real = "Measured memory with and without token pruning: memory dropped 31%."
    assert hypothesis_coverage(claim, real) >= 0.3


def test_short_claims_are_not_over_stripped():
    claim = "Pruning reduces memory."
    assert hypothesis_coverage(claim, "Pruning reduces memory by 31% in our runs.") == 1.0


# ---------------------------------------------------------------------------
# Integration with Member 4's statistics engine (skipped until src/utils/stats_validator.py exists)
# ---------------------------------------------------------------------------
def _validator():
    return pytest.importorskip("src.utils.stats_validator")


def test_real_validator_output_feeds_the_critic():
    np = pytest.importorskip("numpy")
    validator = _validator()
    rng = np.random.default_rng(1)
    baseline, treatment = rng.normal(100, 10, 40), rng.normal(112, 10, 40)

    # The group sizes are added here until the validator reports them itself (see the xfail test below).
    stats = {**validator.validate_experiment(baseline, treatment), "n_baseline": 40, "n_treatment": 40}
    result = review(stats)

    assert result.verdict == "ACCEPTED"
    assert result.p_value == stats["p_value"]


def test_critic_with_the_real_validator_accepts_real_effects_and_rejects_noise():
    np = pytest.importorskip("numpy")
    validator = _validator()
    runs = 100
    accepted_real = accepted_noise = 0
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        baseline = rng.normal(100, 10, 40)
        for kind, treatment in (("real", rng.normal(112, 10, 40)), ("noise", rng.normal(100, 10, 40))):
            stats = {**validator.validate_experiment(baseline, treatment), "n_baseline": 40, "n_treatment": 40}
            if review(stats).verdict == "ACCEPTED":
                accepted_real += kind == "real"
                accepted_noise += kind == "noise"
    assert accepted_real >= 0.95 * runs
    assert accepted_noise <= 0.10 * runs


@pytest.mark.xfail(strict=False, reason="OPEN: validate_experiment does not return n_baseline / n_treatment, "
                                        "so the Critic can not confirm the sample size and never accepts")
def test_validator_reports_the_group_sizes():
    result = _validator().validate_experiment([1, 2, 3, 4], [2, 3, 4, 6])
    assert result["n_baseline"] == 4 and result["n_treatment"] == 4


@pytest.mark.xfail(strict=False, reason="OPEN: the orchestrator's code_executor_node imports "
                                        "compute_statistics_from_output, which stats_validator does not define yet, "
                                        "so the real validator is never used in the pipeline")
def test_validator_provides_the_function_the_orchestrator_imports():
    assert callable(getattr(_validator(), "compute_statistics_from_output", None))
