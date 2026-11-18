"""Tests for the offline proxy experiment in code_executor_agent (real, measured results)."""
import json
import re

from src.agents.code_executor_agent import PROXY_DATASETS, _proxy_settings, generate_code, run_code
from src.agents.critic_agent import critique_experiment
from src.utils.stats_validator import compute_statistics_from_output

HYPOTHESIS = (
    "Hypothesis: A model variant designed to address the issue below achieves a higher mean "
    "cross-validated macro-F1 than an otherwise identical baseline on a controlled proxy task. "
    "Issue: Literature Review: RAG Papers retrieved: 5"
)


def run_experiment(hypothesis=HYPOTHESIS):
    code = generate_code(hypothesis)
    return code, run_code(code, timeout=120)


def test_settings_are_stable_and_valid():
    dataset, seed = _proxy_settings(HYPOTHESIS)
    assert dataset in PROXY_DATASETS
    assert _proxy_settings(HYPOTHESIS) == (dataset, seed)
    assert 0 <= seed < 10_000


def test_generated_code_has_no_placeholders():
    code = generate_code(HYPOTHESIS)
    assert "__HYPOTHESIS__" not in code and "__DATASET__" not in code and "__SEED__" not in code
    assert "random.gauss" not in code


def test_experiment_prints_measured_scores():
    _, output = run_experiment()
    line = next(l for l in output.splitlines() if l.startswith("RESULTS_JSON: "))
    data = json.loads(line[len("RESULTS_JSON: "):])
    assert len(data["baseline"]) == len(data["treatment"]) == 40
    assert all(0.0 <= s <= 1.0 for s in data["baseline"] + data["treatment"])
    for label in ("Baseline mean:", "Method mean:", "Metric delta:"):
        assert re.search(rf"^{label}\s*[-\d.]+", output, re.MULTILINE)


def test_stats_validator_reads_the_real_scores():
    _, output = run_experiment()
    stats = compute_statistics_from_output(output)
    assert stats["n_baseline"] == stats["n_treatment"] == 40
    assert 0.0 <= stats["p_value"] <= 1.0


def test_critic_judges_the_measured_result():
    code, output = run_experiment()
    stats = compute_statistics_from_output(output)
    review = critique_experiment(HYPOTHESIS, output, stats, code=code)
    assert review.verdict in {"ACCEPTED", "REJECTED", "NEEDS_REPLICATION"}
    assert review.checks.sample_size_adequate is True
    assert review.checks.significant == (stats["p_value"] < 0.05)
