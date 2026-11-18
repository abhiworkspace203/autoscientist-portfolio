"""
Tests for Sub-Agents:
1. Literature Reviewer Agent
2. Hypothesis Generator Agent
3. Code Executor Agent (Template verification)
"""

import pytest
from src.agents import literature_agent as lit
from src.agents import hypothesis_agent as hyp
from src.agents import code_executor_agent as cea


# --- Literature Reviewer Tests ---
def test_literature_review_returns_summary():
    summary = lit.run_literature_review("Autonomous Agent Reasoning")
    assert isinstance(summary, str)
    assert len(summary) > 20
    assert "Literature Review" in summary


def test_literature_filter_by_topic():
    sample_papers = [
        {"title": "Paper on AI agents", "abstract": "Reasoning models", "categories": ["cs.AI"]},
        {"title": "Quantum Physics", "abstract": "Entangled photons", "categories": ["physics"]},
    ]
    filtered = lit.filter_papers_by_topic(sample_papers, "AI agents")
    assert len(filtered) >= 1
    assert "AI" in filtered[0]["title"]


# --- Hypothesis Generator Tests ---
def test_hypothesis_generation():
    summary = "Current transformer architectures suffer from quadratic attention memory bottlenecks."
    h = hyp.generate_hypothesis(summary)
    assert isinstance(h, str)
    assert h.startswith("Hypothesis:")


def test_hypothesis_candidate_list():
    summary = "Reinforcement learning in self debugging agent loops."
    candidates = hyp.generate_all_hypotheses(summary)
    assert len(candidates) >= 1
    assert all("Hypothesis" in c for c in candidates)


# --- Code Executor Template Tests ---
def test_code_executor_template_returns_metric_delta():
    test_h = "Dynamic token pruning reduces memory footprint by 30%."
    code, output = cea.generate_and_run_code(test_h)
    assert isinstance(code, str) and len(code) > 0
    assert isinstance(output, str) and len(output) > 0
    assert "Metric delta:" in output
