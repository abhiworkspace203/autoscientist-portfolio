"""
Unit tests for Context Window Memory Manager (Member 1 / Team Lead Deliverable).
Verifies token estimation, numerical preservation, sliding window, and state compression.
"""

import pytest
from src.memory.context_manager import (
    ContextManager,
    estimate_tokens,
    extract_numerical_facts,
    compress_text,
)


def test_estimate_tokens_handles_empty_and_short_strings():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("hello") >= 1


def test_estimate_tokens_scales_with_length():
    short_tokens = estimate_tokens("Short sentence.")
    long_tokens = estimate_tokens("A " * 500)
    assert long_tokens > short_tokens
    assert long_tokens > 100


def test_extract_numerical_facts_identifies_p_values_and_deltas():
    sample_text = """
    We observed Metric delta: +3.45 across 30 trials.
    The two-tailed test yielded p-value: 0.014 with 95% CI [1.2, 5.7].
    Baseline mean: 98.2 and Method mean: 101.65.
    Cohen's d: 0.82 indicates a large effect size.
    """
    facts = extract_numerical_facts(sample_text)
    assert len(facts) >= 3
    facts_str = " ".join(facts)
    assert "0.014" in facts_str or "p-value" in facts_str.lower()
    assert "+3.45" in facts_str or "metric delta" in facts_str.lower()
    assert "98.2" in facts_str or "101.65" in facts_str


def test_compress_text_leaves_short_text_intact():
    short_text = "This is a short statement about transformers."
    assert compress_text(short_text, max_tokens=100) == short_text


def test_compress_text_truncates_large_text_and_retains_numbers():
    lengthy_text = (
        "Opening sentence about agent architecture.\n"
        "Second line describing background.\n"
        "Third line introducing hypothesis.\n"
        + "Filler research fluff discussion.\n" * 50
        + "Final findings show Metric delta: +5.20 and p-value: 0.003."
    )
    compressed = compress_text(lengthy_text, max_tokens=60)
    assert len(compressed) < len(lengthy_text)
    assert "Opening sentence" in compressed
    assert "5.20" in compressed or "+5.20" in compressed
    assert "0.003" in compressed


def test_context_manager_near_overflow_detection():
    cm = ContextManager(max_context_tokens=1000, warning_threshold=0.8)
    assert cm.is_near_overflow(700) is False
    assert cm.is_near_overflow(850) is True

    safe_state = {"research_topic": "Topic A", "logs": ["Step 1", "Step 2"]}
    assert cm.is_near_overflow(safe_state) is False

    bloated_state = {"research_topic": "Topic B", "logs": ["Step " + "A" * 200] * 20}
    assert cm.is_near_overflow(bloated_state) is True


def test_context_manager_sliding_window():
    cm = ContextManager()
    items = [f"Log entry #{i}: Details here." for i in range(50)]
    windowed = cm.sliding_window(items, max_tokens=100)
    assert len(windowed) < len(items)
    # Most recent entries should be retained
    assert "Log entry #49" in windowed[-1]


def test_context_manager_compress_state_when_exceeding_budget():
    cm = ContextManager(max_context_tokens=200, target_summary_tokens=60)
    state = {
        "research_topic": "Token Pruning",
        "literature_summary": "Extensive literature analysis.\n" + ("Very verbose abstract sentence.\n" * 40),
        "execution_result": "Execution completed. Baseline mean: 100. Method mean: 104. Metric delta: +4.0.",
        "error_traceback": "",
        "logs": [f"Log round {i}: agent discussion" for i in range(30)],
    }

    compressed = cm.compress_state(state)
    assert compressed.get("is_compressed") is True
    assert "compressed_context" in compressed
    assert "Preserved Quantitative Findings" in compressed["compressed_context"]
    assert "+4.0" in compressed["compressed_context"] or "104" in compressed["compressed_context"]
    assert len(compressed["logs"]) < len(state["logs"])


def test_context_manager_leaves_compact_state_uncompressed():
    cm = ContextManager(max_context_tokens=4000)
    state = {
        "research_topic": "Token Pruning",
        "literature_summary": "Short summary.",
        "execution_result": "Metric delta: +1.0",
        "logs": ["Log 1", "Log 2"],
    }
    result = cm.compress_state(state)
    assert result.get("is_compressed") is None


def test_format_prompt_within_budget_protects_system_and_query():
    cm = ContextManager(max_context_tokens=200)
    sys_prompt = "You are a scientific discovery assistant."
    user_query = "What is the optimal pruning threshold?"
    giant_context = "Historical research background text.\n" * 100

    prompt = cm.format_prompt_within_budget(sys_prompt, giant_context, user_query, max_prompt_tokens=150)
    assert sys_prompt in prompt
    assert user_query in prompt
    assert estimate_tokens(prompt) <= 200


def test_format_multimodal_context_with_figures():
    cm = ContextManager(max_context_tokens=300)
    text_context = "Transformer inference memory overview.\n" * 20
    figures = [
        {
            "chart_type": "Bar Chart",
            "caption": "Memory footprint across pruning thresholds",
            "trend": "Memory drops monotonically as pruning ratio increases",
            "extracted_metrics": {"Peak Memory": "14.2 GB", "Pruned Memory": "9.8 GB"},
        }
    ]
    multimodal_prompt = cm.format_multimodal_context(text_context, figures_data=figures, max_tokens=250)
    assert "Extracted Multimodal Charts & Figures" in multimodal_prompt
    assert "Figure 1: Bar Chart" in multimodal_prompt
    assert "Peak Memory: 14.2 GB" in multimodal_prompt
    assert estimate_tokens(multimodal_prompt) <= 300
