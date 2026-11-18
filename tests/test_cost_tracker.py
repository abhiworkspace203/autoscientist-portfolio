"""Tests for src/utils/cost_tracker.py (Member 5, Week 4). Offline, no API calls."""
import json
import threading
import time

import pytest

from src.utils.cost_tracker import (
    CostTracker,
    TrackedLLM,
    append_run,
    estimate_tokens,
    format_cost,
    load_runs,
    track_pipeline,
)


class FakeLLM:
    """Stands in for an LLM client; optionally reports exact token usage like a real provider."""

    provider = "fake"

    def __init__(self, reply="four", usage=None, error=None):
        self.reply, self.last_usage, self.error = reply, usage, error

    def complete(self, prompt):
        if self.error:
            raise self.error
        return self.reply


# ---------------------------------------------------------------------------
# Tokens and prices
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [("", 0), (None, 0), ("abcd", 1), ("abcde", 2), ("a" * 400, 100)])
def test_token_estimate_is_about_four_characters_per_token(text, expected):
    assert estimate_tokens(text) == expected


def test_known_model_cost_is_computed_from_the_price_table():
    cost = CostTracker().cost_for("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    assert cost == pytest.approx((1000 * 2.50 + 500 * 10.00) / 1_000_000)


def test_dated_model_names_match_by_prefix():
    tracker = CostTracker()
    assert tracker.price_for("claude-3-5-sonnet-20241022") == tracker.price_for("claude-3-5-sonnet")


def test_the_longest_matching_prefix_wins():
    tracker = CostTracker()
    assert tracker.price_for("gpt-4o-mini-2024-07-18") == tracker.price_for("gpt-4o-mini")
    assert tracker.price_for("gpt-4o-mini") != tracker.price_for("gpt-4o")


@pytest.mark.parametrize("model", ["ollama/llama3", "llama3.1:8b", "local-model", "Qwen2.5"])
def test_local_models_cost_nothing(model):
    assert CostTracker().cost_for(model, 10_000, 10_000) == 0.0


def test_unknown_model_cost_is_unknown_not_zero():
    assert CostTracker().cost_for("mystery-model-9", 1000, 1000) is None


def test_custom_prices_override_and_extend_the_defaults():
    tracker = CostTracker(pricing={"my-model": (1.0, 2.0), "gpt-4o": (5.0, 5.0)})
    assert tracker.cost_for("my-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
    assert tracker.cost_for("gpt-4o", 1_000_000, 0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Recording and totals
# ---------------------------------------------------------------------------
def test_record_stores_tokens_cost_and_latency():
    rec = CostTracker().record("critic", "gpt-4o", 1000, 500, latency_s=1.5)
    assert (rec.agent, rec.prompt_tokens, rec.completion_tokens, rec.total_tokens) == ("critic", 1000, 500, 1500)
    assert rec.cost_usd == pytest.approx(0.0075)
    assert rec.latency_s == 1.5 and rec.estimated is False


@pytest.mark.parametrize("bad", [dict(prompt_tokens=-1, completion_tokens=0), dict(prompt_tokens=0, completion_tokens=-5),
                                 dict(prompt_tokens=1, completion_tokens=1, latency_s=-0.1)])
def test_negative_numbers_are_rejected(bad):
    with pytest.raises(ValueError):
        CostTracker().record("a", "gpt-4o", **bad)


def test_text_based_records_are_marked_as_estimates():
    rec = CostTracker().record_text("coder", "gpt-4o", "a" * 40, "b" * 8)
    assert (rec.prompt_tokens, rec.completion_tokens) == (10, 2)
    assert rec.estimated is True


def test_totals_add_up_across_agents():
    tracker = CostTracker()
    tracker.record("hypothesis", "gpt-4o", 1000, 200)
    tracker.record("coder", "gpt-4o", 3000, 800)
    tracker.record("critic", "gpt-4o", 500, 100)
    summary = tracker.summary()
    assert summary["calls"] == 3
    assert summary["prompt_tokens"] == 4500 and summary["completion_tokens"] == 1100
    assert summary["total_tokens"] == 5600 == tracker.total_tokens()
    assert summary["total_cost_usd"] == pytest.approx(tracker.total_cost())
    assert summary["total_cost_usd"] == pytest.approx((4500 * 2.5 + 1100 * 10) / 1_000_000)


def test_unpriced_calls_are_counted_but_not_added_to_the_total():
    tracker = CostTracker()
    tracker.record("a", "gpt-4o", 1000, 0)
    tracker.record("b", "mystery-model", 5000, 5000)
    summary = tracker.summary()
    assert summary["unpriced_calls"] == 1
    assert summary["total_cost_usd"] == pytest.approx(0.0025)
    assert summary["total_tokens"] == 11_000  # tokens are still counted


def test_by_agent_groups_calls_and_sums_each_agent():
    tracker = CostTracker()
    tracker.record("coder", "gpt-4o", 100, 50, latency_s=1.0)
    tracker.record("coder", "gpt-4o", 200, 70, latency_s=2.0)
    tracker.record("critic", "gpt-4o", 10, 5, latency_s=0.5)
    agents = tracker.by_agent()
    assert agents["coder"]["calls"] == 2
    assert agents["coder"]["prompt_tokens"] == 300 and agents["coder"]["completion_tokens"] == 120
    assert agents["coder"]["total_tokens"] == 420
    assert agents["coder"]["latency_s"] == pytest.approx(3.0)
    assert agents["critic"]["calls"] == 1


def test_estimated_calls_are_counted_in_the_summary():
    tracker = CostTracker()
    tracker.record("a", "gpt-4o", 10, 10)
    tracker.record_text("b", "gpt-4o", "x" * 40, "y" * 40)
    assert tracker.summary()["estimated_calls"] == 1


def test_an_empty_tracker_summarises_to_zeros():
    summary = CostTracker(run_id="empty").summary()
    assert summary["calls"] == 0 and summary["total_cost_usd"] == 0 and summary["by_agent"] == {}


def test_recording_from_many_threads_loses_nothing():
    tracker = CostTracker()

    def work():
        for _ in range(50):
            tracker.record("a", "gpt-4o", 1, 1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(tracker.records) == 400


# ---------------------------------------------------------------------------
# Hooks: wrap an LLM client or a graph node without editing the agents
# ---------------------------------------------------------------------------
def test_wrapped_llm_returns_the_reply_unchanged_and_logs_the_call():
    tracker = CostTracker()
    llm = tracker.wrap_llm(FakeLLM(reply="hello world"), agent="hypothesis", model="gpt-4o")
    assert isinstance(llm, TrackedLLM)
    assert llm.complete("Say hello please") == "hello world"
    rec = tracker.records[0]
    assert rec.agent == "hypothesis" and rec.estimated is True
    assert rec.prompt_tokens == estimate_tokens("Say hello please")
    assert rec.completion_tokens == estimate_tokens("hello world")


def test_wrapped_llm_uses_exact_provider_usage_when_available():
    tracker = CostTracker()
    llm = tracker.wrap_llm(FakeLLM(usage={"prompt_tokens": 123, "completion_tokens": 45}), "coder", "gpt-4o")
    llm.complete("anything")
    rec = tracker.records[0]
    assert (rec.prompt_tokens, rec.completion_tokens, rec.estimated) == (123, 45, False)


def test_wrapped_llm_logs_a_failed_call_and_re_raises():
    tracker = CostTracker()
    llm = tracker.wrap_llm(FakeLLM(error=RuntimeError("API down")), "critic", "gpt-4o")
    with pytest.raises(RuntimeError, match="API down"):
        llm.complete("prompt text")
    assert len(tracker.records) == 1
    assert tracker.records[0].completion_tokens == 0


def test_wrapped_llm_still_exposes_the_original_attributes():
    llm = CostTracker().wrap_llm(FakeLLM(), "a", "gpt-4o")
    assert llm.provider == "fake"


def test_wrapped_node_returns_the_same_result_and_keeps_its_name():
    tracker = CostTracker()

    def my_node(state):
        return {"done": state["x"] + 1}

    wrapped = tracker.wrap_node("code_executor", my_node)
    assert wrapped({"x": 1}) == {"done": 2}
    assert wrapped.__name__ == "my_node"


def test_wrapped_node_times_every_call_even_when_it_fails():
    tracker = CostTracker()

    def slow(state):
        time.sleep(0.02)
        return {}

    def broken(state):
        time.sleep(0.01)
        raise ValueError("boom")

    tracker.wrap_node("slow", slow)({})
    with pytest.raises(ValueError):
        tracker.wrap_node("broken", broken)({})
    assert tracker.node_latency_s["slow"] >= 0.02
    assert tracker.node_latency_s["broken"] >= 0.01
    assert tracker.by_agent()["slow"]["node_latency_s"] >= 0.02


def test_experiment_context_manager_records_latency_even_on_error():
    tracker = CostTracker()
    with tracker.experiment("exp-1"):
        time.sleep(0.02)
    with pytest.raises(RuntimeError):
        with tracker.experiment("exp-2"):
            raise RuntimeError("crash")
    names = [e["name"] for e in tracker.experiments]
    assert names == ["exp-1", "exp-2"]
    assert tracker.experiments[0]["latency_s"] >= 0.02


# ---------------------------------------------------------------------------
# Saving and the multi-run log the dashboard reads
# ---------------------------------------------------------------------------
def test_save_writes_a_readable_json_file(tmp_path):
    tracker = CostTracker(run_id="r1")
    tracker.record("coder", "gpt-4o", 100, 50)
    path = tracker.save(tmp_path / "nested" / "run.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["runs"][0]["run_id"] == "r1"
    assert data["runs"][0]["records"][0]["agent"] == "coder"


def test_append_run_keeps_earlier_runs(tmp_path):
    log = tmp_path / "cost_log.json"
    for run_id in ("a", "b", "c"):
        tracker = CostTracker(run_id=run_id)
        tracker.record("x", "gpt-4o", 10, 10)
        append_run(tracker, log)
    assert [r["run_id"] for r in load_runs(log)] == ["a", "b", "c"]


def test_saving_the_same_run_again_replaces_it_instead_of_duplicating(tmp_path):
    log = tmp_path / "cost_log.json"
    tracker = CostTracker(run_id="same")
    tracker.record("x", "gpt-4o", 10, 10)
    append_run(tracker, log)
    tracker.record("x", "gpt-4o", 10, 10)
    append_run(tracker, log)
    runs = load_runs(log)
    assert len(runs) == 1 and runs[0]["calls"] == 2


@pytest.mark.parametrize("content", [None, "", "not json", "[1, 2]", '{"runs": "oops"}', '{"runs": [1, "x"]}'])
def test_missing_or_broken_log_files_load_as_empty(tmp_path, content):
    path = tmp_path / "log.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert load_runs(path) == []


def test_round_trip_keeps_the_numbers(tmp_path):
    tracker = CostTracker(run_id="rt")
    tracker.record("a", "gpt-4o", 1000, 500, latency_s=1.25)
    tracker.record_text("b", "mystery", "p" * 40, "c" * 40)
    run = load_runs(append_run(tracker, tmp_path / "log.json"))[0]
    assert run["total_cost_usd"] == pytest.approx(tracker.total_cost())
    assert run["unpriced_calls"] == 1 and run["estimated_calls"] == 1
    assert run["by_agent"]["a"]["prompt_tokens"] == 1000


@pytest.mark.parametrize("value, text", [(None, "unknown"), (0.0, "$0.0000"), (0.0075, "$0.0075"), (12.5, "$12.50")])
def test_cost_formatting(value, text):
    assert format_cost(value) == text


# ---------------------------------------------------------------------------
# track_pipeline: one-line hook for the whole orchestrator
# ---------------------------------------------------------------------------
def fake_orchestrator():
    import types

    module = types.SimpleNamespace()
    module.literature_reviewer_node = lambda state: {"a": 1}
    module.code_executor_node = lambda state: {"b": 2}
    module.critic_node = lambda state: {"c": 3}
    return module


def test_track_pipeline_times_the_nodes_inside_the_block():
    tracker, module = CostTracker(), fake_orchestrator()
    with track_pipeline(tracker, module):
        assert module.code_executor_node({}) == {"b": 2}
        module.critic_node({})
    assert set(tracker.node_latency_s) == {"code_executor", "critic"}


def test_track_pipeline_restores_the_original_functions_afterwards():
    tracker, module = CostTracker(), fake_orchestrator()
    originals = (module.literature_reviewer_node, module.code_executor_node, module.critic_node)
    with track_pipeline(tracker, module):
        assert module.critic_node is not originals[2]
    assert (module.literature_reviewer_node, module.code_executor_node, module.critic_node) == originals


def test_track_pipeline_restores_even_when_the_block_fails():
    tracker, module = CostTracker(), fake_orchestrator()
    original = module.critic_node
    with pytest.raises(RuntimeError):
        with track_pipeline(tracker, module):
            raise RuntimeError("boom")
    assert module.critic_node is original


def test_track_pipeline_skips_nodes_the_module_does_not_have():
    tracker, module = CostTracker(), fake_orchestrator()  # no debugger_node, no hypothesis_generator_node
    with track_pipeline(tracker, module):
        assert not hasattr(module, "debugger_node")


def test_track_pipeline_accepts_a_custom_node_map():
    tracker, module = CostTracker(), fake_orchestrator()
    with track_pipeline(tracker, module, nodes={"critic_node": "reviewer"}):
        module.critic_node({})
        module.code_executor_node({})  # not in the map: not timed
    assert set(tracker.node_latency_s) == {"reviewer"}
