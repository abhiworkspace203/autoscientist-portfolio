"""
Tests for dashboard/components/cost_analytics.py (Member 5, Week 4).

The chart builders run with Plotly only. The widget itself is run headless with Streamlit's
AppTest. Each part skips if its library is not installed.
"""
import pytest

pytest.importorskip("plotly")

from dashboard.components.cost_analytics import (  # noqa: E402
    agent_table,
    cumulative_cost_figure,
    demo_runs,
    node_latency_figure,
    token_distribution_figure,
)
from src.utils.cost_tracker import CostTracker, append_run  # noqa: E402


def make_run(run_id="r1", label=""):
    tracker = CostTracker(run_id=run_id, label=label)
    tracker.record("hypothesis_generator", "gpt-4o", 1000, 200, latency_s=1.0)
    tracker.record("code_executor", "gpt-4o", 3000, 900, latency_s=2.0)
    tracker.record("critic", "gpt-4o", 500, 100, latency_s=0.5)
    return tracker


# ---------------------------------------------------------------------------
# Chart 1: cumulative cost per run
# ---------------------------------------------------------------------------
def test_cumulative_chart_has_one_line_per_run():
    runs = [make_run("a").to_dict(), make_run("b").to_dict()]
    fig = cumulative_cost_figure(runs)
    assert [trace.name for trace in fig.data] == ["a", "b"]


def test_cumulative_line_starts_at_zero_and_ends_at_the_run_total():
    tracker = make_run()
    trace = cumulative_cost_figure([tracker.to_dict()]).data[0]
    assert list(trace.x) == [0, 1, 2, 3]
    assert trace.y[0] == 0.0
    assert trace.y[-1] == pytest.approx(tracker.total_cost())


def test_cumulative_line_never_goes_down():
    ys = list(cumulative_cost_figure([make_run().to_dict()]).data[0].y)
    assert ys == sorted(ys)


def test_unpriced_call_does_not_raise_the_line_and_is_noted_on_hover():
    tracker = CostTracker(run_id="u")
    tracker.record("a", "gpt-4o", 1000, 0)
    tracker.record("b", "mystery-model", 9999, 9999)
    trace = cumulative_cost_figure([tracker.to_dict()]).data[0]
    assert trace.y[1] == trace.y[2]
    assert "price unknown" in trace.text[2]


def test_run_label_is_used_as_the_line_name():
    fig = cumulative_cost_figure([make_run("id-1", label="Sparse Transformers").to_dict()])
    assert fig.data[0].name == "Sparse Transformers"


def test_cumulative_chart_with_no_runs_does_not_crash():
    fig = cumulative_cost_figure([])
    assert len(fig.data) == 0
    assert "No runs" in fig.layout.annotations[0].text


# ---------------------------------------------------------------------------
# Chart 2: token distribution by agent
# ---------------------------------------------------------------------------
def test_token_chart_has_prompt_and_completion_bars_per_agent():
    fig = token_distribution_figure(make_run().to_dict())
    assert [trace.name for trace in fig.data] == ["Prompt tokens", "Completion tokens"]
    assert fig.layout.barmode == "stack"


def test_token_chart_values_match_the_run_and_largest_agent_comes_first():
    fig = token_distribution_figure(make_run().to_dict())
    agents = list(fig.data[0].x)
    assert agents == ["code_executor", "hypothesis_generator", "critic"]
    assert dict(zip(agents, fig.data[0].y)) == {"code_executor": 3000, "hypothesis_generator": 1000, "critic": 500}
    assert dict(zip(agents, fig.data[1].y)) == {"code_executor": 900, "hypothesis_generator": 200, "critic": 100}


def test_token_chart_says_when_counts_are_estimated():
    tracker = CostTracker()
    tracker.record_text("a", "gpt-4o", "x" * 400, "y" * 400)
    assert "estimated" in token_distribution_figure(tracker.to_dict()).layout.title.text
    assert "estimated" not in token_distribution_figure(make_run().to_dict()).layout.title.text


@pytest.mark.parametrize("run", [None, {}, {"by_agent": {}}])
def test_token_chart_with_no_usage_does_not_crash(run):
    fig = token_distribution_figure(run)
    assert "No token usage" in fig.layout.annotations[0].text


def test_agent_table_rows_are_sorted_and_formatted():
    rows = agent_table(make_run().to_dict())
    assert [r["Agent"] for r in rows] == ["code_executor", "hypothesis_generator", "critic"]
    assert rows[0]["Prompt tokens"] == 3000 and rows[0]["Cost"].startswith("$")


def test_demo_data_is_labelled_as_demo():
    assert all("DEMO" in run["label"] for run in demo_runs())


# ---------------------------------------------------------------------------
# The Streamlit widget, run headless
# ---------------------------------------------------------------------------
def run_widget(body):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    script = "from dashboard.components.cost_analytics import render_cost_analytics\n" + body
    app = AppTest.from_string(script, default_timeout=30)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_widget_shows_metrics_and_all_charts(tmp_path):
    log = tmp_path / "log.json"
    append_run(make_run("run-a"), log)
    app = run_widget(f"render_cost_analytics(log_path={str(log)!r})")
    labels = [m.label for m in app.metric]
    assert labels == ["Total cost", "Total tokens", "LLM calls", "LLM time"]
    assert [m.value for m in app.metric][1:3] == ["5,700", "3"]
    assert len(app.get("plotly_chart")) == 3


def test_widget_without_data_shows_a_friendly_message(tmp_path):
    app = run_widget(f"render_cost_analytics(log_path={str(tmp_path / 'missing.json')!r})")
    assert len(app.info) == 1 and "No cost data yet" in app.info[0].value
    assert len(app.metric) == 0


def test_widget_warns_about_unpriced_models_and_estimates(tmp_path):
    tracker = CostTracker(run_id="w")
    tracker.record("a", "gpt-4o", 1000, 100)
    tracker.record_text("b", "mystery-model", "p" * 400, "c" * 400)
    log = tmp_path / "log.json"
    append_run(tracker, log)
    app = run_widget(f"render_cost_analytics(log_path={str(log)!r})")
    assert any("no known price" in w.value for w in app.warning)
    assert any("estimated token counts" in c.value for c in app.caption)


def test_widget_lets_you_pick_between_runs(tmp_path):
    log = tmp_path / "log.json"
    append_run(make_run("first"), log)
    append_run(make_run("second"), log)
    app = run_widget(f"render_cost_analytics(log_path={str(log)!r})")
    assert app.selectbox[0].options == ["first", "second"]
    assert app.selectbox[0].value == "second"  # the latest run is shown by default


def test_widget_can_show_a_live_tracker():
    app = run_widget(
        "from src.utils.cost_tracker import CostTracker\n"
        "t = CostTracker(run_id='live')\n"
        "t.record('critic', 'gpt-4o', 100, 50)\n"
        "render_cost_analytics(tracker=t)"
    )
    assert app.metric[2].value == "1"


# ---------------------------------------------------------------------------
# Chart 3: time per agent node (works even when no LLM was called)
# ---------------------------------------------------------------------------
def test_node_latency_chart_shows_each_timed_node_slowest_first():
    tracker = CostTracker()
    tracker.add_node_latency("critic", 0.5)
    tracker.add_node_latency("code_executor", 2.0)
    tracker.add_node_latency("literature_reviewer", 1.0)
    fig = node_latency_figure(tracker.to_dict())
    assert list(fig.data[0].x) == ["code_executor", "literature_reviewer", "critic"]
    assert list(fig.data[0].y) == [2.0, 1.0, 0.5]


@pytest.mark.parametrize("run", [None, {}, {"by_agent": {"a": {"calls": 1}}}])
def test_node_latency_chart_with_no_timings_does_not_crash(run):
    fig = node_latency_figure(run)
    assert "No node timings" in fig.layout.annotations[0].text


def test_widget_says_so_when_a_run_made_no_llm_calls(tmp_path):
    tracker = CostTracker(run_id="offline")
    tracker.add_node_latency("code_executor", 1.5)
    log = tmp_path / "log.json"
    append_run(tracker, log)
    app = run_widget(f"render_cost_analytics(log_path={str(log)!r})")
    assert any("No LLM calls were recorded" in c.value for c in app.caption)
    assert len(app.get("plotly_chart")) == 3
