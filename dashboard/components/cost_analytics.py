"""
Cost analytics widget for the Streamlit dashboard
Member 5 Deliverable (Week 4)

Two Plotly charts built from the runs a CostTracker saved:
  1. Cumulative cost ($) per discovery run
  2. Token distribution by agent (prompt vs completion tokens)

Use it inside any dashboard page:

    from dashboard.components.cost_analytics import render_cost_analytics
    render_cost_analytics()                      # reads output/cost_log.json
    render_cost_analytics(tracker=my_tracker)    # or show a live run

The figure builders (cumulative_cost_figure, token_distribution_figure) are plain functions that
return Plotly figures, so they can be tested and reused without Streamlit.
To preview the widget on its own with DEMO data:
    streamlit run dashboard/components/cost_analytics.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.utils.cost_tracker import DEFAULT_LOG_PATH, CostTracker, format_cost, load_runs
except ImportError:  # started with `streamlit run dashboard/app.py`: the repo root is not on sys.path yet
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.cost_tracker import DEFAULT_LOG_PATH, CostTracker, format_cost, load_runs


def _go():
    import plotly.graph_objects as go  # imported here so the project works without Plotly installed

    return go


def run_label(run: Dict[str, Any]) -> str:
    return run.get("label") or run.get("run_id") or "run"


# ---------------------------------------------------------------------------
# Chart 1: cumulative cost per run
# ---------------------------------------------------------------------------
COLORS = ["#818cf8", "#10b981", "#f59e0b", "#ec4899", "#06b6d4"]

_DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    title_font=dict(family="Inter, sans-serif", color="#c7d2fe", size=15),
    xaxis=dict(gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.2)", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.2)", tickfont=dict(color="#64748b")),
    legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
    margin=dict(t=50, l=60, r=30, b=50),
)


def cumulative_cost_figure(runs: List[Dict[str, Any]]):
    """One line per run: the running total cost after each LLM call (call 0 = $0)."""
    go = _go()
    fig = go.Figure()
    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        records = run.get("records", [])
        running, xs, ys, notes = 0.0, [0], [0.0], ["start"]
        for number, rec in enumerate(records, start=1):
            cost = rec.get("cost_usd")
            running += cost or 0.0
            xs.append(number)
            ys.append(running)
            notes.append(f"{rec.get('agent', '?')} ({rec.get('model', '?')})" + ("" if cost is not None else ", price unknown"))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=run_label(run), text=notes,
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=7, line=dict(color="#0d1117", width=1.5)),
            hovertemplate="call %{x}: %{text}<br>cumulative $%{y:.4f}<extra></extra>",
        ))
    layout = dict(**_DARK_LAYOUT)
    layout["title"] = "Cumulative Cost per Discovery Run"
    layout["xaxis"]["title"] = "LLM Call Number"
    layout["yaxis"]["title"] = "Cumulative Cost (USD)"
    layout["yaxis"]["tickformat"] = "$.4f"
    layout["legend_title"] = "Run"
    fig.update_layout(**layout)
    if not runs:
        fig.add_annotation(text="No runs recorded yet", showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color="#475569", size=14))
    return fig


# ---------------------------------------------------------------------------
# Chart 2: token distribution by agent
# ---------------------------------------------------------------------------
def token_distribution_figure(run: Optional[Dict[str, Any]]):
    """Stacked bars: prompt and completion tokens for each agent, largest first."""
    go = _go()
    agents = {name: a for name, a in (run or {}).get("by_agent", {}).items() if a.get("total_tokens", 0) > 0}
    names = sorted(agents, key=lambda n: agents[n]["total_tokens"], reverse=True)
    fig = go.Figure()
    fig.add_bar(name="Prompt tokens", x=names, y=[agents[n]["prompt_tokens"] for n in names],
                marker_color="#818cf8", marker_line=dict(color="#0d1117", width=1))
    fig.add_bar(name="Completion tokens", x=names, y=[agents[n]["completion_tokens"] for n in names],
                marker_color="#10b981", marker_line=dict(color="#0d1117", width=1))
    title = "Token Distribution by Agent"
    if run and run.get("estimated_calls"):
        title += " (includes estimated counts)"
    layout = dict(**_DARK_LAYOUT)
    layout["barmode"] = "stack"
    layout["title"] = title
    layout["xaxis"]["title"] = "Agent"
    layout["yaxis"]["title"] = "Tokens"
    fig.update_layout(**layout)
    if not names:
        fig.add_annotation(text="No token usage recorded", showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color="#475569", size=14))
    return fig


# ---------------------------------------------------------------------------
# Chart 3: time spent in each agent node
# ---------------------------------------------------------------------------
def node_latency_figure(run: Optional[Dict[str, Any]]):
    """Bars: seconds spent in each graph node (reported even when no LLM was called)."""
    go = _go()
    nodes = {name: a["node_latency_s"] for name, a in (run or {}).get("by_agent", {}).items()
             if a.get("node_latency_s", 0) > 0}
    names = sorted(nodes, key=lambda n: nodes[n], reverse=True)
    # gradient colors by rank
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    fig = go.Figure(go.Bar(
        x=names, y=[round(nodes[n], 4) for n in names], name="Node time",
        marker=dict(color=bar_colors, line=dict(color="#0d1117", width=1)),
    ))
    layout = dict(**_DARK_LAYOUT)
    layout["title"] = "Time per Agent Node"
    layout["xaxis"]["title"] = "Agent Node"
    layout["yaxis"]["title"] = "Seconds"
    fig.update_layout(**layout)
    if not names:
        fig.add_annotation(text="No node timings recorded", showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color="#475569", size=14))
    return fig


# ---------------------------------------------------------------------------
# The Streamlit widget
# ---------------------------------------------------------------------------
def agent_table(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-agent rows for a table."""
    rows = []
    for name, a in sorted(run.get("by_agent", {}).items(), key=lambda item: item[1].get("total_tokens", 0), reverse=True):
        rows.append({
            "Agent": name,
            "Calls": a.get("calls", 0),
            "Prompt tokens": a.get("prompt_tokens", 0),
            "Completion tokens": a.get("completion_tokens", 0),
            "Cost": format_cost(a.get("cost_usd", 0.0)),
            "LLM time (s)": round(a.get("latency_s", 0.0), 2),
        })
    return rows


def render_cost_analytics(
    st: Any = None,
    runs: Optional[List[Dict[str, Any]]] = None,
    log_path: str = DEFAULT_LOG_PATH,
    tracker: Optional[CostTracker] = None,
    key_prefix: str = "cost",
) -> None:
    """Draw the cost widget. Data comes from `tracker` (live run), `runs`, or the log file, in that order."""
    if st is None:
        import streamlit as st  # noqa: F811

    if tracker is not None:
        runs = [tracker.to_dict()]
    elif runs is None:
        runs = load_runs(log_path)

    st.subheader("Execution cost analytics")
    if not runs:
        st.info("No cost data yet. Run a discovery with the cost tracker enabled; runs are saved to "
                f"`{log_path}`.")
        return

    labels = [run_label(r) for r in runs]
    chosen = st.selectbox("Discovery run", labels, index=len(labels) - 1, key=f"{key_prefix}_run")
    run = runs[labels.index(chosen)]

    columns = st.columns(4)
    columns[0].metric("Total cost", format_cost(run.get("total_cost_usd", 0.0)))
    columns[1].metric("Total tokens", f"{run.get('total_tokens', 0):,}")
    columns[2].metric("LLM calls", run.get("calls", 0))
    columns[3].metric("LLM time", f"{run.get('llm_latency_s', 0.0):.1f} s")

    if not run.get("calls"):
        st.caption("No LLM calls were recorded for this run (the offline rule-based agents do not call an LLM), "
                   "so cost and tokens are zero. Node timings are shown below.")
    if run.get("estimated_calls"):
        st.caption(f"{run['estimated_calls']} of {run.get('calls', 0)} calls use estimated token counts "
                   "(about 4 characters per token), not provider-reported numbers.")
    if run.get("unpriced_calls"):
        st.warning(f"{run['unpriced_calls']} call(s) used a model with no known price. Their cost is NOT included in the total.")
    for exp in run.get("experiments", []):
        st.caption(f"Experiment `{exp.get('name')}`: {exp.get('latency_s', 0.0):.2f} s")

    st.plotly_chart(cumulative_cost_figure(runs), key=f"{key_prefix}_cumulative")
    st.plotly_chart(token_distribution_figure(run), key=f"{key_prefix}_tokens")
    st.plotly_chart(node_latency_figure(run), key=f"{key_prefix}_latency")
    rows = agent_table(run)
    if rows:
        st.dataframe(rows)


def demo_runs() -> List[Dict[str, Any]]:
    """Two small made-up runs, ONLY for previewing the widget (always labelled as demo data)."""
    runs = []
    for number, scale in ((1, 1), (2, 2)):
        tracker = CostTracker(run_id=f"demo-{number}", label=f"DEMO run {number}")
        tracker.record("hypothesis_generator", "gpt-4o", 1800 * scale, 400 * scale, latency_s=2.1)
        tracker.record("code_executor", "gpt-4o", 2500 * scale, 900 * scale, latency_s=3.4)
        tracker.record("debugger", "gpt-4o", 1200 * scale, 300 * scale, latency_s=1.8, estimated=True)
        tracker.record("critic", "gpt-4o", 900 * scale, 350 * scale, latency_s=1.2)
        runs.append(tracker.to_dict())
    return runs


if __name__ == "__main__":  # streamlit run dashboard/components/cost_analytics.py
    import streamlit as _st

    _st.caption("Preview with DEMO data (not a real run).")
    render_cost_analytics(_st, runs=demo_runs())
