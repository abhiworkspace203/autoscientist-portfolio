import streamlit as st


AGENTS = [
    "Literature Reviewer",
    "Hypothesis Generator",
    "Code Executor",
    "Debugger Node",
    "Critic Node",
    "LaTeX Synthesizer",
]

def render_agent_trace(active_agent: str | None = None) -> None:
    """Display the agent workflow and highlight the active agent."""
    for agent in AGENTS:
        if agent == active_agent:
            st.success(f"▶ {agent} — active")
        else:
            st.write(f"○ {agent}")

        if agent != AGENTS[-1]:
            st.caption("↓")