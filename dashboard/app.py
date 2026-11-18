from pathlib import Path
import streamlit as st
from components.agent_trace import AGENTS, render_agent_trace
from components.paper_viewer import render_paper_viewer

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoScientist AI — Discovery Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
.main { background: #0a0e1a; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1224 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.9rem;
    font-weight: 500;
    padding: 8px 12px;
    border-radius: 8px;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #c7d2fe !important;
    background: rgba(99,102,241,0.1);
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 40%, #0a0e1a 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(16,185,129,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c7d2fe 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(16,185,129,0.2));
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: inline-block;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #111827, #1a2234);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(99,102,241,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    color: #64748b;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* ── Section header ── */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.25rem;
}
.section-sub {
    color: #475569;
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
}

/* ── Agent pipeline ── */
.pipeline-container {
    display: flex;
    align-items: center;
    gap: 0;
    overflow-x: auto;
    padding: 1rem 0;
}
.agent-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    min-width: 120px;
}
.agent-circle {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: #1a2234;
    border: 2px solid rgba(99,102,241,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    transition: all 0.3s;
}
.agent-circle.active {
    background: linear-gradient(135deg, #4f46e5, #10b981);
    border-color: #818cf8;
    box-shadow: 0 0 24px rgba(99,102,241,0.5);
    animation: pulse 2s infinite;
}
.agent-circle.done {
    background: rgba(16,185,129,0.15);
    border-color: #10b981;
}
.agent-name {
    font-size: 0.7rem;
    color: #64748b;
    font-weight: 500;
    text-align: center;
}
.agent-name.active { color: #a5b4fc; font-weight: 700; }
.pipeline-arrow {
    color: rgba(99,102,241,0.4);
    font-size: 1.5rem;
    padding: 0 4px;
    flex-shrink: 0;
}

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(17, 24, 39, 0.8);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-success { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-warning { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-info    { background: rgba(99,102,241,0.15);  color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }

/* ── Input override ── */
.stTextInput input, .stSelectbox select {
    background: #111827 !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Button override ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca, #4f46e5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.4) !important;
}

/* ── Code block override ── */
.stCode, code {
    background: #0d1117 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #a5b4fc !important;
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    margin: 1.5rem 0;
}

/* ── Animations ── */
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 24px rgba(99,102,241,0.5); }
    50% { box-shadow: 0 0 40px rgba(99,102,241,0.8); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fadeIn 0.4s ease forwards; }

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827, #1a2234);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: #a5b4fc !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }

/* ── Info / warning override ── */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    background: rgba(17,24,39,0.8) !important;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    padding: 1rem 0 1.5rem;
    text-align: center;
}
.sidebar-logo-title {
    font-size: 1.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-logo-sub {
    font-size: 0.7rem;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}
.nav-item {
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 4px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.875rem;
    font-weight: 500;
    color: #64748b;
    transition: all 0.2s;
}
.nav-item.active {
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    border-left: 3px solid #6366f1;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.5rem">🔬</div>
        <div class="sidebar-logo-title">AutoScientist AI</div>
        <div class="sidebar-logo-sub">Discovery Dashboard</div>
    </div>
    <div class="fancy-divider"></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🧪  Discovery Studio", "🌳  Agent Pipeline", "📚  Literature & Citations",
         "📄  Paper Viewer", "💰  Cost Analytics", "📊  Holdout Benchmark"],
        label_visibility="collapsed",
    )
    page = page.split("  ", 1)[1]  # strip emoji prefix

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0 8px;">
        <div style="font-size:0.7rem; color:#374151; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">System Status</div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#10b981;box-shadow:0 0 6px #10b981;"></div>
            <span style="font-size:0.8rem;color:#64748b;">Pipeline Ready</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#10b981;box-shadow:0 0 6px #10b981;"></div>
            <span style="font-size:0.8rem;color:#64748b;">Vector DB Online</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#f59e0b;box-shadow:0 0 6px #f59e0b;"></div>
            <span style="font-size:0.8rem;color:#64748b;">LaTeX (stub mode)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Hero Banner (shown on all pages) ───────────────────────────────────────
PAGE_META = {
    "Discovery Studio":        ("🧪", "Discovery Studio",       "Configure and launch your autonomous scientific research pipeline"),
    "Agent Pipeline":          ("🌳", "Agent Pipeline",          "Track real-time agent progress through the discovery workflow"),
    "Literature & Citations":  ("📚", "Literature & Citations",  "Search and explore papers indexed by the retrieval agent"),
    "Paper Viewer":            ("📄", "Paper Viewer",            "Preview and download the AI-synthesized research paper"),
    "Cost Analytics":          ("💰", "Cost Analytics",          "LLM cost breakdown, token usage, and node latency metrics"),
    "Holdout Benchmark":       ("📊", "Holdout Benchmark",       "Agent vs. human scoring across 9 curated research papers"),
}
icon, title, subtitle = PAGE_META[page]
st.markdown(f"""
<div class="hero-banner animate-in">
    <div class="hero-badge">AutoScientist AI · Week 4</div>
    <h1 class="hero-title">{icon}&nbsp; {title}</h1>
    <p class="hero-subtitle">{subtitle}</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Discovery Studio
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Discovery Studio":

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<p class="section-header">Launch a Research Run</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Enter a topic and let the agent autonomously discover, hypothesize, and validate.</p>', unsafe_allow_html=True)

        topic = st.text_input("🔭 Research Topic", placeholder="e.g. Sparse Transformers, CRISPR in cancer therapy…",
                              label_visibility="collapsed")

        c1, c2 = st.columns(2)
        with c1:
            enable_critic = st.toggle("🎓 Enable Statistical Critic", value=True)
        with c2:
            synth_latex = st.toggle("📝 Synthesize LaTeX Paper", value=False)

        if "run_state" not in st.session_state:
            st.session_state["run_state"] = {}
            st.session_state["active_agent"] = None

        if st.button("🚀 Start Discovery", use_container_width=True):
            if not topic.strip():
                st.warning("⚠️ Please enter a research topic first.")
            else:
                try:
                    from src.orchestrator import build_orchestrator
                    graph = build_orchestrator(enable_critic=enable_critic)
                    run_state = {}

                    with st.status("🔬 Discovery pipeline running…", expanded=True) as status:
                        for event in graph.stream(
                            {"research_topic": topic.strip(), "enable_critic": enable_critic},
                            stream_mode="updates",
                        ):
                            for node_name, update in event.items():
                                run_state.update(update)
                                agent = node_name.replace("_", " ").title()
                                st.session_state["active_agent"] = agent
                                status.write(f"**▶ {agent}** running…")
                                for log in update.get("logs", []):
                                    status.write(f"  `{log}`")
                                st.session_state["run_state"] = run_state.copy()
                        status.update(label="✅ Discovery complete!", state="complete")
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")

    with col_right:
        st.markdown('<p class="section-header">Run Status</p>', unsafe_allow_html=True)
        run = st.session_state["run_state"]
        active = st.session_state.get("active_agent") or "Waiting…"
        status_val = run.get("execution_status", "Not started")
        retries = run.get("retry_count", 0)
        verdict = run.get("critic_verdict", "—")

        st.markdown(f"""
        <div class="glass-card">
            <div style="margin-bottom:1rem;">
                <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Active Agent</div>
                <div style="font-size:1.1rem;font-weight:600;color:#a5b4fc;margin-top:4px;">{active}</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                <div>
                    <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Status</div>
                    <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;margin-top:4px;">{status_val}</div>
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Retries</div>
                    <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;margin-top:4px;">{retries}</div>
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Critic Verdict</div>
                    <div style="font-size:0.95rem;font-weight:600;color:{'#10b981' if verdict=='ACCEPTED' else '#f59e0b'};margin-top:4px;">{verdict}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if run.get("selected_hypothesis"):
            st.markdown("""<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">Hypothesis</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="glass-card" style="font-style:italic;color:#c7d2fe;font-size:0.9rem;">"{run['selected_hypothesis']}"</div>""", unsafe_allow_html=True)

        if run.get("execution_result"):
            with st.expander("📋 Execution Output"):
                st.code(run["execution_result"], language="text")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Agent Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Agent Pipeline":

    active_agent = st.session_state.get("active_agent")

    AGENT_META = {
        "Literature Reviewer":  ("📚", "Retrieves & indexes relevant papers from arXiv and semantic sources"),
        "Hypothesis Generator": ("💡", "Formulates novel, testable scientific hypotheses"),
        "Code Executor":        ("⚙️", "Writes and executes experiments in a sandboxed environment"),
        "Debugger Node":        ("🔧", "Detects and fixes failures in generated code automatically"),
        "Critic Node":          ("🎓", "Validates results with p-value & Cohen's d statistical checks"),
        "LaTeX Synthesizer":    ("📝", "Compiles findings into a publication-ready research paper"),
    }

    st.markdown("""
    <style>
    .pipeline-step {
        background: linear-gradient(135deg, #111827, #1a2234);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    .pipeline-step.step-active {
        border-color: rgba(99,102,241,0.6);
        background: linear-gradient(135deg, rgba(79,70,229,0.15), rgba(17,24,39,1));
        box-shadow: 0 0 32px rgba(99,102,241,0.2);
    }
    .pipeline-step.step-active::before {
        content:'';
        position:absolute;
        left:0;top:0;bottom:0;
        width:4px;
        background: linear-gradient(180deg, #818cf8, #10b981);
        border-radius:14px 0 0 14px;
    }
    .step-icon { font-size: 1.8rem; min-width: 40px; text-align:center; }
    .step-content { flex: 1; }
    .step-name { font-weight: 700; font-size: 1rem; color: #e2e8f0; }
    .step-name.active-name { color: #a5b4fc; }
    .step-desc { font-size: 0.8rem; color: #475569; margin-top: 2px; }
    .step-connector { text-align: center; color: rgba(99,102,241,0.3); font-size: 1.2rem; margin: -4px 0; }
    </style>
    """, unsafe_allow_html=True)

    if not active_agent:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2rem;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">🚀</div>
            <div style="font-size:1rem;font-weight:600;color:#64748b;">No pipeline running yet</div>
            <div style="font-size:0.85rem;color:#374151;margin-top:4px;">Go to Discovery Studio and start a run to see agents activate in real-time</div>
        </div>
        """, unsafe_allow_html=True)

    for i, agent in enumerate(AGENTS):
        emoji, desc = AGENT_META[agent]
        is_active = agent == active_agent
        active_cls = "step-active" if is_active else ""
        name_cls = "active-name" if is_active else ""
        badge = '<span class="status-badge badge-info" style="float:right;margin-top:-2px;">● Live</span>' if is_active else ""
        st.markdown(f"""
        <div class="pipeline-step {active_cls}">
            <div class="step-icon">{emoji}</div>
            <div class="step-content">
                <div class="step-name {name_cls}">{agent} {badge}</div>
                <div class="step-desc">{desc}</div>
            </div>
        </div>
        {"<div class='step-connector'>↓</div>" if i < len(AGENTS)-1 else ""}
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Literature & Citations
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Literature & Citations":

    col_a, col_b = st.columns([3, 1], gap="medium")
    with col_a:
        query = st.text_input("🔍 Search Query", placeholder="e.g. attention mechanisms, sparse neural networks…",
                              label_visibility="collapsed")
    with col_b:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, label_visibility="collapsed")

    min_score = st.slider("Minimum Similarity Score", 0.0, 0.5, 0.0, 0.05,
                          format="%.2f")

    if st.button("🔍 Search Papers", use_container_width=False, disabled=not (query or "").strip()):
        try:
            from src.rag.retriever import get_relevant_literature
            with st.spinner("Searching vector database…"):
                papers = get_relevant_literature(query, top_k=top_k, min_score=min_score)

            if not papers:
                st.markdown("""<div class="glass-card" style="text-align:center;color:#64748b;">No matching papers found. Try a broader query.</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">Found <strong style="color:#a5b4fc">{len(papers)}</strong> papers</p>', unsafe_allow_html=True)
                for paper in papers:
                    title = paper.get("title") or paper.get("paper_id", "Untitled")
                    authors = ", ".join(paper.get("authors") or []) or "Authors unavailable"
                    year = paper.get("year") or "Year unknown"
                    source = paper.get("source", "unknown")
                    snippet = paper.get("snippet") or "No snippet available."
                    sim = paper.get("similarity", "—")
                    rank = paper.get("rank", "")
                    url = paper.get("url", "")

                    sim_pct = f"{float(sim)*100:.1f}%" if isinstance(sim, (float, int)) else str(sim)

                    with st.container():
                        st.markdown(f"""
                        <div class="glass-card" style="margin-bottom:0.75rem;">
                            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
                                <div style="flex:1;">
                                    <div style="font-weight:700;font-size:1rem;color:#c7d2fe;margin-bottom:4px;">
                                        {rank}{"." if rank else ""} {title}
                                    </div>
                                    <div style="font-size:0.78rem;color:#475569;margin-bottom:0.75rem;">
                                        {authors} · {year} · <span style="color:#374151">{source}</span>
                                    </div>
                                    <div style="font-size:0.85rem;color:#94a3b8;line-height:1.5;">{snippet[:300]}{'…' if len(snippet)>300 else ''}</div>
                                </div>
                                <div style="text-align:right;min-width:80px;">
                                    <div class="status-badge badge-info">{sim_pct}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if url:
                            st.link_button("🔗 Open Paper", url)
        except Exception as exc:
            st.error(f"Search failed: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Paper Viewer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Paper Viewer":
    render_paper_viewer()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Cost Analytics
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Cost Analytics":
    try:
        from components.cost_analytics import render_cost_analytics

        # Style the plotly charts dark
        import plotly.io as pio
        pio.templates.default = "plotly_dark"

        render_cost_analytics()
    except Exception as exc:
        st.error(f"Failed to load cost analytics: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Holdout Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Holdout Benchmark":

    st.markdown("""
    <div class="glass-card" style="margin-bottom:1.5rem;">
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1.5rem;text-align:center;">
            <div>
                <div style="font-size:2rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">9</div>
                <div style="font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Holdout Papers</div>
            </div>
            <div>
                <div style="font-size:2rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">4</div>
                <div style="font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Scoring Dimensions</div>
            </div>
            <div>
                <div style="font-size:2rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">3</div>
                <div style="font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Domains</div>
            </div>
            <div>
                <div style="font-size:2rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">10</div>
                <div style="font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Max Score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:0.75rem;margin-bottom:1.5rem;flex-wrap:wrap;">
        <span class="status-badge badge-info">🤖 AI / ML</span>
        <span class="status-badge badge-success">🧬 Computational Biology</span>
        <span class="status-badge badge-warning">⚛️ Physics</span>
        <span class="status-badge badge-info">📅 2024 – 2026</span>
    </div>
    """, unsafe_allow_html=True)

    radar_path = Path("docs/figures/benchmark_radar.svg")
    if radar_path.exists():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(str(radar_path), caption="Agent vs. Human — 4-Dimension Radar Chart")
    else:
        st.markdown("""<div class="glass-card" style="text-align:center;color:#475569;">
            Radar chart not found at <code>docs/figures/benchmark_radar.svg</code>
        </div>""", unsafe_allow_html=True)

    report_path = Path("docs/evaluation_benchmark_report.md")
    if report_path.exists():
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Evaluation Report</p>', unsafe_allow_html=True)
        st.markdown(report_path.read_text(encoding="utf-8"))