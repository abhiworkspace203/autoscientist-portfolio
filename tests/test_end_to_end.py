"""
End-to-end test of the whole pipeline (Member 5, Week 4).

    python -m pytest tests/test_end_to_end.py -v

One command checks that a research query travels through every stage without crashing:

    query -> ingestion -> vector DB -> retrieval -> hypothesis -> code execution
          -> self-debugging -> statistics -> critic -> cost tracking -> paper generation

The test is hermetic: no network, no API keys, no Docker, and it writes nothing into the
repository (the scrapers are stubbed with the sample papers, the vector store and the
episodic memory live in a temporary folder, live arXiv calls are blocked).

The paper-generation stage and the master entrypoint (main.py) belong to other members. Their
tests are skipped with a clear reason until those files exist in the branch.
"""
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PAPERS = ROOT / "data" / "sample_scraped_papers.json"
TOPIC = "AI agent time horizons"
VERDICTS = {"ACCEPTED", "REJECTED", "NEEDS_REPLICATION"}
TIMED_NODES = {"literature_reviewer", "hypothesis_generator", "code_executor", "critic"}


# ---------------------------------------------------------------------------
# Hermetic environment, built once for the whole file
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def offline():
    """Block the network so nothing live is called (the agents fall back to local data)."""
    mp = pytest.MonkeyPatch()

    def no_urlopen(*args, **kwargs):
        raise urllib.error.URLError("network disabled for the end-to-end test")

    def no_dns(*args, **kwargs):
        raise socket.gaierror("network disabled for the end-to-end test")

    mp.setattr(urllib.request, "urlopen", no_urlopen)
    mp.setattr(socket, "getaddrinfo", no_dns)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="module")
def ingested(offline, workspace):
    """Stage 1: the real ingestion pipeline, with the two scrapers returning the sample papers."""
    from src.ingestion import pipeline
    from src.ingestion.schema import load_papers

    sample = load_papers(SAMPLE_PAPERS)
    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline.ArxivScraper, "search", lambda self, *a, **k: [p for p in sample if p.source == "arxiv"])
    mp.setattr(pipeline.PubMedScraper, "search", lambda self, *a, **k: [p for p in sample if p.source == "pubmed"])
    try:
        return pipeline.run_ingestion(["cs.AI"], "machine learning", incremental=False,
                                      state_path=workspace / "ingestion_state.json")
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def vector_db(ingested, workspace):
    """Stage 2: index the ingested papers in a temporary vector store."""
    from src.ingestion.schema import save_papers
    from src.rag.vector_db import ScientificVectorDB

    papers_file = save_papers(ingested, workspace / "papers.json")
    db = ScientificVectorDB(persist_directory=workspace / "vector_store")
    db.load_from_json(str(papers_file))
    return db


@pytest.fixture(scope="module")
def retriever(vector_db):
    """Stage 3: the retriever the orchestrator's literature node uses, pointed at the temporary store."""
    from src.rag import retriever as retriever_module

    ret = retriever_module.LiteratureRetriever(db=vector_db)
    mp = pytest.MonkeyPatch()
    mp.setattr(retriever_module, "_default_retriever", ret)
    yield ret
    mp.undo()


@pytest.fixture(scope="module")
def tracker():
    from src.utils.cost_tracker import CostTracker

    return CostTracker(run_id="end-to-end")


@pytest.fixture(scope="module")
def graph(retriever, workspace, tracker, offline):
    """The real LangGraph pipeline, with every node timed by the cost tracker."""
    from src import orchestrator
    from src.memory import episodic_memory
    from src.utils.cost_tracker import track_pipeline

    mp = pytest.MonkeyPatch()
    mp.setattr(episodic_memory, "DEFAULT_STORAGE_PATH", workspace / "episodes.json")  # keep the repo clean
    with track_pipeline(tracker, orchestrator):  # the graph is built (and run) inside the block
        yield orchestrator.build_orchestrator(enable_critic=True)
    mp.undo()


@pytest.fixture(scope="module")
def normal_run(graph, tracker):
    """A full discovery run: query -> ... -> critic verdict."""
    with tracker.experiment("normal run"):
        return graph.invoke({"research_topic": TOPIC, "enable_critic": True})


@pytest.fixture(scope="module")
def debug_run(graph, tracker):
    """A run whose first experiment crashes (divide by zero) and must be repaired by the debugger."""
    crashing = "values = [4, 8, 15]\ncount = len([])\nprint('Baseline mean: 1')\nprint(sum(values) / count)\n"
    with tracker.experiment("self-debugging run"):
        return graph.invoke({"research_topic": TOPIC, "generated_code": crashing, "enable_critic": False})


# ---------------------------------------------------------------------------
# Stage 1-3: query -> ingestion -> vector DB -> retrieval
# ---------------------------------------------------------------------------
def test_ingestion_returns_valid_papers_from_both_sources(ingested):
    assert len(ingested) == 10
    assert {p.source for p in ingested} == {"arxiv", "pubmed"}
    assert all(p.is_valid() for p in ingested)


def test_ingested_papers_are_indexed(vector_db):
    assert vector_db.count() > 0


def test_retrieval_returns_papers_for_the_query(retriever, ingested):
    results = retriever.retrieve(TOPIC, top_k=3)
    assert 1 <= len(results) <= 3
    known_ids = {p.paper_id for p in ingested}
    for result in results:
        assert result["paper_id"] in known_ids
        assert result["title"] and result["snippet"]


# ---------------------------------------------------------------------------
# Stage 4-8: hypothesis -> execution -> critic, through the real graph
# ---------------------------------------------------------------------------
def test_full_run_finishes_and_fills_every_stage(normal_run):
    assert normal_run["literature_summary"].strip()
    assert normal_run["selected_hypothesis"].strip()
    assert normal_run["generated_code"].strip()
    assert normal_run["execution_status"] == "SUCCESS"
    assert normal_run["execution_result"].strip()


def test_literature_stage_used_the_temporary_vector_store(normal_run, ingested):
    titles = {p.title for p in ingested}
    retrieved = normal_run["retrieved_papers"]
    assert retrieved and all(r["title"] in titles for r in retrieved)


def test_experiment_output_contains_the_measurements(normal_run):
    assert "Metric delta" in normal_run["execution_result"]


def test_critic_gives_a_valid_verdict_and_the_loop_is_bounded(normal_run):
    assert normal_run["critic_verdict"] in VERDICTS
    assert normal_run["critic_critique"].strip()
    assert 1 <= normal_run["experiment_iteration"] <= 2


def test_self_debugging_repairs_a_crashing_experiment(debug_run):
    assert debug_run["execution_status"] == "SUCCESS"
    assert debug_run["retry_count"] == 1
    assert "or 1" in debug_run["generated_code"]  # the divisor guard added by the debugger


def test_episodic_memory_went_to_the_temporary_folder(normal_run, workspace):
    assert (workspace / "episodes.json").exists()


# ---------------------------------------------------------------------------
# Cost tracking across the real pipeline
# ---------------------------------------------------------------------------
def test_every_agent_stage_was_timed(normal_run, debug_run, tracker):
    assert TIMED_NODES <= set(tracker.node_latency_s)
    assert "debugger" in tracker.node_latency_s
    assert all(seconds > 0 for seconds in tracker.node_latency_s.values())


def test_each_experiment_has_a_latency(normal_run, debug_run, tracker):
    assert {e["name"] for e in tracker.experiments} == {"normal run", "self-debugging run"}
    assert all(e["latency_s"] > 0 for e in tracker.experiments)


def test_cost_summary_can_be_saved_and_reloaded(normal_run, tracker, tmp_path):
    from src.utils.cost_tracker import append_run, load_runs

    run = load_runs(append_run(tracker, tmp_path / "cost_log.json"))[0]
    assert run["run_id"] == "end-to-end"
    assert TIMED_NODES <= set(run["by_agent"])


def test_dashboard_can_chart_the_real_runs_node_timings(normal_run, debug_run, tracker):
    pytest.importorskip("plotly")
    from dashboard.components.cost_analytics import node_latency_figure

    fig = node_latency_figure(tracker.to_dict())
    assert TIMED_NODES <= set(fig.data[0].x)


def test_dashboard_charts_can_be_built_from_a_tracked_run(normal_run):
    pytest.importorskip("plotly")
    from dashboard.components.cost_analytics import cumulative_cost_figure, token_distribution_figure
    from src.utils.cost_tracker import CostTracker

    # No LLM is configured in this test, so one call is recorded by hand to give the charts data.
    run_tracker = CostTracker(run_id="charts")
    run_tracker.record("hypothesis_generator", "gpt-4o", 1200, 300, latency_s=0.1)
    run = run_tracker.to_dict()
    assert len(cumulative_cost_figure([run]).data) == 1
    assert len(token_distribution_figure(run).data) == 2


# ---------------------------------------------------------------------------
# Master entrypoint (main.py, Team Lead): skipped until the file exists in the branch
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def main_module():
    if not (ROOT / "main.py").exists():
        pytest.skip("main.py (Team Lead) is not in this branch yet")
    import importlib

    return importlib.import_module("main")


def test_master_entrypoint_shows_help(main_module):
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--help"], capture_output=True, text=True,
                            timeout=120, cwd=ROOT)
    assert result.returncode == 0 and "--topic" in result.stdout


def test_run_pipeline_from_main_returns_the_full_state(main_module, graph):
    result = main_module.run_pipeline(TOPIC, sandbox_mode="local", synthesize_latex=False,
                                      enable_critic=True, quiet=True)
    assert result["execution_status"] == "SUCCESS"
    assert result["critic_verdict"] in VERDICTS
    assert result["selected_hypothesis"].strip() and result["elapsed_seconds"] > 0


# ---------------------------------------------------------------------------
# Paper generation (synthesize_latex_paper in main.py)
# ---------------------------------------------------------------------------
def write_paper(main_module, state, folder):
    info = main_module.synthesize_latex_paper(state, folder)
    return info, Path(info["tex_path"]).read_text(encoding="utf-8")


def make_state(**overrides):
    state = {"research_topic": "Sparse Transformers", "selected_hypothesis": "Pruning reduces memory.",
             "execution_result": "Baseline mean: 100\nMethod mean: 102"}
    state.update(overrides)
    return state


def test_paper_stage_writes_a_complete_tex_file(main_module, normal_run, tmp_path):
    info, tex = write_paper(main_module, normal_run, tmp_path)
    assert info["status"] == "SUCCESS"
    assert normal_run["research_topic"] in tex
    assert "\\begin{document}" in tex and "\\end{document}" in tex


def test_paper_reports_the_runs_verdict_and_p_value(main_module, normal_run, tmp_path):
    _, tex = write_paper(main_module, normal_run, tmp_path)
    assert normal_run["critic_verdict"] in tex
    assert str(normal_run["statistical_metrics"]["p_value"]) in tex


def test_paper_stage_does_not_write_into_the_repository(main_module, normal_run, tmp_path):
    info, _ = write_paper(main_module, normal_run, tmp_path)
    assert Path(info["tex_path"]).parent == tmp_path


# Known problems in the paper stage, found in QA. Non-strict xfail: the file stays green now and
# reports XPASS once each one is fixed. Each reason says what a fix must do.
@pytest.mark.xfail(strict=False, reason="BUG: the abstract says the hypothesis was 'confirmed' even when the "
                                        "critic verdict is REJECTED. Wording must follow the verdict.")
def test_paper_does_not_claim_confirmation_when_the_critic_rejected_it(main_module, tmp_path):
    state = make_state(critic_verdict="REJECTED", statistical_metrics={"p_value": 0.14, "effect_size_cohens_d": 0.3})
    _, tex = write_paper(main_module, state, tmp_path)
    assert "confirmed our central hypothesis" not in tex
    assert "publication-ready empirical finding" not in tex


@pytest.mark.xfail(strict=False, reason="BUG: with no critic result and no statistics the paper invents "
                                        "p < 0.001, Cohen's d 0.85 and verdict ACCEPTED. Missing values must say 'not available'.")
def test_paper_does_not_invent_statistics_or_a_verdict(main_module, tmp_path):
    _, tex = write_paper(main_module, make_state(), tmp_path)
    assert "< 0.001" not in tex and "0.85" not in tex
    assert "ACCEPTED" not in tex


@pytest.mark.xfail(strict=False, reason="BUG: the .pdf written when no LaTeX compiler exists is a text stub "
                                        "('%PDF-1.4 simulated publication draft') that no PDF viewer can open.")
def test_the_pdf_output_is_a_real_pdf(main_module, normal_run, tmp_path):
    fitz = pytest.importorskip("pymupdf", reason="PyMuPDF is needed to check the PDF")
    info, _ = write_paper(main_module, normal_run, tmp_path)
    document = fitz.open(info["pdf_path"])
    assert document.page_count >= 1


@pytest.mark.xfail(strict=False, reason="BUG: LaTeX special characters (% & _ #) in the hypothesis are inserted "
                                        "unescaped, so a % comments out the rest of the line when compiled.")
def test_paper_escapes_latex_special_characters(main_module, tmp_path):
    state = make_state(selected_hypothesis="Pruning cuts memory by 30% & latency for x_1 # tokens.")
    _, tex = write_paper(main_module, state, tmp_path)
    assert "30% &" not in tex  # must appear as 30\% \& (or be removed), never raw


# ---------------------------------------------------------------------------
# Stage owned by another member: skipped until the file exists in this branch
# ---------------------------------------------------------------------------
def test_member2_synthesizer_agent_when_it_exists():
    pytest.importorskip("src.agents.synthesizer_agent",
                        reason="Member 2's synthesizer_agent.py is not in this branch yet")
    pytest.skip("synthesizer_agent.py exists now: extend this test with its real functions "
                "(verified numbers + citations -> .tex -> PDF)")
