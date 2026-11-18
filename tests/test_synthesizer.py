"""
Unit tests for Academic Paper Synthesizer & LaTeX Compiler (Member 2 Deliverable)
"""

from pathlib import Path
import pytest

from src.utils.latex_compiler import is_pdflatex_available, compile_latex_to_pdf, generate_html_fallback
from src.agents.synthesizer_agent import generate_bibtex, synthesize_manuscript, generate_latex_paper


def test_is_pdflatex_available_boolean():
    """Verify pdflatex availability check returns boolean without error."""
    res = is_pdflatex_available()
    assert isinstance(res, bool)


def test_generate_bibtex_creates_valid_bib(tmp_path: Path):
    """Verify BibTeX generation formats preprints properly."""
    papers = [
        {
            "paper_id": "arxiv:2406.12815",
            "title": "Privacy Preserving Federated Learning",
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2024,
        }
    ]
    bib_file = tmp_path / "references.bib"
    generate_bibtex(papers, bib_file)
    assert bib_file.exists()
    content = bib_file.read_text(encoding="utf-8")
    assert "@article{" in content
    assert "arxiv_2406_12815" in content
    assert "Privacy Preserving Federated Learning" in content
    assert "Alice Smith and Bob Jones" in content


def test_synthesize_manuscript_replaces_placeholders():
    """Ensure placeholders are replaced with experimental findings."""
    template = "\\title{<<TITLE>>}\\abstract{<<ABSTRACT>>}\\hyp{<<HYPOTHESIS>>}"
    state = {
        "research_topic": "Attention Pruning",
        "selected_hypothesis": "Dynamic pruning saves memory",
        "critic_verdict": "ACCEPTED",
    }
    res = synthesize_manuscript(state, template)
    assert "Attention Pruning" in res
    assert "Dynamic pruning saves memory" in res
    assert "ACCEPTED" in res
    assert "<<TITLE>>" not in res
    assert "<<ABSTRACT>>" not in res
    assert "<<HYPOTHESIS>>" not in res


def test_compile_latex_to_pdf_fallback(tmp_path: Path):
    """Ensure compiler creates PDF or HTML companion even without pdflatex."""
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text("\\documentclass{article}\\begin{document}Hello\\end{document}", encoding="utf-8")
    pdf_path = tmp_path / "paper.pdf"

    compile_latex_to_pdf(str(tex_path), str(pdf_path))
    assert pdf_path.exists()
    assert (tmp_path / "paper.html").exists()


def test_generate_latex_paper_end_to_end(tmp_path: Path):
    """Verify full synthesis agent creates all 3 artifacts (.tex, .pdf, .bib)."""
    state = {
        "research_topic": "Sparse Graph Transformers",
        "selected_hypothesis": "Sparse message passing maintains F1 while cutting compute by 40%",
        "execution_result": "Baseline: 80\nMethod: 82\nMetric delta: +2",
        "statistical_metrics": {"p_value": 0.01, "effect_size_cohens_d": 0.95},
        "critic_verdict": "ACCEPTED",
        "critic_critique": "Statistically significant improvement confirmed.",
        "retrieved_papers": [{"paper_id": "test_1", "title": "Paper 1"}],
    }

    out = generate_latex_paper(state, output_dir=str(tmp_path))
    assert out["status"] == "SUCCESS"
    assert Path(out["tex_path"]).exists()
    assert Path(out["pdf_path"]).exists()
    assert Path(out["bib_path"]).exists()
