import json

import pytest

from src.evaluation.benchmark_evaluator import (
    DIMENSIONS,
    average_scores,
    evaluate_agent_output,
    evaluate_human_paper,
    format_scores,
    run_benchmark,
    score_empirical,
    score_empirical_from_text,
    score_formatting,
    score_grounding,
    score_grounding_from_references,
    score_novelty,
    similarity,
)

RETRIEVED = {
    "p1": "beta-lactam resistance rising in Escherichia coli under antibiotic pressure",
    "p2": "efflux pumps let Escherichia coli survive sub-inhibitory antibiotic doses",
    "p3": "galaxy rotation curves and dark matter halos",
}
CLAIM = "efflux pump inhibitors slow beta-lactam resistance emergence in Escherichia coli"

AGENT_TEXT = ("\\begin{abstract}" + " ".join(["word"] * 150) + "\\end{abstract}\n"
              "\\section{Introduction}\nx\n\\section{Methods}\nx\n\\begin{figure}x\\end{figure}\n"
              "\\section{Results}\n\\begin{table}x\\end{table}\n\\section{Conclusion}\nx\n"
              "\\section*{References}\n" + "\n".join(f"\\bibitem{{r{i}}} R" for i in range(5)))

HUMAN_TEXT = """Abstract
""" + " ".join(["word"] * 150) + """
1 Introduction
x
2 Methods
We ran five random seeds and report mean ± std.
Figure 1: Overview.
3 Results
Table 1: Results. The improvement is statistically significant (p < 0.01, t-test).
4 Conclusion
x
References
""" + "\n".join(f"[{i}] Author. Title of the paper number {i}. 2024." for i in range(1, 31))


def test_similarity_basics():
    assert similarity("protein folding energy", "protein folding energy") > 0.99
    assert similarity("protein folding", "galaxy rotation") == 0.0
    assert similarity("", "anything") == 0.0


def test_novelty_copy_vs_new_idea():
    refs = ["deep learning predicts protein structure from sequence"]
    copy = score_novelty("deep learning predicts protein structure from sequence", refs)
    new = score_novelty("quantum annealing schedules for portfolio optimisation", refs)
    assert copy["score"] < 1.0
    assert new["score"] == 10.0
    assert score_novelty("idea", [])["score"] == 5.0
    assert score_novelty("", refs)["score"] == 0.0


def test_grounding_rewards_real_relevant_citations():
    good = score_grounding(CLAIM, ["p1", "p2", "p1"], RETRIEVED)
    assert good["citations"] == 2 and good["authentic"] == 2
    fake = score_grounding(CLAIM, ["p1", "p2", "made-up"], RETRIEVED)
    assert fake["invalid_ids"] == ["made-up"]
    irrelevant = score_grounding(CLAIM, ["p3", "p3b", "p3c"], {**RETRIEVED, "p3b": "stars", "p3c": "planets"})
    full = score_grounding(CLAIM, ["p1", "p2", "p2b"], {**RETRIEVED, "p2b": RETRIEVED["p2"]})
    assert full["score"] > fake["score"] > 0
    assert full["score"] > irrelevant["score"]
    assert score_grounding(CLAIM, [], RETRIEVED)["score"] == 0.0


def test_grounding_from_references():
    assert score_grounding_from_references(25)["score"] == 10.0
    assert score_grounding_from_references(10)["score"] == 4.0
    assert score_grounding_from_references(0)["score"] == 0.0


@pytest.mark.parametrize("p, d, ci, expected", [
    (0.01, 0.8, [1.0, 3.0], 10.0),
    (0.01, 0.3, None, 7.0),
    (0.01, None, [-1.0, 2.0], 6.0),
    (0.20, 0.9, [0.5, 2.0], 3.0),
    (None, None, None, 0.0),
])
def test_empirical(p, d, ci, expected):
    assert score_empirical(p, d, ci)["score"] == expected


def test_empirical_from_text():
    r = score_empirical_from_text(HUMAN_TEXT, n_tables=1, n_figures=1)
    assert r["score"] == 10.0
    assert score_empirical_from_text("we describe a new idea")["score"] == 0.0


def test_formatting():
    full = score_formatting({"sections": ["abstract", "introduction", "method", "results",
                                          "conclusion", "references"],
                             "abstract_words": 200, "figures": 2, "tables": 1, "references": 20})
    assert full["score"] == 10.0
    partial = score_formatting({"sections": ["abstract", "introduction"], "abstract_words": 20})
    assert partial["score"] == 2.0
    assert "results" in partial["missing_sections"]


def test_evaluate_agent_output():
    out = {"paper_id": "agent:1", "title": "T", "paper_text": AGENT_TEXT, "hypothesis": CLAIM,
           "cited_ids": ["p1", "p2", "made-up"],
           "retrieved_papers": [{"paper_id": k, "title": "", "abstract": v} for k, v in RETRIEVED.items()],
           "p_value": 0.01, "effect_size": 0.6, "ci": [0.2, 1.4]}
    s = evaluate_agent_output(out)
    assert s.source == "agent"
    assert s.formatting == 10.0 and s.empirical_validity == 10.0
    assert s.details["grounding"]["invalid_ids"] == ["made-up"]
    assert 0 < s.novelty < 10
    assert s.overall == round(sum(getattr(s, d) for d in DIMENSIONS) / 4, 2)
    empty = evaluate_agent_output({})
    assert empty.overall < 2


def test_evaluate_human_paper_with_text():
    paper = {"paper_id": "arxiv:1", "title": "Seeds and significance", "abstract": "a b c",
             "domain": "AI"}
    other = {"paper_id": "arxiv:2", "title": "Unrelated physics", "abstract": "lattice phonons"}
    s = evaluate_human_paper(paper, [paper, other], text=HUMAN_TEXT)
    assert s.source == "human"
    assert s.grounding == 10.0 and s.empirical_validity == 10.0 and s.formatting == 10.0
    assert s.details["novelty"]["compared_with"] == 1


def test_run_benchmark_and_report(tmp_path):
    papers = [{"paper_id": f"arxiv:{i}", "title": f"Paper {i}", "abstract": f"topic{i} study",
               "domain": "AI", "structure": {"sections": ["abstract", "introduction"],
                                             "references": 10}} for i in range(3)]
    (tmp_path / "benchmark_metadata.json").write_text(json.dumps({"papers": papers}))
    agent = {"title": "A", "paper_text": AGENT_TEXT, "hypothesis": "new idea", "p_value": 0.01}
    result = run_benchmark(agent, tmp_path)
    assert len(result["human"]) == 3
    assert result["agent"]["source"] == "agent"
    assert result["human_average"]["grounding"] == 4.0
    text = format_scores(result)
    assert "Human average" in text and "Agent" in text


def test_average_scores_empty():
    assert average_scores([]) == {d: 0.0 for d in DIMENSIONS + ["overall"]}
