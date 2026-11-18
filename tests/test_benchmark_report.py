import json
import xml.dom.minidom

from src.evaluation.benchmark_report import (
    comparison_rows,
    domain_averages,
    generate,
    markdown_comparison,
    markdown_domains,
    markdown_papers,
    radar_points,
    radar_svg,
)


def score(pid, domain, n, g, e, f, source="human"):
    return {"paper_id": pid, "title": f"Title {pid} | x", "source": source, "domain": domain,
            "novelty": n, "grounding": g, "empirical_validity": e, "formatting": f,
            "overall": round((n + g + e + f) / 4, 2)}


HUMAN = [score("arxiv:1", "AI", 9, 10, 4, 8), score("arxiv:2", "AI", 8, 10, 6, 10),
         score("arxiv:3", "Physics", 9, 5, 2, 8)]
AVG = {"novelty": 8.67, "grounding": 8.33, "empirical_validity": 4.0, "formatting": 8.67, "overall": 7.42}
RESULT = {"agent": score("agent:sample-1", "AI", 7, 9.5, 10, 10, source="agent"),
          "human": HUMAN, "human_average": AVG}


def test_domain_averages():
    d = domain_averages(HUMAN)
    assert list(d) == ["AI", "Physics"]
    assert d["AI"]["papers"] == 2 and d["AI"]["empirical_validity"] == 5.0
    assert d["Physics"]["overall"] == 6.0


def test_comparison_rows_leaders():
    rows = {r["dimension"]: r for r in comparison_rows(RESULT)}
    assert rows["Novelty"]["leader"] == "Human" and rows["Novelty"]["gap"] == -1.67
    assert rows["Empirical Validity"]["leader"] == "Agent"
    assert len(rows) == 5


def test_comparison_without_agent():
    rows = comparison_rows({**RESULT, "agent": None})
    assert all(r["agent"] is None and r["leader"] == "-" for r in rows)
    assert "| - |" in markdown_comparison({**RESULT, "agent": None})


def test_markdown_tables():
    assert "| Novelty | 7.00 | 8.67 | -1.67 | Human |" in markdown_comparison(RESULT)
    assert "| AI | 2 |" in markdown_domains(RESULT)
    papers = markdown_papers(RESULT)
    assert "[arxiv:1](https://arxiv.org/abs/1)" in papers
    assert "**Agent**" in papers
    assert "Title arxiv:1 / x" in papers


def test_radar_geometry():
    pts = radar_points([10, 10, 10, 10], 100, 100, 50)
    assert pts[0] == (100.0, 50.0) and pts[1] == (150.0, 100.0)
    assert radar_points([0, 0, 0, 0], 100, 100, 50)[2] == (100.0, 100.0)
    assert radar_points([20], 0, 0, 10)[0] == (0.0, -10.0)


def test_radar_svg_is_valid_xml():
    svg = radar_svg({"Agent": [7, 9, 10, 10], "Human average": [9, 9, 5, 8]},
                    ["Novelty", "Grounding", "Empirical", "Formatting & Structure"])
    xml.dom.minidom.parseString(svg)
    assert svg.count("<polygon") == 5 + 2
    assert "Formatting &amp; Structure" in svg


def test_generate_writes_report_and_charts(tmp_path):
    report = tmp_path / "docs" / "evaluation_benchmark_report.md"
    figures = tmp_path / "docs" / "figures"
    paths = generate(RESULT, report, figures)
    assert report.is_file() and (figures / "benchmark_radar.svg").is_file()
    text = report.read_text()
    assert "# Evaluation Benchmark Report" in text
    assert "figures/benchmark_radar" in text
    assert "sample paper" in text
    assert paths["report"] == report.as_posix()
    json.dumps(paths)


def test_findings_name_leads_and_gaps():
    from src.evaluation.benchmark_report import findings

    text = "\n".join(findings(RESULT))
    assert "biggest lead is **Empirical Validity**" in text
    assert "biggest gap is **Novelty**" in text
    assert "AI papers scored highest" in text
    no_agent = "\n".join(findings({**RESULT, "agent": None}))
    assert "biggest" not in no_agent and "weakest dimension" in no_agent


def test_report_has_findings_and_limitations(tmp_path):
    report = tmp_path / "docs" / "report.md"
    generate(RESULT, report, tmp_path / "docs" / "figures")
    text = report.read_text()
    for heading in ("## Key findings", "## Limitations", "## Next steps", "## Reproduce"):
        assert heading in text
