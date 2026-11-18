import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

try:
    import fitz
except ImportError:
    fitz = None

if fitz is None:
    pytest.skip("PyMuPDF is not installed", allow_module_level=True)

from src.evaluation.holdout_curator import (
    HOLDOUT_PLAN,
    Slot,
    curate,
    load_holdout,
    passes_metadata_rules,
    slot_query,
)
from src.evaluation.paper_structure import analyse_structure, find_sections, pdf_text

LONG_ABSTRACT = " ".join(["word"] * 150)

PAPER_TEXT = """Title of a paper
Abstract
We study something important. """ + LONG_ABSTRACT + """
1 Introduction
Text here.
2 Related Work
More text.
3. Methods
How we did it.
Figure 1: Overview of the method.
4 Results
Table 1: Main results.
Figure 2: Accuracy curve.
5 Conclusion
We conclude.
References
[1] A. Author. A paper. 2024.
[2] B. Author. Another paper. 2025.
[3] C. Author. Third paper. 2023.
"""


def fake_result(n, year=2025, abstract=LONG_ABSTRACT, pdf=True):
    return SimpleNamespace(
        get_short_id=lambda: f"25{n:02d}.0000{n}v1",
        title=f"Paper  {n}",
        authors=[SimpleNamespace(name="Ada Lovelace")],
        published=datetime(year, 5, 1, tzinfo=timezone.utc),
        entry_id=f"http://arxiv.org/abs/25{n:02d}.0000{n}v1",
        pdf_url=f"http://arxiv.org/pdf/25{n:02d}.0000{n}v1" if pdf else None,
        summary=abstract,
    )


def write_pdf(path, text):
    doc = fitz.open()
    page = doc.new_page()
    y = 40
    for line in text.splitlines():
        page.insert_text((40, y), line[:90], fontsize=8)
        y += 10
        if y > 800:
            page = doc.new_page()
            y = 40
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return True


def test_sections_found_in_order():
    assert find_sections(PAPER_TEXT) == [
        "abstract", "introduction", "related_work", "method", "results", "conclusion", "references"]


def test_structure_counts():
    s = analyse_structure(PAPER_TEXT)
    assert s["figures"] == 2
    assert s["tables"] == 1
    assert s["references"] == 3
    assert s["missing_sections"] == []
    assert s["abstract_words"] >= 150


def test_latex_and_markdown_structure():
    tex = ("\\begin{abstract}Short abstract here.\\end{abstract}\n\\section{Introduction}\nx\n"
           "\\section{Methodology}\n\\begin{figure}x\\end{figure}\n\\section{Results}\n"
           "\\begin{table}x\\end{table}\n\\section{Conclusion}\n\\section*{References}\n"
           "\\bibitem{a} A\n\\bibitem{b} B\n")
    s = analyse_structure(tex)
    assert s["missing_sections"] == []
    assert s["figures"] == 1 and s["tables"] == 1 and s["references"] == 2
    md = "## Abstract\nx\n## Introduction\n![f](fig_1.png)\n## Results\n"
    assert find_sections(md) == ["abstract", "introduction", "results"]


def test_slot_query():
    q = slot_query(Slot("AI", "cs.LG", 2024, 6))
    assert q == "cat:cs.LG AND submittedDate:[202406010000 TO 202406282359]"


def test_plan_covers_domains_and_years():
    assert 5 <= len(HOLDOUT_PLAN) <= 10
    assert {s.domain for s in HOLDOUT_PLAN} == {"AI", "Computational Biology", "Physics"}
    assert {s.year for s in HOLDOUT_PLAN} == {2024, 2025, 2026}


def test_metadata_rules():
    rec = {"paper_id": "arxiv:1", "published_date": "2025-01-01", "pdf_url": "u",
           "abstract": LONG_ABSTRACT}
    assert passes_metadata_rules(rec, [])
    assert not passes_metadata_rules(rec, ["arxiv:1"])
    assert not passes_metadata_rules({**rec, "published_date": "2023-05-01"}, [])
    assert not passes_metadata_rules({**rec, "abstract": "too short"}, [])
    assert not passes_metadata_rules({**rec, "pdf_url": None}, [])


def test_curate_with_pdfs(tmp_path):
    plan = [Slot("AI", "cs.LG", 2025, 1), Slot("Physics", "physics.comp-ph", 2025, 2)]
    results = {
        "cs.LG": [fake_result(1, year=2023), fake_result(2)],
        "physics.comp-ph": [fake_result(2), fake_result(3, abstract="short"), fake_result(4)],
    }
    meta = curate(tmp_path, plan=plan, search=lambda s: results[s.category],
                  fetch=lambda url, path: write_pdf(path, PAPER_TEXT), pause=0)
    ids = [p["paper_id"] for p in meta["papers"]]
    assert ids == ["arxiv:2502.00002v1", "arxiv:2504.00004v1"]
    assert meta["count"] == 2
    first = meta["papers"][0]
    assert first["domain"] == "AI" and first["structure"]["references"] == 3
    assert (tmp_path / "pdfs" / "arxiv_2502.00002v1.pdf").is_file()
    saved = json.loads((tmp_path / "benchmark_metadata.json").read_text())
    assert saved["papers"] == meta["papers"]
    assert load_holdout(tmp_path) == meta["papers"]


def test_curate_skips_unreadable_pdf(tmp_path):
    plan = [Slot("AI", "cs.LG", 2025, 1)]
    meta = curate(tmp_path, plan=plan, search=lambda s: [fake_result(1), fake_result(2)],
                  fetch=lambda url, path: "00001" not in url and write_pdf(path, PAPER_TEXT), pause=0)
    assert [p["paper_id"] for p in meta["papers"]] == ["arxiv:2502.00002v1"]


def test_curate_without_download(tmp_path):
    meta = curate(tmp_path, plan=[Slot("AI", "cs.LG", 2025, 1)], download=False,
                  search=lambda s: [fake_result(1)], pause=0)
    assert meta["count"] == 1 and "structure" not in meta["papers"][0]


def test_missing_files_are_safe(tmp_path):
    assert load_holdout(tmp_path) == []
    assert pdf_text(str(tmp_path / "nope.pdf")) == ""
