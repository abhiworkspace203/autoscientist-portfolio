import json

from src.ingestion.figure_extractor import load_extracted_figures
from src.ingestion.figure_report import (
    figures_for_chart_reader,
    format_report,
    summarise_figures,
)


def make_store(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"png")
    folder = tmp_path / "arxiv_1"
    folder.mkdir()
    figs = [
        {"paper_id": "arxiv:1", "path": str(img), "caption": "Figure 1: Accuracy", "kind": "raster"},
        {"paper_id": "arxiv:1", "path": str(tmp_path / "gone.png"), "caption": "Figure 2: Loss", "kind": "vector"},
        {"paper_id": "arxiv:1", "path": str(img), "caption": "", "kind": "raster"},
    ]
    (folder / "figures.json").write_text(json.dumps(figs))
    return figs


def test_summary_counts(tmp_path):
    make_store(tmp_path)
    s = summarise_figures(load_extracted_figures(tmp_path))
    assert s["total_papers"] == 1
    assert s["total_figures"] == 3
    assert s["total_captioned"] == 2
    assert s["papers"]["arxiv:1"]["vector"] == 1
    assert s["caption_rate"] == round(2 / 3, 3)


def test_empty_summary():
    s = summarise_figures([])
    assert s["total_figures"] == 0
    assert s["caption_rate"] == 0.0


def test_chart_reader_list_skips_missing_and_uncaptioned(tmp_path):
    figs = make_store(tmp_path)
    ready = figures_for_chart_reader(figs)
    assert len(ready) == 1
    assert ready[0]["caption"] == "Figure 1: Accuracy"


def test_format_report(tmp_path):
    make_store(tmp_path)
    text = format_report(summarise_figures(load_extracted_figures(tmp_path)))
    assert "arxiv:1" in text
    assert "3 figures from 1 papers, 2 with captions (67%)" in text
