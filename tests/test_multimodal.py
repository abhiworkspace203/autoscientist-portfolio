"""
Multimodal ingestion QA (Member 5, Week 3).

Independent checks on the "Giving the AI Eyes" pipeline:

  PDF  ->  figure_extractor (Member 3)  ->  PNG files + figures.json  ->  chart_reader (Member 2)  ->  JSON

Part A  the hand-off the chart reader relies on (what the extractor promises)       needs figure_extractor
Part B  regression tests for bugs found in QA and fixed by Member 3                  needs figure_extractor
Part C  data/sample_chart_extractions.json (Member 2): fields, honesty, schema       needs the JSON file
Part D  extractor -> chart reader integration, with a fake vision model              needs both modules

Each part skips on its own when what it needs is not in the branch yet.
All tests are offline: synthetic PDFs built with PyMuPDF and a fake vision client, no API calls.
Run:  python -m pytest tests/test_multimodal.py -v
"""
import json
from pathlib import Path

import pytest

try:  # Member 3's module (needs PyMuPDF)
    from src.ingestion import figure_extractor as extractor

    fitz = extractor.fitz
    extract_figures = extractor.extract_figures
except ImportError:
    extractor = fitz = extract_figures = None
if fitz is None:  # the module imports without PyMuPDF (fitz = None) but can not extract anything
    extractor = extract_figures = None

try:  # Member 2's module
    from src.ingestion import chart_reader
except ImportError:
    chart_reader = None

requires_extractor = pytest.mark.skipif(extractor is None, reason="src/ingestion/figure_extractor.py (or PyMuPDF) not available yet")
requires_chart_reader = pytest.mark.skipif(chart_reader is None, reason="src/ingestion/chart_reader.py not available yet")

REQUIRED_KEYS = {
    "paper_id", "index", "page", "path", "width", "height",
    "bbox", "caption", "figure_number", "kind", "parts",
}


# ---------------------------------------------------------------------------
# Helpers: synthetic PDFs
# ---------------------------------------------------------------------------
def make_image(width, height, color=(0.2, 0.4, 0.8)):
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), 0)
    pix.set_rect(pix.irect, tuple(int(c * 255) for c in color))
    return pix


def add_image(page, rect, width, height, color=(0.2, 0.4, 0.8)):
    page.insert_image(fitz.Rect(rect), pixmap=make_image(width, height, color))


def save_pdf(doc, path):
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def three_figure_pdf(tmp_path):
    """One page, three captioned charts stacked down the page."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i, top in enumerate((50, 300, 550), start=1):
        add_image(page, (72, top, 400, top + 200), 600, 400, (0.2 * i, 0.4, 0.8))
        page.insert_text((72, top + 215), f"Figure {i}: Chart number {i}.", fontsize=10)
    return save_pdf(doc, tmp_path / "three.pdf")


# ---------------------------------------------------------------------------
# Part A: what the chart reader can rely on
# ---------------------------------------------------------------------------
@requires_extractor
def test_every_manifest_entry_has_all_fields_for_the_chart_reader(three_figure_pdf, tmp_path):
    figures = extract_figures(three_figure_pdf, "paper", out_dir=tmp_path / "out")
    assert len(figures) == 3
    for fig in figures:
        assert REQUIRED_KEYS <= set(fig), f"missing {REQUIRED_KEYS - set(fig)}"


@requires_extractor
def test_saved_images_exist_and_are_valid_pngs(three_figure_pdf, tmp_path):
    for fig in extract_figures(three_figure_pdf, "paper", out_dir=tmp_path / "out"):
        path = Path(fig["path"])
        assert path.is_file()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        loaded = fitz.Pixmap(str(path))
        assert (loaded.width, loaded.height) == (fig["width"], fig["height"])


@requires_extractor
def test_every_image_meets_the_minimum_size(three_figure_pdf, tmp_path):
    for fig in extract_figures(three_figure_pdf, "paper", out_dir=tmp_path / "out"):
        assert fig["width"] >= 300 and fig["height"] >= 300


@requires_extractor
def test_captions_stay_with_the_right_figure(three_figure_pdf, tmp_path):
    figures = extract_figures(three_figure_pdf, "paper", out_dir=tmp_path / "out")
    assert [f["figure_number"] for f in figures] == [1, 2, 3]
    assert [f["caption"] for f in figures] == [f"Figure {i}: Chart number {i}." for i in (1, 2, 3)]


@requires_extractor
def test_figures_are_numbered_in_reading_order(tmp_path):
    """The lower figure is drawn first in the file, but must still become fig_2."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    add_image(page, (72, 450, 400, 650), 600, 400)
    page.insert_text((72, 665), "Figure 2: Lower chart.", fontsize=10)
    add_image(page, (72, 100, 400, 300), 600, 400)
    page.insert_text((72, 315), "Figure 1: Upper chart.", fontsize=10)
    pdf = save_pdf(doc, tmp_path / "order.pdf")

    figures = extract_figures(pdf, "paper", out_dir=tmp_path / "out")
    assert [f["index"] for f in figures] == [1, 2]
    assert [f["figure_number"] for f in figures] == [1, 2]
    assert [Path(f["path"]).name for f in figures] == ["fig_1.png", "fig_2.png"]


@requires_extractor
def test_manifest_on_disk_matches_what_the_loader_returns(three_figure_pdf, tmp_path):
    out = tmp_path / "out"
    returned = extract_figures(three_figure_pdf, "arxiv:1234.5678v1", out_dir=out)
    on_disk = json.loads((out / "arxiv_1234.5678v1" / "figures.json").read_text(encoding="utf-8"))
    assert on_disk == returned
    assert extractor.load_extracted_figures(out, paper_id="arxiv:1234.5678v1") == returned


@requires_extractor
def test_accented_captions_survive_the_json_round_trip(tmp_path):
    # Characters the PDF's built-in font can draw (Latin-1): plus-minus, u-umlaut, e-acute, degree.
    caption = "Figure 1: Accuracy (\u00b1 std) of the M\u00fcller caf\u00e9 model at 25\u00b0C."
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    add_image(page, (72, 100, 400, 300), 600, 400)
    page.insert_text((72, 315), caption, fontsize=10)
    pdf = save_pdf(doc, tmp_path / "accents.pdf")

    out = tmp_path / "out"
    figure = extract_figures(pdf, "p", out_dir=out)[0]
    loaded = extractor.load_extracted_figures(out)[0]
    assert figure["caption"] == caption
    assert loaded["caption"] == caption


@requires_extractor
def test_a_large_image_repeated_on_every_page_is_extracted_once(tmp_path):
    """A big journal logo in every page header must not become three figures."""
    doc = fitz.open()
    logo = make_image(500, 500, (0.9, 0.1, 0.1))
    for _ in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_image(fitz.Rect(40, 30, 240, 230), pixmap=logo)
    pdf = save_pdf(doc, tmp_path / "logo.pdf")

    figures = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert [f["page"] for f in figures] == [1]


@requires_extractor
def test_grayscale_figure_is_extracted(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    gray = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 500, 400), 0)
    gray.set_rect(gray.irect, (120,))
    page.insert_image(fitz.Rect(72, 60, 400, 300), pixmap=gray)
    page.insert_text((72, 315), "Figure 1: A grayscale chart.", fontsize=10)
    pdf = save_pdf(doc, tmp_path / "gray.pdf")

    figures = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert len(figures) == 1 and figures[0]["caption"].startswith("Figure 1")


@requires_extractor
def test_paper_ids_can_not_write_outside_the_output_folder(three_figure_pdf, tmp_path):
    out = tmp_path / "out"
    figures = extract_figures(three_figure_pdf, "../../escape", out_dir=out)
    assert figures
    for fig in figures:
        assert out.resolve() in Path(fig["path"]).resolve().parents
    assert not (tmp_path / "escape").exists()


@requires_extractor
def test_pdf_without_images_or_charts_returns_an_empty_list(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Only text here.")
    pdf = save_pdf(doc, tmp_path / "blank.pdf")
    assert extract_figures(pdf, "p", out_dir=tmp_path / "out") == []


# ---------------------------------------------------------------------------
# Part B: regression tests for bugs found in QA (both fixed by Member 3)
# ---------------------------------------------------------------------------
@requires_extractor
def test_encrypted_pdf_returns_empty_instead_of_crashing(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "secret")
    pdf = tmp_path / "encrypted.pdf"
    doc.save(pdf, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    assert extract_figures(pdf, "p", out_dir=tmp_path / "out") == []


@requires_extractor
def test_rerun_does_not_leave_stale_images(three_figure_pdf, tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    add_image(page, (72, 50, 400, 250), 600, 400)
    page.insert_text((72, 265), "Figure 1: Only one chart now.", fontsize=10)
    one_pdf = save_pdf(doc, tmp_path / "one.pdf")

    out = tmp_path / "out"
    extract_figures(three_figure_pdf, "paper", out_dir=out)
    figures = extract_figures(one_pdf, "paper", out_dir=out)

    files = sorted(p.name for p in (out / "paper").glob("fig_*.png"))
    assert files == [Path(f["path"]).name for f in figures]


# ---------------------------------------------------------------------------
# Part C: Member 2's chart extractions (data/sample_chart_extractions.json)
# ---------------------------------------------------------------------------
CHART_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_chart_extractions.json"

# Week 3 plan: chart type, axis labels/units, baseline + peak numbers, trend conclusion.
# Several key spellings are accepted; add Member 2's names here if they change.
CHART_FIELDS = {
    "chart type": ["chart_type", "type"],
    "x-axis": ["x_axis", "x_axis_label", "x_label", "xaxis"],
    "y-axis": ["y_axis", "y_axis_label", "y_label", "yaxis"],
    "baseline": ["baseline_values", "baseline", "baseline_numbers", "baseline_value"],
    "peak": ["peak_performance", "peak", "peak_value", "peak_values"],
    "trend": ["trend_summary", "trend", "trend_conclusion", "overall_trend", "conclusion"],
}
READING_TYPES = {"observed", "estimated", "uncertain"}


def _has_key(entry, aliases):
    """True if any alias exists as a key (even with an empty value), also inside nested dicts."""
    if isinstance(entry, dict):
        if any(key in entry for key in aliases):
            return True
        return any(_has_key(value, aliases) for value in entry.values())
    return False


def _find(entry, aliases):
    """Value of the first alias that is non-empty, also inside nested dicts."""
    if isinstance(entry, dict):
        for key in aliases:
            if entry.get(key) not in (None, "", [], {}):
                return entry[key]
        for value in entry.values():
            found = _find(value, aliases)
            if found is not None:
                return found
    return None


def _load_chart_entries():
    """Return the list of extractions (the `extraction` part of each result that succeeded)."""
    if not CHART_FILE.exists() or CHART_FILE.stat().st_size == 0:
        pytest.skip("data/sample_chart_extractions.json not pushed yet (Member 2)")
    data = json.loads(CHART_FILE.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if lists else [data]
    entries = []
    for item in data:
        if isinstance(item, dict) and item.get("status", "ok") != "ok":
            continue
        entries.append(item.get("extraction", item) if isinstance(item, dict) else item)
    return entries


def _numeric_readings(extraction):
    readings = [v for v in extraction.get("baseline_values", []) if v.get("value") is not None]
    peak = extraction.get("peak_performance")
    if peak and peak.get("value") is not None:
        readings.append(peak)
    readings += [p for p in extraction.get("data_points", []) if p.get("y") is not None]
    return readings


def test_chart_extractions_cover_at_least_three_charts():
    assert len(_load_chart_entries()) >= 3


def test_every_chart_extraction_has_the_planned_fields():
    """The fields must exist on every chart. An unreadable value may be empty (see the next tests)."""
    for position, entry in enumerate(_load_chart_entries(), start=1):
        for field, aliases in CHART_FIELDS.items():
            assert _has_key(entry, aliases), f"chart {position}: no {field} field (tried {aliases})"


def test_most_charts_have_real_baseline_and_peak_numbers():
    entries = _load_chart_entries()
    with_baseline = [e for e in entries if _find(e, CHART_FIELDS["baseline"])]
    with_peak = [e for e in entries if _find(e, CHART_FIELDS["peak"])]
    assert len(with_baseline) >= 2 and len(with_peak) >= 2


def test_every_chart_has_a_trend_conclusion():
    for position, entry in enumerate(_load_chart_entries(), start=1):
        assert _find(entry, CHART_FIELDS["trend"]), f"chart {position}: empty trend conclusion"


def test_readings_are_honestly_labelled():
    """Every number says whether it was observed, estimated or uncertain; no number is silently exact."""
    for position, entry in enumerate(_load_chart_entries(), start=1):
        numbers = _numeric_readings(entry)
        for reading in numbers:
            assert reading.get("reading_type") in READING_TYPES, f"chart {position}: unlabelled number {reading}"
        for point in entry.get("data_points", []):
            if point.get("y") is None:
                assert point.get("reading_type") == "uncertain", f"chart {position}: empty value not marked uncertain"
        if numbers and not entry.get("evidence"):
            assert entry.get("uncertainties"), f"chart {position}: numbers without evidence must be flagged"


@requires_chart_reader
def test_every_chart_extraction_validates_against_the_chart_reader_schema():
    for position, entry in enumerate(_load_chart_entries(), start=1):
        chart_reader.ChartExtraction.model_validate(entry)  # raises if the saved file drifted from the schema


# ---------------------------------------------------------------------------
# Part D: extractor -> chart reader, with a fake vision model (no API call)
# ---------------------------------------------------------------------------
FAKE_READING = {
    "chart_title": "Accuracy by dataset size",
    "chart_type": "bar",
    "x_axis": {"label": "Dataset size", "unit": None},
    "y_axis": {"label": "Accuracy", "unit": "%"},
    "baseline_values": [{"label": "1", "value": 32.0, "unit": "%", "series": None,
                         "reading_type": "estimated", "note": None}],
    "peak_performance": {"label": "100", "value": 100.0, "unit": "%", "series": None,
                         "reading_type": "estimated", "note": None},
    "data_points": [],
    "series": [],
    "trend_summary": "Accuracy rises with dataset size.",
    "evidence": ["Bar heights read against the y-axis."],
    "uncertainties": [],
}


class FakeVision:
    """Stands in for the vision model and records what it was sent."""

    def __init__(self):
        self.calls = []

    def describe(self, system, prompt, image_bytes, mime):
        self.calls.append({"mime": mime, "bytes": len(image_bytes), "magic": image_bytes[:8]})
        return json.dumps(FAKE_READING)


@requires_extractor
@requires_chart_reader
def test_every_extracted_figure_is_accepted_by_the_chart_reader(three_figure_pdf, tmp_path):
    figures = extract_figures(three_figure_pdf, "paper", out_dir=tmp_path / "out")
    client = FakeVision()
    for fig in figures:
        result = chart_reader.extract_chart_information(fig["path"], client=client)
        assert result["chart_type"] == "bar"
        assert result["baseline_values"][0]["reading_type"] == "estimated"
    assert len(client.calls) == len(figures) == 3
    assert all(call["mime"] == "image/png" and call["magic"] == b"\x89PNG\r\n\x1a\n" for call in client.calls)


@requires_extractor
@requires_chart_reader
def test_figures_listed_in_the_manifest_can_all_be_read_back(three_figure_pdf, tmp_path):
    out = tmp_path / "out"
    extract_figures(three_figure_pdf, "paper", out_dir=out)
    manifest = extractor.load_extracted_figures(out, paper_id="paper")
    client = FakeVision()
    results = [chart_reader.extract_chart_information(f["path"], client=client) for f in manifest]
    assert len(results) == 3
    json.dumps(results)  # JSON-serialisable for the vector DB / hypothesis generator


@requires_extractor
@requires_chart_reader
def test_a_figure_over_the_size_limit_gives_a_typed_error_the_pipeline_can_catch(three_figure_pdf, tmp_path):
    """Photo-like figures can exceed the reader's size limit; callers must catch ImageValidationError."""
    figure = extract_figures(three_figure_pdf, "paper", out_dir=tmp_path / "out")[0]
    with pytest.raises(chart_reader.ImageValidationError, match="limit"):
        chart_reader.extract_chart_information(figure["path"], client=FakeVision(), env={"CHART_MAX_IMAGE_MB": "0.000001"})
