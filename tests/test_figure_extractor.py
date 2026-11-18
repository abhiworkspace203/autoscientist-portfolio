"""
Tests for the PDF figure extractor (Member 3 - Week 3).

Builds small synthetic "paper" PDFs with PyMuPDF (charts, a logo, captions above
and below), so no downloads are needed.

Run:  python -m pytest tests/test_figure_extractor.py -v
"""

import json

import pytest

from src.ingestion.figure_extractor import (
    CAPTION_RE,
    extract_figures,
    fitz,
    safe_folder_name,
)

if fitz is None:
    pytest.skip("PyMuPDF is not installed", allow_module_level=True)


def make_image(width, height, color=(0.2, 0.4, 0.8)):
    """A solid-colour RGB image of width x height pixels."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), 0)
    pix.set_rect(pix.irect, tuple(int(c * 255) for c in color))
    return pix


def add_image(page, rect, width, height, color=(0.2, 0.4, 0.8)):
    page.insert_image(fitz.Rect(rect), pixmap=make_image(width, height, color))


@pytest.fixture
def paper_pdf(tmp_path):
    """
    Page 1: journal logo (80x80 px) + Figure 1 chart (600x400 px, caption below)
    Page 2: caption ABOVE Figure 2 (500x500 px) + an uncaptioned big image
    """
    doc = fitz.open()
    p1 = doc.new_page(width=612, height=792)
    add_image(p1, (40, 30, 90, 80), 80, 80, (0.9, 0.1, 0.1))                 # logo
    p1.insert_text((72, 120), "Introduction. Deep models need lots of memory.", fontsize=10)
    add_image(p1, (72, 150, 540, 462), 600, 400)                            # Figure 1
    p1.insert_text((72, 480), "Figure 1: Accuracy versus training epochs.", fontsize=10)

    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((72, 90), "Fig. 2. Loss curves for three optimisers.", fontsize=10)
    add_image(p2, (72, 100, 400, 428), 500, 500, (0.1, 0.7, 0.3))           # Figure 2
    add_image(p2, (72, 500, 400, 700), 640, 400, (0.5, 0.5, 0.5))           # no caption
    path = tmp_path / "paper.pdf"
    doc.save(path)
    doc.close()
    return path


def test_extracts_big_figures_and_skips_logo(paper_pdf, tmp_path):
    figs = extract_figures(paper_pdf, "arxiv:1234.5678v1", out_dir=tmp_path / "out")
    assert len(figs) == 3                          # logo (80x80) filtered out
    assert all(f["width"] >= 300 and f["height"] >= 300 for f in figs)
    assert [f["page"] for f in figs] == [1, 2, 2]


def test_caption_below_is_paired(paper_pdf, tmp_path):
    fig1 = extract_figures(paper_pdf, "p1", out_dir=tmp_path / "out")[0]
    assert fig1["figure_number"] == 1
    assert fig1["caption"] == "Figure 1: Accuracy versus training epochs."


def test_caption_above_is_paired_and_uncaptioned_figure_kept(paper_pdf, tmp_path):
    figs = extract_figures(paper_pdf, "p1", out_dir=tmp_path / "out")
    assert figs[1]["figure_number"] == 2
    assert figs[1]["caption"].startswith("Fig. 2. Loss curves")
    assert figs[2]["caption"] == "" and figs[2]["figure_number"] is None


def test_files_and_manifest_are_saved(paper_pdf, tmp_path):
    out = tmp_path / "out"
    figs = extract_figures(paper_pdf, "arxiv:1234.5678v1", out_dir=out)
    folder = out / "arxiv_1234.5678v1"
    assert [p.name for p in sorted(folder.glob("fig_*.png"))] == ["fig_1.png", "fig_2.png", "fig_3.png"]
    manifest = json.loads((folder / "figures.json").read_text())
    assert manifest == figs
    saved = fitz.Pixmap(str(folder / "fig_1.png"))
    assert saved.width >= 300 and saved.height >= 300


def test_min_size_is_configurable(paper_pdf, tmp_path):
    figs = extract_figures(paper_pdf, "p1", out_dir=tmp_path / "out",
                           min_width=50, min_height=50)
    assert len(figs) == 4                          # the logo is now kept too


def test_small_on_page_but_high_resolution_image_is_kept(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    add_image(page, (72, 72, 172, 139), 900, 600)  # drawn 100x67 pt, stored 900x600 px
    pdf = tmp_path / "small_drawn.pdf"
    doc.save(pdf)
    figs = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert len(figs) == 1 and figs[0]["width"] == 900 and figs[0]["height"] == 600


def test_pdf_without_figures_returns_empty_and_writes_nothing(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Text only. Figure 1: mentioned but not drawn.")
    pdf = tmp_path / "text_only.pdf"
    doc.save(pdf)
    out = tmp_path / "out"
    assert extract_figures(pdf, "p", out_dir=out) == []
    assert not (out / "p").exists()


def test_broken_or_missing_pdf_returns_empty(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"this is not a pdf")
    assert extract_figures(bad, out_dir=tmp_path / "out") == []
    assert extract_figures(tmp_path / "missing.pdf", out_dir=tmp_path / "out") == []


def test_caption_pattern_and_safe_folder_name():
    assert CAPTION_RE.match("Figure 3: Results").group(1) == "3"
    assert CAPTION_RE.match("Fig. 12. Ablation").group(1) == "12"
    assert CAPTION_RE.match("FIG 4: x").group(1) == "4"
    assert CAPTION_RE.match("As shown in Figure 3: x") is None    # must start the block
    assert safe_folder_name("arxiv:2610.12445v1") == "arxiv_2610.12445v1"
    assert safe_folder_name("pubmed:42827724") == "pubmed_42827724"


# ------------------------------------------------ strip merging (real layout)
def test_stacked_strips_become_one_captioned_figure(tmp_path):
    """Mirrors BrickBench p.1: three full-width strips, caption under the last one."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for y0, y1 in ((164, 251), (265, 347), (379, 465)):          # 14 pt and 32 pt gaps
        add_image(page, (71, y0, 524, y1), 3840, 730)
    page.insert_text((71, 485), "Figure 1: We evaluate coding agents on LEGO design.", fontsize=9)
    pdf = tmp_path / "strips.pdf"
    doc.save(pdf)
    figs = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert len(figs) == 1
    assert figs[0]["parts"] == 3 and figs[0]["figure_number"] == 1
    assert abs(figs[0]["bbox"][1] - 164) < 2 and abs(figs[0]["bbox"][3] - 465) < 2
    assert figs[0]["height"] > figs[0]["width"] / 3            # the whole figure, not a strip


def test_thin_strips_alone_are_dropped_but_merged_ones_kept(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    add_image(page, (71, 100, 524, 120), 3840, 150)             # lone thin strip
    page.insert_text((71, 150), "A long paragraph of body text that separates the two "
                     "areas of the page so they are not merged together at all.", fontsize=6)
    for y0 in (300, 330, 360, 390):                             # four thin strips stacked
        add_image(page, (71, y0, 524, y0 + 25), 3840, 150)
    pdf = tmp_path / "thin.pdf"
    doc.save(pdf)
    figs = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert [f["parts"] for f in figs] == [4]


def test_paragraph_between_images_keeps_figures_separate(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    add_image(page, (71, 100, 524, 300), 900, 400)
    page.insert_text((71, 312), "This sentence is ordinary body text of the paper that "
                     "discusses results; it is long enough to count as a paragraph.", fontsize=6)
    add_image(page, (71, 320, 524, 520), 900, 400, (0.8, 0.2, 0.2))
    pdf = tmp_path / "two.pdf"
    doc.save(pdf)
    figs = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert [f["parts"] for f in figs] == [1, 1]


# ------------------------------------------------------- vector figures
def draw_bar_chart(page, x0, y0, x1, y1):
    page.draw_line((x0, y1), (x1, y1))                          # x axis
    page.draw_line((x0, y0), (x0, y1))                          # y axis
    width = (x1 - x0) / 7
    for i, h in enumerate((0.4, 0.7, 0.55, 0.9, 0.3, 0.65)):
        bx = x0 + (i + 0.5) * width
        page.draw_rect(fitz.Rect(bx, y1 - h * (y1 - y0), bx + width * 0.7, y1),
                       color=(0, 0, 0), fill=(0.3, 0.5, 0.8))


def test_vector_chart_is_extracted_with_caption(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((71, 80), "Results. Our method beats the baselines on every task.", fontsize=9)
    draw_bar_chart(page, 100, 120, 480, 380)
    page.insert_text((71, 400), "Figure 3: Accuracy of six models on the benchmark.", fontsize=9)
    pdf = tmp_path / "vector.pdf"
    doc.save(pdf)
    figs = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert len(figs) == 1
    fig = figs[0]
    assert fig["kind"] == "vector" and fig["figure_number"] == 3
    assert fig["width"] >= 300 and fig["height"] >= 300
    assert 115 <= fig["bbox"][1] <= 125 and 375 <= fig["bbox"][3] <= 385


def test_a_single_rule_above_a_caption_is_not_a_figure(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.draw_line((71, 380), (524, 380))                       # table / section rule
    page.insert_text((71, 400), "Figure 5: Mentioned here but drawn on another page.", fontsize=9)
    pdf = tmp_path / "rule.pdf"
    doc.save(pdf)
    assert extract_figures(pdf, "p", out_dir=tmp_path / "out") == []


def test_raster_figure_wins_over_its_own_frame(tmp_path):
    """An image with a drawn border must not be extracted twice."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    add_image(page, (100, 100, 500, 400), 800, 600)
    page.draw_rect(fitz.Rect(98, 98, 502, 402), color=(0, 0, 0))
    page.insert_text((100, 420), "Figure 1: A framed photograph.", fontsize=9)
    pdf = tmp_path / "framed.pdf"
    doc.save(pdf)
    figs = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert len(figs) == 1 and figs[0]["kind"] == "raster" and figs[0]["figure_number"] == 1


# --------------------------------------------- real-paper layouts (BrickBench)
def test_small_print_labels_between_strips_do_not_split_the_figure(tmp_path):
    """Each strip has a long 6 pt prompt under it; body text is 10 pt."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(71, 400, 524, 800),               # a real paragraph:
                        "Body text of the paper discussing the results. " * 25,   # body = 10 pt
                        fontsize=10)
    for i, y0 in enumerate((100, 160, 220)):
        add_image(page, (71, y0, 524, y0 + 40), 3840, 340, (0.2 + 0.2 * i, 0.4, 0.6))
        page.insert_text((73, y0 + 48), "Model A long prompt describing the assembly in "
                         "detail so that it is clearly longer than eighty characters.", fontsize=6)
    page.insert_text((73, 278), "GPT-6 Astra   Claude Opus 5.5   Qwen 3.8 Flash", fontsize=6)
    page.insert_text((71, 300), "Figure 4: Samples from every agent.", fontsize=10)
    pdf = tmp_path / "labels.pdf"
    doc.save(pdf)
    figs = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert len(figs) == 1
    assert figs[0]["parts"] == 3 and figs[0]["figure_number"] == 4
    assert figs[0]["bbox"][3] >= 276                     # model-name legend included


def test_appendix_and_subfigure_captions():
    assert CAPTION_RE.match("Figure A6: Additional samples").group(1) == "A6"
    assert CAPTION_RE.match("(a) (b) Figure 6: Human evaluation").group(1) == "6"


def test_short_wide_vector_chart_is_rendered_sharp_enough(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    draw_bar_chart(page, 80, 100, 520, 210)              # only 110 pt tall
    page.insert_text((71, 225), "Figure 3: Summary metrics.", fontsize=9)
    pdf = tmp_path / "wide.pdf"
    doc.save(pdf)
    figs = extract_figures(pdf, "p", out_dir=tmp_path / "out")
    assert len(figs) == 1 and figs[0]["kind"] == "vector"
    assert figs[0]["height"] >= 300                      # rendered at a higher zoom


def test_load_extracted_figures_for_the_vlm(paper_pdf, tmp_path):
    from src.ingestion.figure_extractor import load_extracted_figures
    out = tmp_path / "out"
    extract_figures(paper_pdf, "arxiv:1", out_dir=out)
    extract_figures(paper_pdf, "pubmed:2", out_dir=out)
    assert len(load_extracted_figures(out)) == 6
    one = load_extracted_figures(out, paper_id="arxiv:1")
    assert len(one) == 3 and one[0]["caption"].startswith("Figure 1")


# ------------------------------------------------------------- robustness
def test_password_protected_pdf_returns_empty(tmp_path):
    doc = fitz.open()
    add_image(doc.new_page(), (72, 72, 472, 372), 800, 600)
    pdf = tmp_path / "locked.pdf"
    doc.save(pdf, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    assert extract_figures(pdf, "p", out_dir=tmp_path / "out") == []


def test_owner_password_only_pdf_is_still_extracted(tmp_path):
    doc = fitz.open()
    add_image(doc.new_page(), (72, 72, 472, 372), 800, 600)
    pdf = tmp_path / "owner_only.pdf"
    doc.save(pdf, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner")   # no user password
    assert len(extract_figures(pdf, "p", out_dir=tmp_path / "out")) == 1


def test_rerun_with_fewer_figures_removes_stale_files(paper_pdf, tmp_path):
    out = tmp_path / "out"
    assert len(extract_figures(paper_pdf, "p", out_dir=out)) == 3
    doc = fitz.open()
    add_image(doc.new_page(), (72, 72, 472, 372), 800, 600)
    smaller = tmp_path / "smaller.pdf"
    doc.save(smaller)
    figs = extract_figures(smaller, "p", out_dir=out)              # same paper id, 1 figure
    assert len(figs) == 1
    assert sorted(p.name for p in (out / "p").glob("fig_*.png")) == ["fig_1.png"]
    assert len(json.loads((out / "p" / "figures.json").read_text())) == 1


def test_rerun_with_no_figures_clears_old_output(paper_pdf, tmp_path):
    out = tmp_path / "out"
    extract_figures(paper_pdf, "p", out_dir=out)
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Text only.")
    text_only = tmp_path / "text_only.pdf"
    doc.save(text_only)
    assert extract_figures(text_only, "p", out_dir=out) == []
    assert list((out / "p").glob("fig_*.png")) == []
    assert not (out / "p" / "figures.json").exists()


def test_requirements_list_one_package_per_line():
    from pathlib import Path
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    if not req.exists():
        pytest.skip("requirements.txt not found")
    for line in req.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert line.count(">=") <= 1 and line.count("==") <= 1, f"two packages on one line: {line}"
