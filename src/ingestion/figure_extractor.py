"""
PDF Figure Extractor (Member 3 - Week 3: "Giving the AI Eyes").

Finds the figures (charts, plots, diagrams) inside a paper PDF, drops logos and
tiny icons, pairs each figure with its "Figure X:" / "Fig. X." caption, and saves
it as a PNG for the Vision-Language Model chart reader (Member 2):

    data/extracted_figures/<paper_id>/fig_1.png
    data/extracted_figures/<paper_id>/fig_2.png
    data/extracted_figures/<paper_id>/figures.json   <- captions, pages, sizes

Usage (from repo root):
    python -m src.ingestion.figure_extractor path/to/paper.pdf --paper-id arxiv:2610.12445v1

    from src.ingestion.figure_extractor import extract_figures
    figures = extract_figures("paper.pdf", paper_id="arxiv:2610.12445v1")
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:                                  # PyMuPDF >= 1.24 prefers `import pymupdf`
    import pymupdf as fitz
except ImportError:                   # older releases only provide `fitz`
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("data/extracted_figures")
MIN_WIDTH = 300                       # pixels - smaller images are logos / icons / headshots
MIN_HEIGHT = 300
RENDER_ZOOM = 2.0                     # render figures at 2x (144 dpi) for the VLM
CAPTION_MAX_GAP = 120                 # points between a figure and its caption
STRIP_MAX_GAP = 40                    # points between image strips of ONE figure
PARAGRAPH_MIN_CHARS = 80              # text this long between images splits a figure ...
LABEL_FONT_RATIO = 0.85               # ... unless it is small print (a label inside the figure)
MAX_ZOOM = 4.0                        # cap when rendering high-resolution figures
VECTOR_MIN_ITEMS = 5                  # a vector chart has many drawn lines / shapes
VECTOR_MIN_SIZE = 100                 # points; smaller drawings are rules, not charts
VECTOR_MAX_GAP = 30                   # points between drawing pieces of one chart

# "Figure 3:", "Fig. 3.", "Figure A6:" (appendix), "(a) (b) Figure 6:" (sub-figure labels first)
CAPTION_RE = re.compile(r"^\s*(?:\([a-z]\)\s*)*(?:Figure|Fig\.?)\s*([A-Z]?\d+)\s*[:.]",
                        re.IGNORECASE)

Rect = Tuple[float, float, float, float]


@dataclass
class Figure:
    """One extracted figure."""
    paper_id: str
    index: int                        # fig_<index>.png, in reading order
    page: int                         # 1-based page number
    path: str                         # saved PNG
    width: int                        # saved image size in pixels
    height: int
    bbox: List[float]                 # position on the page (points): x0, y0, x1, y1
    caption: str = ""
    figure_number: Optional[Any] = None   # the "X" in "Figure X:" (int, or e.g. "A6")
    kind: str = "raster"                  # "raster" (embedded images) | "vector" (drawn)
    parts: int = 1                        # image strips merged into this figure


# ------------------------------------------------------------------ helpers
def safe_folder_name(paper_id: str) -> str:
    """'arxiv:2610.12445v1' -> 'arxiv_2610.12445v1' (safe on every OS)."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(paper_id)).strip("._")
    return name or "paper"


def clean_caption(text: str) -> str:
    return " ".join((text or "").split())


def find_captions(page: Any) -> List[Tuple[Rect, str, int]]:
    """Text blocks on the page that start with 'Figure X:' / 'Fig. X.'."""
    captions = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[:5]
        text = clean_caption(text)
        match = CAPTION_RE.match(text)
        if match:
            number = match.group(1)
            captions.append(((x0, y0, x1, y1), text,
                             int(number) if number.isdigit() else number.upper()))
    return captions


def horizontal_overlap(a: Rect, b: Rect) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def pair_caption(bbox: Rect, captions: List[Tuple[Rect, str, int]],
                 used: set) -> Optional[Tuple[Rect, str, int]]:
    """
    The caption closest to the figure, preferring one just BELOW it (the usual
    layout), then one just ABOVE it. Captions must overlap the figure horizontally.
    """
    best, best_gap = None, None
    for cap in captions:
        if id(cap) in used:
            continue
        crect = cap[0]
        if horizontal_overlap(bbox, crect) <= 0:
            continue
        below_gap = crect[1] - bbox[3]                 # caption top - figure bottom
        above_gap = bbox[1] - crect[3]                 # figure top - caption bottom
        if -5 <= below_gap <= CAPTION_MAX_GAP:
            gap = below_gap
        elif -5 <= above_gap <= CAPTION_MAX_GAP:
            gap = above_gap + CAPTION_MAX_GAP          # rank captions above as 2nd choice
        else:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = cap, gap
    return best


def image_placements(page: Any) -> List[Dict[str, Any]]:
    """Raster images drawn on the page, with native pixel size and page position."""
    placements = []
    for info in page.get_image_info(xrefs=True):
        bbox = tuple(info.get("bbox") or ())
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        placements.append({
            "xref": info.get("xref", 0),
            "bbox": bbox,
            "width": int(info.get("width", 0)),         # native pixels of the image
            "height": int(info.get("height", 0)),
        })
    placements.sort(key=lambda p: (p["bbox"][1], p["bbox"][0]))   # reading order
    return placements


def is_big_enough(width: int, height: int,
                  min_width: int = MIN_WIDTH, min_height: int = MIN_HEIGHT) -> bool:
    return width >= min_width and height >= min_height


def union(rects: List[Rect]) -> Rect:
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


def body_font_size(doc: Any) -> float:
    """Most common font size in the whole document (by characters) = body text size."""
    sizes: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes[round(span.get("size", 0), 1)] += len(span.get("text", "").strip())
    sizes.pop(0.0, None)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def text_blocks(page: Any) -> List[Tuple[Rect, str, float]]:
    """(bbox, text, average font size) for each text block on the page."""
    blocks = []
    for block in page.get_text("dict").get("blocks", []):
        spans = [sp for ln in block.get("lines", []) for sp in ln.get("spans", [])]
        text = clean_caption(" ".join(sp.get("text", "") for sp in spans))
        if not text:
            continue
        chars = sum(len(sp.get("text", "")) for sp in spans) or 1
        size = sum(sp.get("size", 0) * len(sp.get("text", "")) for sp in spans) / chars
        blocks.append((tuple(block["bbox"]), text, size))
    return blocks


def splits_figure(blocks: List[Tuple[Rect, str, float]], upper: Rect, lower: Rect,
                  body_size: float = 10.0) -> bool:
    """
    True if a caption or a paragraph of BODY text sits between two images.
    Small-print lines (row labels, legends, prompts under image strips) belong to
    the figure and do not split it.
    """
    top, bottom = upper[3], lower[1]
    span = union([upper, lower])
    for rect, text, size in blocks:
        mid = (rect[1] + rect[3]) / 2
        if not (top - 2 <= mid <= bottom + 2) or horizontal_overlap(rect, span) <= 0:
            continue
        if CAPTION_RE.match(text):
            return True
        if len(text) >= PARAGRAPH_MIN_CHARS and size >= LABEL_FONT_RATIO * body_size:
            return True
    return False


def group_strips(placements: List[Dict[str, Any]], blocks: List[Tuple[Rect, str, float]],
                 body_size: float = 10.0) -> List[List[Dict[str, Any]]]:
    """
    Merge images stacked vertically in the same column (small gap, no caption or
    paragraph between) into one figure. Papers often store one figure as several
    horizontal image strips.
    """
    groups: List[List[Dict[str, Any]]] = []
    for img in placements:                                   # already in reading order
        b = img["bbox"]
        for group in reversed(groups):
            last = group[-1]["bbox"]
            narrower = min(b[2] - b[0], last[2] - last[0])
            same_column = horizontal_overlap(b, last) >= 0.5 * narrower
            gap = b[1] - last[3]
            if same_column and -5 <= gap <= STRIP_MAX_GAP and not splits_figure(blocks, last, b, body_size):
                group.append(img)
                break
        else:
            groups.append([img])
    return groups


def include_labels(bbox: Rect, blocks: List[Tuple[Rect, str, float]],
                   body_size: float = 10.0) -> Rect:
    """
    Grow a figure's box to take in small-print text right above/below it (row
    labels, legends, model names) - part of the figure, useful for the VLM.
    Captions and body-size text are never included.
    """
    changed = True
    while changed:
        changed = False
        for rect, text, size in blocks:
            if CAPTION_RE.match(text) or size >= LABEL_FONT_RATIO * body_size:
                continue
            if horizontal_overlap(rect, bbox) <= 0 or (rect[0] < bbox[0] - 5 or rect[2] > bbox[2] + 5):
                continue
            below = 0 <= rect[1] - bbox[3] <= STRIP_MAX_GAP / 2
            above = 0 <= bbox[1] - rect[3] <= STRIP_MAX_GAP / 2
            if below or above:
                bbox = union([bbox, rect])
                changed = True
    return bbox


def vector_figure_region(page: Any, caption_rect: Rect,
                         taken: List[Rect]) -> Optional[Rect]:
    """
    Area of a chart drawn with vector graphics (lines, curves, shapes) just above
    or below a caption. Returns None for a few stray lines (table rules, borders).
    """
    try:
        # lines have zero width or height: give every piece at least 1 pt
        drawings = [(r[0] - 0.5, r[1] - 0.5, r[2] + 0.5, r[3] + 0.5)
                    for r in (tuple(d["rect"]) for d in page.get_drawings())]
    except Exception:
        return None
    drawings = [r for r in drawings
                if not any(horizontal_overlap(r, t) > 0 and r[1] >= t[1] - 1
                            and r[3] <= t[3] + 1 for t in taken)]     # not inside a figure

    def grow(candidates: List[Rect], upward: bool) -> List[Rect]:
        chosen: List[Rect] = []
        edge = caption_rect[1] if upward else caption_rect[3]
        ordered = sorted(candidates, key=lambda r: -r[3] if upward else r[1])
        for r in ordered:
            if horizontal_overlap(r, caption_rect) <= 0 and not (
                    chosen and horizontal_overlap(r, union(chosen)) > 0):
                continue
            gap = (edge - r[3]) if upward else (r[1] - edge)
            if gap > VECTOR_MAX_GAP:
                break
            chosen.append(r)
            edge = min(edge, r[1]) if upward else max(edge, r[3])
        return chosen

    above = grow([r for r in drawings if r[3] <= caption_rect[1] + 2], upward=True)
    below = grow([r for r in drawings if r[1] >= caption_rect[3] - 2], upward=False)
    for chosen in (above, below):
        if len(chosen) >= VECTOR_MIN_ITEMS:
            region = union(chosen)
            if (region[2] - region[0] >= VECTOR_MIN_SIZE
                    and region[3] - region[1] >= VECTOR_MIN_SIZE):
                return region
    return None


def render(page: Any, rect: Rect, zoom: float) -> Any:
    clip = fitz.Rect(rect) & page.rect
    if clip.is_empty:
        return None
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)


# ---------------------------------------------------------------- main API
def extract_figures(
    pdf_path: str | Path,
    paper_id: Optional[str] = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
    zoom: float = RENDER_ZOOM,
) -> List[Dict[str, Any]]:
    """
    Extract the figures from one PDF.

    Returns a list of dicts (paper_id, index, page, path, width, height, bbox,
    caption, figure_number) and writes the PNGs plus a figures.json manifest to
    <out_dir>/<paper_id>/.
    """
    pdf_path = Path(pdf_path)
    paper_id = paper_id or pdf_path.stem
    target = Path(out_dir) / safe_folder_name(paper_id)

    figures: List[Figure] = []
    seen_xrefs: Dict[int, int] = {}          # image xref -> first page it appeared on
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Cannot open %s: %s", pdf_path, exc)
        return []

    # Encrypted PDFs: many only have an owner password and open with an empty
    # user password; if a real password is needed we cannot read the pages.
    if doc.needs_pass and not doc.authenticate(""):
        logger.warning("Skipping %s: PDF is password-protected", pdf_path)
        doc.close()
        return []

    # A re-run must not leave figures from the previous run behind
    # (e.g. old fig_7.png when the paper now yields only 5 figures).
    if target.is_dir():
        for old in [*target.glob("fig_*.png"), *target.glob("tmp_*.png"),
                    target / "figures.json"]:
            old.unlink(missing_ok=True)

    with doc:
        body_size = body_font_size(doc)
        for page_number, page in enumerate(doc, start=1):
            captions = find_captions(page)
            blocks = text_blocks(page)
            used_captions: set = set()
            taken: List[Rect] = []

            def save(pix: Any, bbox: Rect, cap: Any, kind: str, parts: int) -> None:
                index = len(figures) + 1
                target.mkdir(parents=True, exist_ok=True)
                path = target / f"fig_{index}.png"
                pix.save(str(path))
                if cap:
                    used_captions.add(id(cap))
                taken.append(bbox)
                figures.append(Figure(
                    paper_id=paper_id, index=index, page=page_number, path=str(path),
                    width=pix.width, height=pix.height,
                    bbox=[round(v, 1) for v in bbox],
                    caption=cap[1] if cap else "",
                    figure_number=cap[2] if cap else None,
                    kind=kind, parts=parts,
                ))

            # --- A. raster figures (embedded images), strips merged ---------------
            candidates = []
            for img in image_placements(page):
                # logos / icons / headshots: small on BOTH sides. Strips are long on one.
                if max(img["width"], img["height"]) < max(min_width, min_height):
                    continue
                # the same image (e.g. a journal logo) repeated on later pages
                if img["xref"] and seen_xrefs.get(img["xref"], page_number) != page_number:
                    continue
                seen_xrefs.setdefault(img["xref"], page_number)
                candidates.append(img)

            for group in group_strips(candidates, blocks, body_size):
                bbox = include_labels(union([g["bbox"] for g in group]), blocks, body_size)
                # render at (close to) the images' own resolution, capped at MAX_ZOOM
                native = max(g["width"] / max(g["bbox"][2] - g["bbox"][0], 1) for g in group)
                pix = render(page, bbox, max(zoom, min(MAX_ZOOM, native)))
                if pix is None:
                    continue
                if not is_big_enough(pix.width, pix.height, min_width, min_height):
                    if len(group) != 1:
                        continue
                    # drawn small on the page but stored large: keep native resolution
                    try:
                        pix = fitz.Pixmap(doc, group[0]["xref"])
                        if pix.n - pix.alpha >= 4:          # CMYK -> RGB for PNG
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                    except Exception:
                        continue
                    if not is_big_enough(pix.width, pix.height, min_width, min_height):
                        continue
                save(pix, bbox, pair_caption(bbox, captions, used_captions), "raster", len(group))

            # --- B. vector figures: a "Figure X:" caption with no image -----------
            for cap in captions:
                if id(cap) in used_captions:
                    continue
                region = vector_figure_region(page, cap[0], taken)
                if region is None:
                    continue
                # short, wide charts: render sharper (up to MAX_ZOOM) to reach min size
                w, h = region[2] - region[0], region[3] - region[1]
                vzoom = min(MAX_ZOOM, max(zoom, min_width / max(w, 1), min_height / max(h, 1)))
                pix = render(page, region, vzoom)
                if pix is None or not is_big_enough(pix.width, pix.height, min_width, min_height):
                    continue
                save(pix, region, cap, "vector", 1)

        figures.sort(key=lambda f: (f.page, f.bbox[1], f.bbox[0]))
        for new_index, fig in enumerate(figures, start=1):        # reading order names
            if fig.index != new_index:
                tmp = Path(fig.path).with_name(f"tmp_{new_index}.png")
                Path(fig.path).rename(tmp)
                fig.path, fig.index = str(tmp), new_index
        for fig in figures:
            final = Path(fig.path).with_name(f"fig_{fig.index}.png")
            if Path(fig.path) != final:
                Path(fig.path).rename(final)
                fig.path = str(final)

    result = [asdict(f) for f in figures]
    if result:
        (target / "figures.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("%s: %d figure(s) extracted -> %s", paper_id, len(result), target)
    return result


def load_extracted_figures(out_dir: str | Path = DEFAULT_OUT_DIR,
                           paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    All figures saved so far (from every <paper>/figures.json), for the VLM chart
    reader (Member 2). Each item has: paper_id, index, page, path, width, height,
    bbox, caption, figure_number, kind, parts.

        for fig in load_extracted_figures():
            result = read_chart(fig["path"], caption=fig["caption"])
    """
    root = Path(out_dir)
    folders = [root / safe_folder_name(paper_id)] if paper_id else sorted(root.glob("*"))
    figures: List[Dict[str, Any]] = []
    for folder in folders:
        manifest = folder / "figures.json"
        if manifest.is_file():
            figures.extend(json.loads(manifest.read_text(encoding="utf-8")))
    return figures


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Extract figures + captions from a paper PDF")
    ap.add_argument("pdf", help="path to a PDF file")
    ap.add_argument("--paper-id", default=None, help="e.g. arxiv:2610.12445v1")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--min-size", type=int, default=MIN_WIDTH,
                    help="minimum width and height in pixels (default 300)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    figs = extract_figures(args.pdf, args.paper_id, args.out, args.min_size, args.min_size)
    for f in figs:
        caption = f["caption"][:80] or "(no caption found)"
        print(f"fig_{f['index']}  page {f['page']}  {f['kind']:6} {f['width']}x{f['height']}"
              f"  parts={f['parts']}  {caption}")
    if not figs:
        print("No figures found.")


if __name__ == "__main__":
    main()
