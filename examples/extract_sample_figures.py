"""
Week 3 demo (Member 3): download 3 real papers and extract their figures.

    python examples/extract_sample_figures.py                 # first 3 arXiv papers
    python examples/extract_sample_figures.py --limit 5 --source arxiv pubmed
    python examples/extract_sample_figures.py --pdf ~/Downloads/paper.pdf --paper-id arxiv:2610.12452v1

PDFs go to data/pdfs/, figures to data/extracted_figures/<paper_id>/fig_N.png,
and a summary of every run to data/sample_figure_extractions.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # run from anywhere

from src.ingestion.figure_extractor import DEFAULT_OUT_DIR, extract_figures, safe_folder_name

PDF_DIR = Path("data/pdfs")
SUMMARY = Path("data/sample_figure_extractions.json")
HEADERS = {"User-Agent": "AutoScientistAI/1.0 (research prototype)"}


def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF once (skips if already downloaded). Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    url = url.replace("http://arxiv.org", "https://arxiv.org")
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            logging.warning("Not a PDF (probably a login/captcha page): %s", url)
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except requests.RequestException as exc:
        logging.warning("Download failed for %s: %s", url, exc)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract figures from sample paper PDFs")
    ap.add_argument("--papers", default="data/sample_papers.json")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--source", nargs="+", default=["arxiv"], help="arxiv and/or pubmed")
    ap.add_argument("--pdf", help="use one local PDF instead of downloading")
    ap.add_argument("--paper-id", help="paper id for --pdf")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    jobs = []                                   # (paper_id, title, pdf path)
    if args.pdf:
        jobs.append((args.paper_id or Path(args.pdf).stem, "", Path(args.pdf).expanduser()))
    else:
        papers = json.loads(Path(args.papers).read_text(encoding="utf-8"))
        for p in papers:
            if len(jobs) >= args.limit:
                break
            if p.get("source") not in args.source or not p.get("pdf_url"):
                continue
            dest = PDF_DIR / f"{safe_folder_name(p['paper_id'])}.pdf"
            if download_pdf(p["pdf_url"], dest):
                jobs.append((p["paper_id"], p.get("title", ""), dest))

    summary = []
    for paper_id, title, pdf in jobs:
        figs = extract_figures(pdf, paper_id, DEFAULT_OUT_DIR)
        captioned = sum(1 for f in figs if f["caption"])
        print(f"\n{paper_id}  {title[:70]}")
        print(f"  {len(figs)} figure(s), {captioned} with caption")
        for f in figs:
            print(f"   fig_{f['index']:<3} p.{f['page']:<3} {f['kind']:6} "
                  f"{f['width']}x{f['height']}  {f['caption'][:60] or '(no caption)'}")
        summary.append({"paper_id": paper_id, "title": title, "pdf": str(pdf),
                        "figures": len(figs), "with_caption": captioned,
                        "figure_list": figs})

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary for {len(summary)} paper(s) -> {SUMMARY}")


if __name__ == "__main__":
    main()
