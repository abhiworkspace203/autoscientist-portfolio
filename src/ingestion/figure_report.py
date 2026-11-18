"""
Figure report (Member 3, Week 3).

Summarises every figure extracted so far: how many per paper, how many have a
caption, and which captioned figures are ready for the chart reader (Member 2).

    python -m src.ingestion.figure_report
    python -m src.ingestion.figure_report --json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.ingestion.figure_extractor import DEFAULT_OUT_DIR, load_extracted_figures


def summarise_figures(figures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts per paper plus overall totals."""
    papers: Dict[str, Dict[str, int]] = {}
    for fig in figures:
        row = papers.setdefault(fig.get("paper_id", "unknown"),
                                {"figures": 0, "captioned": 0, "vector": 0})
        row["figures"] += 1
        if fig.get("caption"):
            row["captioned"] += 1
        if fig.get("kind") == "vector":
            row["vector"] += 1
    total = sum(r["figures"] for r in papers.values())
    captioned = sum(r["captioned"] for r in papers.values())
    return {
        "papers": papers,
        "total_papers": len(papers),
        "total_figures": total,
        "total_captioned": captioned,
        "caption_rate": round(captioned / total, 3) if total else 0.0,
    }


def figures_for_chart_reader(figures: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Only captioned figures whose image file exists: path, caption, paper_id."""
    return [
        {"paper_id": f.get("paper_id", ""), "path": f["path"], "caption": f["caption"]}
        for f in figures
        if f.get("caption") and f.get("path") and Path(f["path"]).is_file()
    ]


def format_report(summary: Dict[str, Any]) -> str:
    lines = [f"{'Paper':<30} {'Figures':>8} {'Captioned':>10} {'Vector':>7}"]
    for pid, r in sorted(summary["papers"].items()):
        lines.append(f"{pid:<30} {r['figures']:>8} {r['captioned']:>10} {r['vector']:>7}")
    lines.append(
        f"\n{summary['total_figures']} figures from {summary['total_papers']} papers, "
        f"{summary['total_captioned']} with captions ({summary['caption_rate']:.0%})"
    )
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Summarise extracted figures")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args()

    summary = summarise_figures(load_extracted_figures(args.out))
    if not summary["total_figures"]:
        print("No figures found. Run: python examples/extract_sample_figures.py")
        return
    print(json.dumps(summary, indent=2) if args.json else format_report(summary))


if __name__ == "__main__":
    main()
