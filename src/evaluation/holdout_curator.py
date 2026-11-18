"""
Holdout paper curator (Member 3, Week 4).

Builds data/holdout_papers/: a small, fixed set of recent human-written papers
(2024-2026) from three domains (AI, computational biology, physics). The agent's
generated paper is later graded against them by benchmark_evaluator.py.

Selection rules (one paper per slot in HOLDOUT_PLAN):
  - arXiv category and month fixed per slot, so the set covers 2024, 2025, 2026
  - abstract of at least MIN_ABSTRACT_WORDS words, PDF available
  - the PDF text must be readable and contain an Introduction and References
  - no duplicates

    python -m src.evaluation.holdout_curator
    python -m src.evaluation.holdout_curator --no-download
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.evaluation.paper_structure import analyse_structure, pdf_text

HOLDOUT_DIR = Path("data/holdout_papers")
METADATA_FILE = "benchmark_metadata.json"
MIN_ABSTRACT_WORDS = 120
CANDIDATES_PER_SLOT = 15


@dataclass(frozen=True)
class Slot:
    domain: str
    category: str
    year: int
    month: int


HOLDOUT_PLAN: List[Slot] = [
    Slot("AI", "cs.LG", 2024, 6),
    Slot("AI", "cs.AI", 2025, 3),
    Slot("AI", "cs.CL", 2026, 5),
    Slot("Computational Biology", "q-bio.BM", 2024, 9),
    Slot("Computational Biology", "q-bio.QM", 2025, 6),
    Slot("Computational Biology", "q-bio.GN", 2026, 3),
    Slot("Physics", "physics.comp-ph", 2024, 11),
    Slot("Physics", "cond-mat.mtrl-sci", 2025, 9),
    Slot("Physics", "physics.chem-ph", 2026, 7),
]


def slot_query(slot: Slot) -> str:
    start = f"{slot.year}{slot.month:02d}010000"
    end = f"{slot.year}{slot.month:02d}282359"
    return f"cat:{slot.category} AND submittedDate:[{start} TO {end}]"


def safe_name(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)


def result_to_record(result: Any, slot: Slot) -> Dict[str, Any]:
    short_id = result.get_short_id()
    return {
        "paper_id": f"arxiv:{short_id}",
        "title": " ".join(result.title.split()),
        "authors": [a.name for a in result.authors],
        "published_date": result.published.date().isoformat(),
        "domain": slot.domain,
        "category": slot.category,
        "url": result.entry_id,
        "pdf_url": result.pdf_url,
        "abstract": " ".join(result.summary.split()),
    }


def passes_metadata_rules(record: Dict[str, Any], chosen_ids: Iterable[str]) -> bool:
    year = int(record["published_date"][:4])
    return (
        2024 <= year <= 2026
        and record["paper_id"] not in set(chosen_ids)
        and bool(record.get("pdf_url"))
        and len(record["abstract"].split()) >= MIN_ABSTRACT_WORDS
    )


def passes_structure_rules(structure: Dict[str, Any]) -> bool:
    return "introduction" in structure["sections"] and "references" in structure["sections"]


def download_pdf(url: str, path: Path, timeout: int = 60) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "zaalima-holdout-curator"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF"):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return True
    except Exception:
        return False


def search_slot(slot: Slot, client: Any = None) -> List[Any]:
    import arxiv

    client = client or arxiv.Client(page_size=CANDIDATES_PER_SLOT, delay_seconds=3, num_retries=3)
    search = arxiv.Search(query=slot_query(slot), max_results=CANDIDATES_PER_SLOT,
                          sort_by=arxiv.SortCriterion.Relevance)
    return list(client.results(search))


def curate(out_dir: Path = HOLDOUT_DIR, plan: Optional[List[Slot]] = None, download: bool = True,
           search=search_slot, fetch=download_pdf, pause: float = 1.0) -> Dict[str, Any]:
    """Pick one paper per slot and write benchmark_metadata.json."""
    plan = plan or HOLDOUT_PLAN
    out_dir = Path(out_dir)
    papers: List[Dict[str, Any]] = []
    for slot in plan:
        picked = None
        for result in search(slot):
            record = result_to_record(result, slot)
            if not passes_metadata_rules(record, [p["paper_id"] for p in papers]):
                continue
            if download:
                pdf_path = out_dir / "pdfs" / f"{safe_name(record['paper_id'])}.pdf"
                if not (pdf_path.is_file() or fetch(record["pdf_url"], pdf_path)):
                    continue
                structure = analyse_structure(pdf_text(str(pdf_path)))
                if not passes_structure_rules(structure):
                    continue
                record["pdf_path"] = pdf_path.as_posix()
                record["structure"] = structure
            picked = record
            break
        if picked:
            papers.append(picked)
            print(f"[{slot.domain} {slot.year}] {picked['paper_id']}  {picked['title'][:70]}")
        else:
            print(f"[{slot.domain} {slot.year}] no paper met the rules ({slot.category})")
        time.sleep(pause)

    metadata = {
        "description": "Holdout set of recent human-authored papers for benchmarking the agent",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_rules": {
            "years": "2024-2026",
            "domains": sorted({s.domain for s in plan}),
            "min_abstract_words": MIN_ABSTRACT_WORDS,
            "requires": ["pdf", "introduction", "references"] if download else ["pdf_url"],
            "plan": [{"domain": s.domain, "category": s.category, "year": s.year, "month": s.month}
                     for s in plan],
        },
        "count": len(papers),
        "papers": papers,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return metadata


def load_holdout(out_dir: Path = HOLDOUT_DIR) -> List[Dict[str, Any]]:
    path = Path(out_dir) / METADATA_FILE
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("papers", [])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Curate the holdout set of human-written papers")
    ap.add_argument("--out", default=str(HOLDOUT_DIR))
    ap.add_argument("--no-download", action="store_true", help="metadata only, skip PDFs")
    args = ap.parse_args()
    meta = curate(Path(args.out), download=not args.no_download)
    domains: Dict[str, int] = {}
    for p in meta["papers"]:
        domains[p["domain"]] = domains.get(p["domain"], 0) + 1
    print(f"\n{meta['count']} holdout papers saved to {Path(args.out) / METADATA_FILE}")
    for d, n in sorted(domains.items()):
        print(f"  {d}: {n}")


if __name__ == "__main__":
    main()
