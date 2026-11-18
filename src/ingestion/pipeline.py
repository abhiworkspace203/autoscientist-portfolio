"""
Combined ingestion pipeline: arXiv + PubMed -> one de-duplicated JSON file.

This is what produces the Week 1 deliverable data/sample_scraped_papers.json
(5 arXiv + 5 PubMed) and what Member 4 will call to feed the vector DB.

Usage (from repo root):
    # one-off sample (Week 1 deliverable)
    python -m src.ingestion.pipeline --arxiv-max 5 --pubmed-max 5

    # incremental: only papers published since the last run and not seen before
    python -m src.ingestion.pipeline --incremental --out data/new_papers.json

Every run also writes a timestamped snapshot to data/raw/ (disable with --no-snapshot),
e.g. data/raw/papers_2026-10-13_183000.json, so there is a history of each run.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from src.ingestion.arxiv_scraper import ArxivScraper
from src.ingestion.pubmed_scraper import PubMedScraper
from src.ingestion.schema import Paper, deduplicate, save_papers
from src.ingestion.state import DEFAULT_STATE_PATH, IngestionState

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")


def resolve_start_date(today: date, days: int, state: Optional[IngestionState]) -> date:
    """Incremental runs start from the last run date; otherwise `days` back."""
    if state is not None and state.last_run is not None:
        return state.last_run
    return today - timedelta(days=days)


def snapshot_path(raw_dir: Path | str, when: datetime) -> Path:
    """data/raw/papers_YYYY-MM-DD_HHMMSS.json (UTC) for one pipeline run."""
    return Path(raw_dir) / f"papers_{when.strftime('%Y-%m-%d_%H%M%S')}.json"


def save_snapshot(papers: List[Paper], raw_dir: Path | str = DEFAULT_RAW_DIR,
                  when: Optional[datetime] = None) -> Path:
    """Save this run's papers to a new dated file; never overwrites an older run."""
    when = when or datetime.now(timezone.utc)
    path = snapshot_path(raw_dir, when)
    n = 1
    while path.exists():                       # two runs in the same second
        path = path.with_name(f"{path.stem}_{n}.json")
        n += 1
    save_papers(papers, path)
    logger.info("Snapshot: %d papers -> %s", len(papers), path)
    return path


def run_ingestion(
    arxiv_categories: List[str],
    pubmed_query: str,
    days: int = 30,
    arxiv_max: int = 5,
    pubmed_max: int = 5,
    incremental: bool = False,
    state_path: Path | str = DEFAULT_STATE_PATH,
) -> List[Paper]:
    today = datetime.now(timezone.utc).date()
    state = IngestionState.load(state_path) if incremental else None
    start = resolve_start_date(today, days, state)
    logger.info("Fetching papers from %s to %s", start, today)

    arxiv_papers = ArxivScraper().search(arxiv_categories, start, today, arxiv_max)
    pubmed_papers = PubMedScraper().search(pubmed_query, start, today, pubmed_max)
    papers = deduplicate(arxiv_papers + pubmed_papers)

    if state is not None:
        before = len(papers)
        papers = state.filter_new(papers)
        state.mark_seen(papers, today)
        state.save(state_path)
        logger.info("Incremental: %d fetched, %d new (state -> %s)",
                    before, len(papers), state_path)

    logger.info("Ingestion done: %d arXiv + %d PubMed = %d unique",
                len(arxiv_papers), len(pubmed_papers), len(papers))
    return papers


def main() -> None:
    ap = argparse.ArgumentParser(description="Run arXiv + PubMed ingestion")
    ap.add_argument("--categories", nargs="+", default=["cs.AI", "q-bio.BM"])
    ap.add_argument("--query", default="machine learning drug discovery")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--arxiv-max", type=int, default=5)
    ap.add_argument("--pubmed-max", type=int, default=5)
    ap.add_argument("--incremental", action="store_true",
                    help="only return papers not seen in earlier runs")
    ap.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    ap.add_argument("--out", default="data/sample_scraped_papers.json")
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                    help="folder for dated snapshots of each run")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="do not write a dated snapshot to --raw-dir")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    papers = run_ingestion(args.categories, args.query, args.days,
                           args.arxiv_max, args.pubmed_max,
                           incremental=args.incremental, state_path=args.state)
    path = save_papers(papers, args.out)
    print(f"Saved {len(papers)} papers -> {path}")
    if not args.no_snapshot:
        snap = save_snapshot(papers, args.raw_dir)
        print(f"Snapshot          -> {snap}")


if __name__ == "__main__":
    main()
