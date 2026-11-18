"""
arXiv ingestion pipeline (Member 3 — Week 1, Task 2).

Queries arXiv by category and submission-date range using the official
`arxiv` Python client and returns papers in the unified `Paper` schema.

Usage (from repo root):
    python -m src.ingestion.arxiv_scraper --categories cs.AI q-bio.BM --days 7 --max 5
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List, Optional

import arxiv

from src.ingestion.schema import Paper, clean_text, deduplicate, save_papers

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "q-bio.BM"]


def build_query(
    categories: Iterable[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    keywords: Optional[str] = None,
) -> str:
    """
    Build an arXiv API query string, e.g.
    (cat:cs.AI OR cat:q-bio.BM) AND submittedDate:[202610010000 TO 202610112359]
    """
    cats = list(categories)
    if not cats:
        raise ValueError("At least one arXiv category is required")
    parts = ["(" + " OR ".join(f"cat:{c}" for c in cats) + ")"]
    if start_date or end_date:
        start = (start_date or date(1991, 1, 1)).strftime("%Y%m%d") + "0000"
        end = (end_date or datetime.now(timezone.utc).date()).strftime("%Y%m%d") + "2359"
        parts.append(f"submittedDate:[{start} TO {end}]")
    if keywords:
        parts.append(f'all:"{keywords}"')
    return " AND ".join(parts)


def result_to_paper(result: "arxiv.Result") -> Paper:
    """Convert one arxiv.Result into the unified Paper schema."""
    short_id = result.get_short_id()                       # e.g. 2410.01234v1
    return Paper(
        paper_id=f"arxiv:{short_id}",
        source="arxiv",
        source_id=short_id,
        title=clean_text(result.title),
        abstract=clean_text(result.summary),
        authors=[a.name for a in (result.authors or [])],
        published_date=result.published.date().isoformat(),
        url=result.entry_id,
        pdf_url=result.pdf_url,
        doi=result.doi or None,
        categories=list(result.categories or []),
        journal=result.journal_ref or None,
    )


class ArxivScraper:
    def __init__(self, page_size: int = 50, delay_seconds: float = 3.0, num_retries: int = 3):
        # arXiv asks for >= 3 s between requests; the client enforces it.
        self.client = arxiv.Client(
            page_size=page_size, delay_seconds=delay_seconds, num_retries=num_retries
        )

    def search(
        self,
        categories: Iterable[str] = DEFAULT_CATEGORIES,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        max_results: int = 20,
        keywords: Optional[str] = None,
    ) -> List[Paper]:
        query = build_query(categories, start_date, end_date, keywords)
        logger.info("arXiv query: %s (max %d)", query, max_results)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        papers: List[Paper] = []
        try:
            for result in self.client.results(search):
                try:
                    paper = result_to_paper(result)
                except Exception as exc:                      # one bad record ≠ crash
                    logger.warning("Skipping malformed arXiv record: %s", exc)
                    continue
                if paper.is_valid():
                    papers.append(paper)
        except (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError) as exc:
            logger.error("arXiv request failed: %s", exc)
        except Exception as exc:                              # timeouts, DNS, etc.
            logger.error("arXiv network error: %s", exc)
        papers = deduplicate(papers)
        logger.info("arXiv returned %d papers", len(papers))
        return papers


def fetch_recent_arxiv(
    categories: Iterable[str] = DEFAULT_CATEGORIES, days: int = 7, max_results: int = 20
) -> List[Paper]:
    """Convenience wrapper: papers from the last `days` days."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return ArxivScraper().search(categories, start, end, max_results)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape recent arXiv papers")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--keywords", default=None)
    ap.add_argument("--out", default="data/arxiv_papers.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    end = datetime.now(timezone.utc).date()
    papers = ArxivScraper().search(
        args.categories, end - timedelta(days=args.days), end, args.max, args.keywords
    )
    path = save_papers(papers, args.out)
    print(f"Saved {len(papers)} arXiv papers -> {path}")


if __name__ == "__main__":
    main()
