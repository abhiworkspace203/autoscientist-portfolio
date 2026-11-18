"""
PubMed ingestion pipeline (Member 3 — Week 1, Task 2).

Uses NCBI Entrez E-utilities through Biopython:
  1. Entrez.esearch  -> list of PMIDs matching a query + date range
  2. Entrez.efetch   -> full XML records for those PMIDs
and converts each record into the unified `Paper` schema.

NCBI rules: always send an email; max 3 requests/s without an API key
(10/s with one). Put both in a .env file — never hard-code or commit them.

Usage (from repo root):
    python -m src.ingestion.pubmed_scraper --query "protein folding deep learning" --days 30 --max 5
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from Bio import Entrez

from src.ingestion.schema import Paper, clean_text, deduplicate, save_papers

logger = logging.getLogger(__name__)

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _configure_entrez(email: Optional[str], api_key: Optional[str]) -> None:
    try:
        from dotenv import load_dotenv      # optional dependency
        load_dotenv()
    except ImportError:
        pass
    Entrez.email = email or os.getenv("NCBI_EMAIL") or "your.email@example.com"
    key = api_key or os.getenv("NCBI_API_KEY")
    if key:
        Entrez.api_key = key
    Entrez.tool = "AutoScientistAI"


# ---------------------------------------------------------------- parsing ---
def _parse_date(article: Dict[str, Any]) -> str:
    """Best-effort ISO date: ArticleDate -> Journal PubDate -> MedlineDate."""
    art = article["MedlineCitation"]["Article"]
    for d in art.get("ArticleDate", []):
        try:
            return date(int(d["Year"]), int(d["Month"]), int(d["Day"])).isoformat()
        except (KeyError, ValueError):
            pass
    pub = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    year = pub.get("Year") or (pub.get("MedlineDate", "")[:4])
    if not year:
        return ""
    month_raw = str(pub.get("Month", "1"))
    month = MONTHS.get(month_raw[:3], int(month_raw) if month_raw.isdigit() else 1)
    day = int(pub.get("Day", 1)) if str(pub.get("Day", "1")).isdigit() else 1
    return date(int(year), month, day).isoformat()


def _parse_abstract(art: Dict[str, Any]) -> str:
    """Join structured abstract sections, keeping their labels (BACKGROUND: …)."""
    sections = art.get("Abstract", {}).get("AbstractText", [])
    parts = []
    for sec in sections:
        label = getattr(sec, "attributes", {}).get("Label")
        text = clean_text(str(sec))
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def _parse_authors(art: Dict[str, Any]) -> List[str]:
    authors = []
    for a in art.get("AuthorList", []):
        if "CollectiveName" in a:
            authors.append(str(a["CollectiveName"]))
        elif "LastName" in a:
            authors.append(f"{a.get('ForeName', '')} {a['LastName']}".strip())
    return authors


def _parse_doi(article: Dict[str, Any]) -> Optional[str]:
    for aid in article.get("PubmedData", {}).get("ArticleIdList", []):
        if getattr(aid, "attributes", {}).get("IdType") == "doi":
            return str(aid)
    return None


def _parse_pmc_pdf(article: Dict[str, Any]) -> Optional[str]:
    """Free full text exists only if the paper has a PMC id."""
    for aid in article.get("PubmedData", {}).get("ArticleIdList", []):
        if getattr(aid, "attributes", {}).get("IdType") == "pmc":
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{aid}/pdf/"
    return None


def record_to_paper(article: Dict[str, Any]) -> Paper:
    cit = article["MedlineCitation"]
    art = cit["Article"]
    pmid = str(cit["PMID"])
    mesh = [str(m["DescriptorName"]) for m in cit.get("MeshHeadingList", [])]
    return Paper(
        paper_id=f"pubmed:{pmid}",
        source="pubmed",
        source_id=pmid,
        title=clean_text(str(art.get("ArticleTitle", ""))),
        abstract=_parse_abstract(art),
        authors=_parse_authors(art),
        published_date=_parse_date(article),
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        pdf_url=_parse_pmc_pdf(article),
        doi=_parse_doi(article),
        categories=mesh,
        journal=str(art.get("Journal", {}).get("Title", "")) or None,
    )


# ---------------------------------------------------------------- scraper ---
class PubMedScraper:
    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None,
                 retries: int = 3, backoff: float = 2.0):
        _configure_entrez(email, api_key)
        self.retries = retries
        self.backoff = backoff

    def _call(self, fn, **kwargs):
        """Call an Entrez function with retry + exponential back-off."""
        for attempt in range(1, self.retries + 1):
            try:
                handle = fn(**kwargs)
                try:
                    return Entrez.read(handle)
                finally:
                    handle.close()
            except Exception as exc:          # HTTPError, URLError, timeouts
                logger.warning("Entrez %s failed (attempt %d/%d): %s",
                               fn.__name__, attempt, self.retries, exc)
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff ** attempt)

    def search_ids(self, query: str, start_date: Optional[date] = None,
                   end_date: Optional[date] = None, max_results: int = 20) -> List[str]:
        params = dict(db="pubmed", term=query, retmax=max_results, sort="pub_date")
        if start_date or end_date:
            params.update(
                datetype="pdat",
                mindate=(start_date or date(1900, 1, 1)).strftime("%Y/%m/%d"),
                maxdate=(end_date or datetime.now(timezone.utc).date()).strftime("%Y/%m/%d"),
            )
        result = self._call(Entrez.esearch, **params)
        ids = list(result.get("IdList", []))
        logger.info("PubMed esearch '%s' -> %d PMIDs", query, len(ids))
        return ids

    def fetch(self, pmids: List[str]) -> List[Paper]:
        if not pmids:
            return []
        records = self._call(Entrez.efetch, db="pubmed", id=",".join(pmids), retmode="xml")
        papers = []
        for art in records.get("PubmedArticle", []):
            try:
                paper = record_to_paper(art)
            except Exception as exc:
                logger.warning("Skipping malformed PubMed record: %s", exc)
                continue
            if paper.is_valid():          # drops records with no abstract
                papers.append(paper)
        return papers

    def search(self, query: str, start_date: Optional[date] = None,
               end_date: Optional[date] = None, max_results: int = 20) -> List[Paper]:
        try:
            # over-fetch a little: some records have no abstract and get dropped
            ids = self.search_ids(query, start_date, end_date, max_results * 2)
            papers = self.fetch(ids)
        except Exception as exc:
            logger.error("PubMed request failed: %s", exc)
            return []
        return deduplicate(papers)[:max_results]


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape recent PubMed abstracts")
    ap.add_argument("--query", default="machine learning drug discovery")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--out", default="data/pubmed_papers.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    end = datetime.now(timezone.utc).date()
    papers = PubMedScraper().search(args.query, end - timedelta(days=args.days), end, args.max)
    path = save_papers(papers, args.out)
    print(f"Saved {len(papers)} PubMed papers -> {path}")


if __name__ == "__main__":
    main()
