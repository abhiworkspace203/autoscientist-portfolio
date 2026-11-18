"""
Unified paper schema shared by every ingestion source.

Both the arXiv and PubMed scrapers return a list of `Paper` objects, so the
rest of the system (Member 4's vector DB, Member 2's Literature Reviewer agent)
never needs to know where a paper came from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

REQUIRED_FIELDS = ("paper_id", "source", "title", "abstract", "authors", "published_date")


@dataclass
class Paper:
    """One scientific paper, normalised across sources."""

    paper_id: str                      # "arxiv:2410.01234v1" or "pubmed:39123456"
    source: str                        # "arxiv" | "pubmed"
    source_id: str                     # raw ID at the source (arXiv short id / PMID)
    title: str
    abstract: str
    authors: List[str]
    published_date: str                # ISO date "YYYY-MM-DD"
    url: str                           # landing page
    pdf_url: Optional[str] = None      # arXiv always has one; PubMed usually not
    doi: Optional[str] = None
    categories: List[str] = field(default_factory=list)   # arXiv cats / MeSH terms
    journal: Optional[str] = None
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # ---- helpers -----------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def content_hash(self) -> str:
        """Stable hash of title+abstract — used for de-duplication."""
        text = (self.title.strip().lower() + "|" + self.abstract.strip().lower()).encode()
        return hashlib.sha256(text).hexdigest()[:16]

    def is_valid(self) -> bool:
        """True when every required field is present and non-empty."""
        return all(bool(getattr(self, f)) for f in REQUIRED_FIELDS)


def clean_text(text: Optional[str]) -> str:
    """Collapse whitespace/newlines that both APIs put inside titles/abstracts."""
    return " ".join((text or "").split())


def deduplicate(papers: Iterable[Paper]) -> List[Paper]:
    """Drop duplicates by paper_id, DOI, or identical title+abstract."""
    seen_ids, seen_dois, seen_hashes, out = set(), set(), set(), []
    for p in papers:
        h = p.content_hash()
        doi = (p.doi or "").lower()
        if p.paper_id in seen_ids or h in seen_hashes or (doi and doi in seen_dois):
            continue
        seen_ids.add(p.paper_id)
        seen_hashes.add(h)
        if doi:
            seen_dois.add(doi)
        out.append(p)
    return out


def save_papers(papers: Iterable[Paper], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [p.to_dict() for p in papers]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_papers(path: str | Path) -> List[Paper]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Paper(**d) for d in data]
