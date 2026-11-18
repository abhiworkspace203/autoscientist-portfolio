"""
Literature Retriever (Member 3 - Week 2: Vector Retrieval & Query Engineering)

Queries the team's ChromaDB store (ScientificVectorDB in src.rag.vector_db) and
turns the matching chunks into a short, ranked, citable context block for the
Hypothesis Generator (Member 2), so the LLM never runs out of context on
irrelevant text.

Pipeline for one query:
    1. vector search      <- ScientificVectorDB.search() over paper chunks (ChromaDB)
    2. cosine similarity  -> computed from the Chroma distance (see `to_cosine`)
    3. score threshold    -> chunks below `min_score` are discarded
    4. metadata filters   -> publication year range, domain tags (cs.AI, ...), source
    5. best chunk / paper -> top_k papers, highest similarity first
    6. formatting         -> "[Paper 1: Title (Authors, Year)] ID: ... Abstract snippet: ..."

Main entry point (Week 2 spec):
    get_relevant_literature(topic: str, top_k: int = 5) -> List[Dict]

Example:
    from src.rag.retriever import get_relevant_literature, format_context
    hits = get_relevant_literature("deep learning for drug discovery", top_k=3,
                                   year_from=2026, categories=["cs.AI"])
    prompt_context = format_context(hits)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.rag.vector_db import ScientificVectorDB

logger = logging.getLogger(__name__)

DEFAULT_MIN_SCORE = 0.20          # cosine similarity below this = irrelevant
DEFAULT_MAX_CONTEXT_CHARS = 4000  # total budget for the formatted prompt context
SNIPPET_CHARS = 450               # max characters of chunk text shown per paper
MIN_CANDIDATES = 20               # chunks fetched from Chroma before filtering


# ------------------------------------------------------------------ helpers
def split_list(value: Any) -> List[str]:
    """Chroma metadata stores lists as 'a, b, c' strings; turn them back into lists."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value or "").split(", ") if part.strip()]


def year_of(date_text: Any) -> Optional[int]:
    """Publication year from 'YYYY-MM-DD' (or any string starting with a year)."""
    match = re.match(r"(\d{4})", str(date_text or ""))
    return int(match.group(1)) if match else None


def short_authors(authors: Sequence[str], limit: int = 2) -> str:
    """'A. Smith', 'A. Smith & B. Lee', or 'A. Smith et al.'."""
    authors = [a for a in (authors or []) if a]
    if not authors:
        return "Unknown authors"
    if len(authors) == 1:
        return authors[0]
    if len(authors) <= limit:
        return " & ".join(authors)
    return f"{authors[0]} et al."


def truncate(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    """Cut at a word boundary and add an ellipsis when shortened."""
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + " ..."


def strip_title(chunk_text: str, title: str) -> str:
    """vector_db stores chunk 0 as 'Title\\n\\nAbstract...'; drop the repeated title."""
    text = (chunk_text or "").strip()
    if title and text.startswith(title.strip()):
        text = text[len(title.strip()):].lstrip()
    return text


def distance_space(collection: Any) -> str:
    """Distance metric of a Chroma collection: 'l2' (Chroma default), 'cosine' or 'ip'."""
    try:
        cfg = collection.configuration or {}
        space = (cfg.get("hnsw") or {}).get("space")
        if space:
            return str(space)
    except Exception:
        pass
    meta = getattr(collection, "metadata", None) or {}
    return str(meta.get("hnsw:space", "l2"))


def to_cosine(distance: float, space: str = "l2") -> float:
    """
    Convert a Chroma distance to cosine similarity in [0, 1].
      cosine space : d = 1 - cos            -> cos = 1 - d
      ip space     : d = 1 - dot            -> cos = 1 - d   (unit vectors)
      l2 space     : d = ||a - b||^2 = 2 - 2cos (unit vectors) -> cos = 1 - d / 2
    all-MiniLM-L6-v2 and the vector_db hash fallback both output unit-length vectors.
    """
    d = float(distance)
    cos = 1.0 - d / 2.0 if space == "l2" else 1.0 - d
    return max(0.0, min(1.0, cos))


# ---------------------------------------------------------- metadata filters
def matches_filters(
    meta: Dict[str, Any],
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    categories: Optional[Iterable[str]] = None,
    sources: Optional[Iterable[str]] = None,
) -> bool:
    """
    True if a chunk's metadata passes every filter that was given.
      year_from / year_to : inclusive publication-year range
      categories          : keep papers having ANY of these tags (case-insensitive),
                            e.g. ["cs.AI", "cs.LG"] or MeSH terms like ["Machine Learning"]
      sources             : e.g. ["arxiv"] or ["pubmed"]
    """
    year = year_of(meta.get("published_date"))
    if year_from is not None and (year is None or year < year_from):
        return False
    if year_to is not None and (year is None or year > year_to):
        return False
    if categories:
        have = str(meta.get("categories") or "").lower()
        # tags are joined with ", " - match whole tags only (MeSH tags may contain commas)
        if not any(re.search(rf"(^|, ){re.escape(c.lower())}(,|$)", have) for c in categories):
            return False
    if sources:
        if str(meta.get("source", "")).lower() not in {s.lower() for s in sources}:
            return False
    return True


# ---------------------------------------------------------------- retriever
class LiteratureRetriever:
    """Query ScientificVectorDB and return clean, ranked, citable paper context."""

    def __init__(self, db: Optional[ScientificVectorDB] = None,
                 min_score: float = DEFAULT_MIN_SCORE):
        if db is None:
            db = ScientificVectorDB()             # the project's persistent store
            if db.count() == 0:                    # first run: index the sample papers
                db.load_from_json()
        self.db = db
        self.min_score = min_score
        self.space = distance_space(db.collection)

    def _search_chunks(self, topic: str, filtered: bool, top_k: int) -> List[Dict[str, Any]]:
        total_chunks = self.db.collection.count()
        if total_chunks == 0:
            return []
        # Filters are applied after the vector search, so fetch every chunk when
        # filtering (stores are small); otherwise a generous candidate pool.
        n = total_chunks if filtered else min(total_chunks, max(MIN_CANDIDATES, top_k * 4))
        return self.db.search(topic, top_k=n)

    def _first_body_chunk(self, paper_id: str, title: str) -> str:
        """
        Long papers are stored as a title-only chunk plus abstract chunks. If the
        query matched only the title chunk, use the paper's first abstract chunk.
        """
        try:
            got = self.db.collection.get(where={"paper_id": paper_id},
                                         include=["documents", "metadatas"])
        except Exception:
            return ""
        pairs = sorted(zip(got.get("documents") or [], got.get("metadatas") or []),
                       key=lambda dm: (dm[1] or {}).get("chunk_number", 0))
        for doc, _ in pairs:
            body = strip_title(doc, title)
            if body:
                return body
        return ""

    def retrieve(
        self,
        topic: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        categories: Optional[Iterable[str]] = None,
        sources: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not topic or not topic.strip() or top_k <= 0:
            return []
        threshold = self.min_score if min_score is None else min_score
        filtered = any(f for f in (year_from, year_to, categories, sources))

        best: Dict[str, Dict[str, Any]] = {}       # paper_id -> best score + snippet
        for hit in self._search_chunks(topic, filtered, top_k):
            meta = hit.get("metadata") or {}
            pid = hit.get("paper_id") or meta.get("paper_id")
            if not pid:
                continue
            score = to_cosine(hit.get("distance", 2.0), self.space)
            if score < threshold:
                continue
            if not matches_filters(meta, year_from, year_to, categories, sources):
                continue
            body = strip_title(hit.get("text", ""), str(meta.get("title", "")))
            entry = best.setdefault(pid, {"similarity": score, "text": "", "meta": meta})
            entry["similarity"] = max(entry["similarity"], score)
            if body and not entry["text"]:          # hits arrive best-first
                entry["text"] = body

        for pid, entry in best.items():             # matched only the title chunk
            if not entry["text"]:
                entry["text"] = self._first_body_chunk(pid, str(entry["meta"].get("title", "")))

        ranked = sorted(best.items(), key=lambda kv: kv[1]["similarity"], reverse=True)[:top_k]
        results = []
        for rank, (pid, b) in enumerate(ranked, start=1):
            meta = b["meta"]
            title = str(meta.get("title", "")).strip()
            results.append({
                "rank": rank,
                "paper_id": pid,
                "title": title,
                "authors": split_list(meta.get("authors")),
                "year": year_of(meta.get("published_date")),
                "source": meta.get("source", "unknown"),
                "categories": split_list(meta.get("categories")),
                "url": meta.get("url", ""),
                "similarity": round(b["similarity"], 4),
                "snippet": truncate(b["text"]),
            })
        logger.info("Retrieved %d papers for %r (min_score=%.2f, space=%s)",
                    len(results), topic, threshold, self.space)
        return results


# --------------------------------------------------------------- formatting
def format_paper_block(result: Dict[str, Any], number: int) -> str:
    """[Paper 1: Title (Authors, Year)] / ID: ... / Abstract snippet: ..."""
    year = result.get("year") or "n.d."
    return (f"[Paper {number}: {result.get('title', '').strip()} "
            f"({short_authors(result.get('authors', []))}, {year})]\n"
            f"ID: {result.get('paper_id')} | Similarity: {result.get('similarity', 0):.2f}\n"
            f"Abstract snippet: {result.get('snippet', '')}")


def format_context(results: List[Dict[str, Any]],
                   max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> str:
    """
    Join paper blocks into one prompt-ready string, stopping before `max_chars`
    so the LLM context window is never flooded. Paper IDs are included so the
    Hypothesis agent can cite them (cited_paper_ids).
    """
    blocks, used = [], 0
    for i, r in enumerate(results, start=1):
        block = format_paper_block(r, i)
        if blocks and used + len(block) + 2 > max_chars:
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def valid_paper_ids(results: List[Dict[str, Any]]) -> List[str]:
    """IDs the Hypothesis agent is allowed to cite (to reject made-up citations)."""
    return [r["paper_id"] for r in results if r.get("paper_id")]


# --------------------------------------------------------------- public API
_default_retriever: Optional[LiteratureRetriever] = None


def get_retriever() -> LiteratureRetriever:
    """Shared retriever over the project's vector store (created on first use)."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = LiteratureRetriever()
    return _default_retriever


def get_relevant_literature(topic: str, top_k: int = 5, **filters: Any) -> List[Dict]:
    """
    Week 2 spec entry point.
    Returns up to `top_k` papers most similar to `topic` (cosine similarity),
    each with: rank, paper_id, title, authors, year, source, categories, url,
    similarity, snippet.
    Optional filters: min_score, year_from, year_to, categories, sources.
    """
    return get_retriever().retrieve(topic, top_k=top_k, **filters)


def retrieve_for_hypothesis(topic: str, top_k: int = 5, **filters: Any) -> Dict[str, Any]:
    """
    One-call helper for the Hypothesis agent (Member 2) and the orchestrator:
        {
          "topic": ...,
          "context": "[Paper 1: ...] ...",        # paste into the LLM prompt
          "valid_paper_ids": ["arxiv:...", ...],  # only these may appear in cited_paper_ids
          "papers": [ {...}, ... ],               # full results from get_relevant_literature
        }
    """
    papers = get_relevant_literature(topic, top_k=top_k, **filters)
    return {
        "topic": topic,
        "context": format_context(papers),
        "valid_paper_ids": valid_paper_ids(papers),
        "papers": papers,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Query the literature retriever")
    ap.add_argument("topic")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--categories", nargs="+")
    ap.add_argument("--sources", nargs="+")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    hits = LiteratureRetriever(min_score=args.min_score).retrieve(
        args.topic, args.top_k, year_from=args.year_from, year_to=args.year_to,
        categories=args.categories, sources=args.sources)
    print(format_context(hits) if hits else "No papers above the similarity threshold.")
