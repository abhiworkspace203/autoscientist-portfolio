"""
RAG grounding helpers (Member 3 - Week 2, Day 4: Retriever -> Hypothesis Generator).

Glue between the literature retriever (src/rag/retriever.py) and the Hypothesis
agent (Member 2):

    build_grounded_prompt(topic)        -> prompt text + the paper IDs it may cite
    find_invalid_citations(cited, ok)   -> cited IDs that were never retrieved
    check_hypothesis_citations(h, ok)   -> same check on a hypothesis dict / Pydantic model

Usage in the Hypothesis agent:
    from src.rag.grounding import build_grounded_prompt, check_hypothesis_citations

    grounded = build_grounded_prompt("antimicrobial resistance", top_k=5)
    raw = llm.complete(grounded["prompt"])
    hypothesis = GroundedHypothesis.model_validate_json(raw)
    bad = check_hypothesis_citations(hypothesis, grounded["valid_paper_ids"])
    if bad:
        ...  # reject or regenerate: the LLM cited papers it was never given
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.rag.retriever import retrieve_for_hypothesis

GROUNDING_INSTRUCTION = (
    "You are an academic researcher. Formulate a novel hypothesis that directly "
    "bridges the gap between the papers below (e.g. Paper A and Paper B). Every "
    "hypothesis must cite the Paper ID and the specific findings it builds on. "
    "Use ONLY the papers listed below; never invent papers, IDs, datasets or results."
)

OUTPUT_FORMAT = (
    "Return a single JSON object with exactly these fields:\n"
    '{"title": str, "problem_statement": str, "proposed_solution": str, '
    '"cited_paper_ids": [str, ...], "falsifiable_metric": str, '
    '"expected_code_modules": [str, ...]}'
)

NO_PAPERS_MESSAGE = "No sufficiently relevant papers were found for this topic."


def build_grounded_prompt(topic: str, top_k: int = 5, **filters: Any) -> Dict[str, Any]:
    """
    Retrieve papers for `topic` and build the full RAG prompt for the LLM.

    Returns:
        {
          "topic": str,
          "prompt": str,                  # send this to the LLM
          "valid_paper_ids": [str, ...],  # the only IDs allowed in cited_paper_ids
          "papers": [dict, ...],          # retriever results (for logging / UI)
        }
    Filters (min_score, year_from, year_to, categories, sources) are passed to the retriever.
    """
    bundle = retrieve_for_hypothesis(topic, top_k=top_k, **filters)
    ids = bundle["valid_paper_ids"]
    context = bundle["context"] or NO_PAPERS_MESSAGE
    prompt = (
        f"{GROUNDING_INSTRUCTION}\n\n"
        f"Research topic: {topic}\n\n"
        f"Retrieved literature:\n{context}\n\n"
        f"Allowed Paper IDs for cited_paper_ids: {', '.join(ids) if ids else '(none)'}\n\n"
        f"{OUTPUT_FORMAT}"
    )
    return {"topic": topic, "prompt": prompt, "valid_paper_ids": ids,
            "papers": bundle["papers"]}


def find_invalid_citations(cited_paper_ids: Iterable[str],
                           valid_paper_ids: Iterable[str]) -> List[str]:
    """Cited IDs that were NOT in the retrieved set (hallucinated citations), in order."""
    valid = {str(v).strip() for v in valid_paper_ids}
    seen, invalid = set(), []
    for pid in cited_paper_ids or []:
        pid = str(pid).strip()
        if pid not in valid and pid not in seen:
            invalid.append(pid)
            seen.add(pid)
    return invalid


def check_hypothesis_citations(hypothesis: Any, valid_paper_ids: Iterable[str],
                               min_citations: int = 1) -> List[str]:
    """
    Validate one hypothesis (dict or Pydantic model with `cited_paper_ids`).
    Returns a list of problems; an empty list means the hypothesis is properly grounded.
    """
    if isinstance(hypothesis, dict):
        cited = hypothesis.get("cited_paper_ids") or []
    else:
        cited = getattr(hypothesis, "cited_paper_ids", None) or []
    problems = []
    if len(cited) < min_citations:
        problems.append(f"cites {len(cited)} paper(s); at least {min_citations} required")
    for pid in find_invalid_citations(cited, valid_paper_ids):
        problems.append(f"cites unknown paper ID: {pid}")
    return problems
