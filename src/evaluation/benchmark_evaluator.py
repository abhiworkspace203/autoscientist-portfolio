"""
Benchmark evaluator (Member 3, Week 4).

Grades a research paper on 4 dimensions, each 0-10:

  1. Novelty            - how different the idea is from existing papers (TF-IDF similarity)
  2. Literature grounding - are the citations real (retrieved) and relevant?
  3. Empirical validity - is there statistical evidence (p < 0.05, effect size, CI)?
  4. Formatting         - standard academic structure (sections, abstract, figures, references)

The same scorer is used for the agent's generated paper and for every human paper
in data/holdout_papers/, so the two can be compared fairly. Scoring is
deterministic and offline (no LLM calls), so results are reproducible.

    python -m src.evaluation.benchmark_evaluator --agent examples/sample_agent_output.json
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.evaluation.holdout_curator import HOLDOUT_DIR, load_holdout
from src.evaluation.paper_structure import REQUIRED_SECTIONS, analyse_structure, pdf_text

DIMENSIONS = ["novelty", "grounding", "empirical_validity", "formatting"]
SIGNIFICANCE_LEVEL = 0.05
MIN_CITATIONS = 3
TARGET_REFERENCES = 25
RELEVANCE_FULL_AT = 0.25

_STOPWORDS = set("""
a about above after again all also an and any are as at be because been being below between both
but by can could did do does doing during each few for from further had has have having here how
however if in into is it its itself just more most no nor not of off on once only or other our out
over own same she should so some such than that the their them then there these they this those
through to too under until up very was we were what when where which while who whom why will with
would you your our ours paper study propose proposed show shows results result using use used based
new approach method methods model models data also two one three via within across between
""".split())
_WORD = re.compile(r"[a-z][a-z\-]{2,}")


@dataclass
class PaperScore:
    paper_id: str
    title: str
    source: str
    domain: str = ""
    novelty: float = 0.0
    grounding: float = 0.0
    empirical_validity: float = 0.0
    formatting: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall(self) -> float:
        return round(sum(getattr(self, d) for d in DIMENSIONS) / len(DIMENSIONS), 2)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overall"] = self.overall
        return d


def _clip(x: float) -> float:
    return round(max(0.0, min(10.0, float(x))), 2)


def tokenize(text: str) -> List[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOPWORDS]


class TfidfIndex:
    """Tiny TF-IDF + cosine similarity (no external dependencies)."""

    def __init__(self, documents: Sequence[str]):
        self.docs = [Counter(tokenize(d)) for d in documents]
        n = len(self.docs)
        df: Counter = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vectors = [self._weigh(doc) for doc in self.docs]

    def _weigh(self, counts: Counter) -> Dict[str, float]:
        vec = {t: c * self.idf.get(t, math.log(1 + len(self.docs)) + 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def similarities(self, text: str) -> List[float]:
        q = self._weigh(Counter(tokenize(text)))
        return [sum(w * vec.get(t, 0.0) for t, w in q.items()) for vec in self.vectors]


def similarity(a: str, b: str) -> float:
    return TfidfIndex([b]).similarities(a)[0] if tokenize(a) and tokenize(b) else 0.0


# ---------------------------------------------------------------- 1. novelty
def score_novelty(text: str, reference_texts: Sequence[str]) -> Dict[str, Any]:
    """10 = nothing similar exists; 0 = near-copy of an existing paper."""
    refs = [r for r in reference_texts if tokenize(r)]
    if not tokenize(text):
        return {"score": 0.0, "max_similarity": None, "mean_similarity": None, "compared_with": 0}
    if not refs:
        return {"score": 5.0, "max_similarity": None, "mean_similarity": None, "compared_with": 0,
                "note": "no reference papers to compare with; neutral score"}
    sims = TfidfIndex(refs).similarities(text)
    top = max(sims)
    mean = sum(sims) / len(sims)
    overlap = 0.7 * top + 0.3 * mean
    return {"score": _clip(10 * (1 - overlap)), "max_similarity": round(top, 3),
            "mean_similarity": round(mean, 3), "compared_with": len(refs)}


# ---------------------------------------------------------------- 2. grounding
def score_grounding(claim_text: str, cited_ids: Iterable[str],
                    known_papers: Dict[str, str]) -> Dict[str, Any]:
    """
    Agent papers: citations are checked against the papers the retriever actually
    returned (known_papers: id -> abstract). 60% authenticity, 40% relevance,
    scaled down when fewer than MIN_CITATIONS are given.
    """
    cited = list(dict.fromkeys(cited_ids))
    if not cited:
        return {"score": 0.0, "citations": 0, "authentic": 0, "invalid_ids": [], "relevance": 0.0}
    authentic = [c for c in cited if c in known_papers]
    invalid = [c for c in cited if c not in known_papers]
    auth_ratio = len(authentic) / len(cited)
    sims = [similarity(claim_text, known_papers[c]) for c in authentic]
    relevance = min(1.0, (sum(sims) / len(sims)) / RELEVANCE_FULL_AT) if sims else 0.0
    coverage = min(1.0, len(cited) / MIN_CITATIONS)
    score = 10 * (0.6 * auth_ratio + 0.4 * relevance) * coverage
    return {"score": _clip(score), "citations": len(cited), "authentic": len(authentic),
            "invalid_ids": invalid, "relevance": round(relevance, 3)}


def score_grounding_from_references(n_references: int) -> Dict[str, Any]:
    """Human papers: references are assumed real; score reflects how well-referenced it is."""
    return {"score": _clip(10 * min(1.0, n_references / TARGET_REFERENCES)),
            "references": n_references, "note": f"proxy: {TARGET_REFERENCES}+ references = 10"}


# ---------------------------------------------------------------- 3. empirical validity
def score_empirical(p_value: Optional[float], effect_size: Optional[float] = None,
                    ci: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """
    Agent papers (numbers from the critic / stats validator):
      significant (p < 0.05) = 6, effect size |d| >= 0.5 = +2 (>= 0.2 = +1),
      95% CI that excludes zero = +2. Not significant: at most 3.
    """
    if p_value is None:
        return {"score": 0.0, "significant": False, "note": "no p-value reported"}
    significant = p_value < SIGNIFICANCE_LEVEL
    score = 6.0 if significant else 2.0 * (1 - min(1.0, p_value))
    if effect_size is not None:
        d = abs(effect_size)
        score += 2.0 if d >= 0.5 else 1.0 if d >= 0.2 else 0.0
    ci_ok = bool(ci and len(ci) == 2 and (ci[0] > 0 or ci[1] < 0))
    if ci_ok:
        score += 2.0
    if not significant:
        score = min(score, 3.0)
    return {"score": _clip(score), "significant": significant, "p_value": p_value,
            "effect_size": effect_size, "ci_excludes_zero": ci_ok}


_P_REPORTED = re.compile(r"\bp\s*[<=≤]\s*0?\.\d+|\bp-values?\b|statistically significant", re.I)
_UNCERTAINTY = re.compile(r"±|\+/-|confidence interval|\bstd\b|standard deviation|error bars?|"
                          r"standard error|\bCI\b", re.I)
_REPEATS = re.compile(r"\b(?:\d+|five|three|ten)\s+(?:random\s+)?(?:seeds|runs|trials|"
                      r"replicates|folds)\b|cross-validation", re.I)
_STAT_TEST = re.compile(r"t-test|wilcoxon|mann[- ]whitney|anova|bootstrap|chi-?square|"
                        r"cohen'?s\s*d|effect size", re.I)


def score_empirical_from_text(text: str, n_tables: int = 0, n_figures: int = 0) -> Dict[str, Any]:
    """Human papers: evidence of statistical rigour found in the text."""
    checks = {
        "significance_reported": bool(_P_REPORTED.search(text)),
        "statistical_test": bool(_STAT_TEST.search(text)),
        "uncertainty_reported": bool(_UNCERTAINTY.search(text)),
        "repeated_runs": bool(_REPEATS.search(text)),
        "results_tables": n_tables > 0,
        "result_figures": n_figures > 0,
    }
    weights = {"significance_reported": 3, "statistical_test": 2, "uncertainty_reported": 2,
               "repeated_runs": 1, "results_tables": 1, "result_figures": 1}
    score = sum(weights[k] for k, ok in checks.items() if ok)
    return {"score": _clip(score), **checks}


# ---------------------------------------------------------------- 4. formatting
def score_formatting(structure: Dict[str, Any]) -> Dict[str, Any]:
    """6 points for the required sections, abstract length 1.5, figures 1, tables 0.5, refs 1."""
    present = [s for s in REQUIRED_SECTIONS if s in structure.get("sections", [])]
    score = 6.0 * len(present) / len(REQUIRED_SECTIONS)
    words = structure.get("abstract_words", 0)
    if 100 <= words <= 300:
        score += 1.5
    elif 50 <= words <= 400:
        score += 0.75
    score += 1.0 if structure.get("figures", 0) >= 1 else 0.0
    score += 0.5 if structure.get("tables", 0) >= 1 else 0.0
    score += 1.0 if structure.get("references", 0) >= 5 else 0.5 if structure.get("references", 0) else 0.0
    return {"score": _clip(score), "sections_present": present,
            "missing_sections": [s for s in REQUIRED_SECTIONS if s not in present],
            "abstract_words": words, "figures": structure.get("figures", 0),
            "tables": structure.get("tables", 0), "references": structure.get("references", 0)}


# ---------------------------------------------------------------- whole papers
def evaluate_agent_output(output: Dict[str, Any], holdout: Sequence[Dict[str, Any]] = ()) -> PaperScore:
    """
    Score the agent's paper. Expected keys (missing ones score 0):
      title, paper_text (LaTeX or Markdown), hypothesis, cited_ids,
      retrieved_papers [{paper_id, abstract}], p_value, effect_size, ci [low, high]
    """
    text = output.get("paper_text", "")
    hypothesis = output.get("hypothesis", "")
    retrieved = {p["paper_id"]: f"{p.get('title', '')} {p.get('abstract', '')}"
                 for p in output.get("retrieved_papers", [])}
    references = list(retrieved.values()) + [f"{h['title']} {h['abstract']}" for h in holdout]
    structure = analyse_structure(text)
    claim = f"{hypothesis} {output.get('abstract', '')}".strip() or text[:2000]

    novelty = score_novelty(claim, references)
    grounding = score_grounding(claim, output.get("cited_ids", []), retrieved)
    empirical = score_empirical(output.get("p_value"), output.get("effect_size"), output.get("ci"))
    formatting = score_formatting(structure)
    return PaperScore(
        paper_id=output.get("paper_id", "agent"), title=output.get("title", "Agent paper"),
        source="agent", domain=output.get("domain", ""),
        novelty=novelty["score"], grounding=grounding["score"],
        empirical_validity=empirical["score"], formatting=formatting["score"],
        details={"novelty": novelty, "grounding": grounding, "empirical_validity": empirical,
                 "formatting": formatting},
    )


def evaluate_human_paper(paper: Dict[str, Any], others: Sequence[Dict[str, Any]],
                         text: Optional[str] = None) -> PaperScore:
    """Score one holdout paper; novelty is measured against the other holdout papers."""
    if text is None:
        text = pdf_text(paper["pdf_path"]) if paper.get("pdf_path") else ""
    structure = analyse_structure(text) if text else paper.get("structure", {})
    claim = f"{paper['title']} {paper['abstract']}"
    refs = [f"{o['title']} {o['abstract']}" for o in others if o["paper_id"] != paper["paper_id"]]

    novelty = score_novelty(claim, refs)
    grounding = score_grounding_from_references(structure.get("references", 0))
    empirical = score_empirical_from_text(text or paper["abstract"], structure.get("tables", 0),
                                          structure.get("figures", 0))
    formatting = score_formatting(structure)
    return PaperScore(
        paper_id=paper["paper_id"], title=paper["title"], source="human",
        domain=paper.get("domain", ""),
        novelty=novelty["score"], grounding=grounding["score"],
        empirical_validity=empirical["score"], formatting=formatting["score"],
        details={"novelty": novelty, "grounding": grounding, "empirical_validity": empirical,
                 "formatting": formatting, "text_available": bool(text)},
    )


def evaluate_holdout(holdout: Optional[Sequence[Dict[str, Any]]] = None,
                     holdout_dir: Path = HOLDOUT_DIR) -> List[PaperScore]:
    holdout = list(holdout) if holdout is not None else load_holdout(holdout_dir)
    return [evaluate_human_paper(p, holdout) for p in holdout]


def average_scores(scores: Sequence[PaperScore]) -> Dict[str, float]:
    if not scores:
        return {d: 0.0 for d in DIMENSIONS + ["overall"]}
    avg = {d: round(sum(getattr(s, d) for s in scores) / len(scores), 2) for d in DIMENSIONS}
    avg["overall"] = round(sum(avg.values()) / len(DIMENSIONS), 2)
    return avg


def run_benchmark(agent_output: Optional[Dict[str, Any]] = None,
                  holdout_dir: Path = HOLDOUT_DIR) -> Dict[str, Any]:
    holdout = load_holdout(holdout_dir)
    human = evaluate_holdout(holdout)
    agent = evaluate_agent_output(agent_output, holdout) if agent_output else None
    return {
        "agent": agent.to_dict() if agent else None,
        "human": [s.to_dict() for s in human],
        "human_average": average_scores(human),
    }


def format_scores(result: Dict[str, Any]) -> str:
    head = f"{'Paper':<34} {'Novelty':>8} {'Ground.':>8} {'Empir.':>8} {'Format':>8} {'Overall':>8}"
    lines = [head, "-" * len(head)]

    def row(name: str, s: Dict[str, Any]) -> str:
        return (f"{name[:34]:<34} {s['novelty']:>8.2f} {s['grounding']:>8.2f} "
                f"{s['empirical_validity']:>8.2f} {s['formatting']:>8.2f} {s['overall']:>8.2f}")

    for s in result["human"]:
        lines.append(row(f"{s['paper_id']} ({s['domain'][:4]})", s))
    lines.append("-" * len(head))
    lines.append(row("Human average", result["human_average"]))
    if result["agent"]:
        lines.append(row("Agent", result["agent"]))
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Score the agent's paper against the holdout set")
    ap.add_argument("--agent", default=None, help="JSON file with the agent's output")
    ap.add_argument("--holdout", default=str(HOLDOUT_DIR))
    ap.add_argument("--save", default=None, help="write all scores to this JSON file")
    args = ap.parse_args()

    agent = json.loads(Path(args.agent).read_text(encoding="utf-8")) if args.agent else None
    result = run_benchmark(agent, Path(args.holdout))
    if not result["human"]:
        print("No holdout papers found. Run: python -m src.evaluation.holdout_curator")
        return
    print(format_scores(result))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nScores saved to {args.save}")


if __name__ == "__main__":
    main()
