"""
Critic Agent (the harsh peer reviewer)
Member 5 Deliverable (Week 3)

Reviews one finished experiment like a referee for a top venue (Nature / NeurIPS).
It reads the hypothesis, the sandbox result and Member 4's statistical validation, runs
three checks and returns a structured verdict:

    VERDICT: ACCEPTED | REJECTED | NEEDS_REPLICATION   + critique bullets

Checks
  1. Did the experiment actually test the core hypothesis claim?
  2. Is the sample size adequate?
  3. Is p < 0.05?  (p >= 0.05 -> inconclusive / rejected)

Design rule: the statistics decide the verdict, never the LLM. An LLM may (optionally)
judge Check 1 and write the critique text, but it can not turn a non-significant result
into ACCEPTED. With no LLM the agent runs fully offline with simple rules.

Contract for the orchestrator (critic_node):
    review = critique_experiment(hypothesis, execution_result, stats, code=generated_code)
    review.verdict        -> "ACCEPTED" | "REJECTED" | "NEEDS_REPLICATION"
    review.critique       -> list of bullet strings
    review.model_dump()   -> plain dict for the graph state

`stats` is the dict from src/utils/stats_validator.py:
    {"p_value", "is_significant", "confidence_interval_95", "effect_size_cohens_d", "summary"}
and, optionally, "n_baseline" and "n_treatment" (group sizes) for the sample-size check.
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ALPHA = 0.05                 # significance level
MIN_SAMPLE_SIZE = 30         # minimum observations per group
NEGLIGIBLE_EFFECT = 0.2      # |Cohen's d| below this is too small to matter
MIN_HYPOTHESIS_COVERAGE = 0.3  # share of hypothesis key terms the experiment must mention
COPY_THRESHOLD = 0.8            # a string or line holding this share of the claim's key terms is a copy of the claim
MIN_TERMS_FOR_COPY_DETECTION = 4  # shorter claims only get exact-copy removal (avoids dropping real result lines)

CRITIC_PROMPT = """You are a rigorous peer reviewer for a top-tier venue (Nature / NeurIPS).
Review the experiment below. Be strict and specific.

Hypothesis under test:
{hypothesis}

Experiment code:
```python
{code}
```

Sandbox execution result:
{execution_result}

Statistical validation (computed by a separate engine, do not recompute it):
{stats}

Evaluate:
1. Did the experiment actually test the core claim of the hypothesis (its variables and its direction)?
2. Is the sample size adequate, and are there obvious sources of bias?
3. Does the statistical evidence (p-value, confidence interval, effect size) support the claim?

Return only JSON, with no other text:
{{"hypothesis_tested": true or false, "critique": ["specific critique bullet", "..."]}}
"""

_STOPWORDS = {
    "with", "that", "this", "than", "from", "into", "over", "under", "when", "while", "then",
    "will", "would", "should", "could", "have", "has", "had", "are", "was", "were", "been",
    "does", "not", "and", "the", "for", "its", "their", "which", "using", "use", "uses",
    "hypothesis", "novel", "new", "improve", "improves", "improved", "increase", "increases",
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
class Checks(BaseModel):
    """Result of each check. None means 'could not be determined'."""

    hypothesis_tested: Optional[bool] = None
    sample_size_adequate: Optional[bool] = None
    significant: Optional[bool] = None


class CriticReview(BaseModel):
    """The Critic's structured decision."""

    verdict: Literal["ACCEPTED", "REJECTED", "NEEDS_REPLICATION"]
    checks: Checks
    critique: List[str] = Field(min_length=1)
    p_value: Optional[float] = None

    def as_text(self) -> str:
        """Human-readable review: the verdict line followed by the critique bullets."""
        return f"VERDICT: {self.verdict}\n" + "\n".join(f"- {line}" for line in self.critique)


# ---------------------------------------------------------------------------
# Prompt and LLM reply handling
# ---------------------------------------------------------------------------
def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def build_critic_prompt(
    hypothesis: str, execution_result: Any, stats: Optional[Dict[str, Any]], code: str = ""
) -> str:
    """Fill the reviewer prompt with the hypothesis, code, result and statistics."""
    return CRITIC_PROMPT.format(
        hypothesis=(hypothesis or "(not provided)").strip(),
        code=(code or "(not provided)").rstrip(),
        execution_result=_as_text(execution_result).strip() or "(empty)",
        stats=_as_text(stats) if stats else "(no statistical validation provided)",
    )


def _parse_llm_reply(reply: str) -> Optional[Dict[str, Any]]:
    """Read {"hypothesis_tested": bool, "critique": [str]} from an LLM reply, or None."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    tested = data.get("hypothesis_tested")
    critique = data.get("critique")
    if not isinstance(tested, bool) or not isinstance(critique, list):
        return None
    bullets = [str(c).strip() for c in critique if str(c).strip()]
    return {"hypothesis_tested": tested, "critique": bullets[:8]}


# ---------------------------------------------------------------------------
# Offline checks
# ---------------------------------------------------------------------------
def _key_terms(text: str) -> set:
    words = re.findall(r"[a-z][a-z\-]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


_STRING_LITERAL = re.compile(
    r'"""(.*?)"""|\'\'\'(.*?)\'\'\'|"((?:\\.|[^"\\\n])*)"|\'((?:\\.|[^\'\\\n])*)\'', re.DOTALL
)


def _restates_claim(piece: str, terms: set) -> bool:
    """True if `piece` holds (almost) all the key terms of the claim, i.e. it is a copy of it."""
    if len(terms) < MIN_TERMS_FOR_COPY_DETECTION:
        return False
    piece = piece.lower()
    return sum(1 for t in terms if t in piece) / len(terms) >= COPY_THRESHOLD


def hypothesis_coverage(hypothesis: str, experiment_text: str) -> float:
    """Share of the hypothesis' key terms that appear in the experiment.

    The hypothesis text itself is removed first, so a script that merely prints the claim
    does not count as testing it.
    """
    claim = re.sub(r"^\s*hypothesis\s*:\s*", "", hypothesis, flags=re.IGNORECASE).strip()
    terms = _key_terms(claim)
    if not terms:
        return 1.0
    text = experiment_text.lower()
    for variant in {hypothesis.strip().lower(), claim.lower()}:
        if variant:
            text = text.replace(variant, " ")

    # Copies of the claim that are not byte-identical: e.g. HYPOTHESIS = '...' written with repr(),
    # which changes newlines and quotes. Drop string literals and lines that restate the claim.
    def drop_copy(match):
        inner = next((g for g in match.groups() if g is not None), "")
        return " " if _restates_claim(inner, terms) else match.group(0)

    text = _STRING_LITERAL.sub(drop_copy, text)
    text = "\n".join(line for line in text.splitlines() if not _restates_claim(line, terms))
    return len([t for t in terms if t in text]) / len(terms)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_sizes(stats: Dict[str, Any]) -> List[int]:
    sizes = []
    for key in ("n_baseline", "n_treatment", "sample_size"):
        n = _to_float(stats.get(key))
        if n is not None:
            sizes.append(int(n))
    return sizes


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
def critique_experiment(
    hypothesis: str,
    execution_result: Any,
    stats: Optional[Dict[str, Any]],
    code: str = "",
    llm: Optional[Any] = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> CriticReview:
    """Review one experiment and return a CriticReview."""
    critique: List[str] = []
    stats = stats or {}

    # ---- Check 1: did the experiment test the hypothesis? ----
    tested: Optional[bool] = None
    llm_bullets: List[str] = []
    if llm is not None:
        try:
            parsed = _parse_llm_reply(
                llm.complete(build_critic_prompt(hypothesis, execution_result, stats, code))
            )
            if parsed:
                tested, llm_bullets = parsed["hypothesis_tested"], parsed["critique"]
        except Exception:  # LLM unavailable or bad reply: use the offline rules
            parsed = None
    if tested is None:
        coverage = hypothesis_coverage(hypothesis, f"{code}\n{_as_text(execution_result)}")
        tested = coverage >= MIN_HYPOTHESIS_COVERAGE
        if tested:
            critique.append(
                f"Hypothesis coverage: the experiment mentions {coverage:.0%} of the hypothesis' key terms."
            )
        else:
            critique.append(
                f"Hypothesis not tested: the experiment mentions only {coverage:.0%} of the hypothesis' "
                "key terms, so it does not measure the claimed effect."
            )
    elif not tested:
        critique.append("Hypothesis not tested: the experiment does not test the core claim of the hypothesis.")

    # ---- Check 2: sample size ----
    sizes = _group_sizes(stats)
    sample_ok: Optional[bool] = None
    if sizes:
        sample_ok = min(sizes) >= min_sample_size
        critique.append(
            f"Sample size {'adequate' if sample_ok else 'too small'}: smallest group has n={min(sizes)} "
            f"(minimum {min_sample_size})."
        )
    else:
        critique.append("Sample size not reported, so adequacy cannot be verified.")

    # ---- Check 3: statistical significance ----
    p_value = _to_float(stats.get("p_value"))
    significant: Optional[bool] = None
    if p_value is None:
        critique.append("No p-value available: the result has not been statistically validated.")
    else:
        significant = p_value < ALPHA
        critique.append(
            f"p = {p_value:.4g}: "
            + ("below 0.05, statistically significant." if significant else "not below 0.05, inconclusive.")
        )
        flag = stats.get("is_significant")
        if isinstance(flag, bool) and flag != significant:
            critique.append("The validator's is_significant flag disagrees with its p-value; the p-value was used.")

    effect = _to_float(stats.get("effect_size_cohens_d"))
    negligible = effect is not None and abs(effect) < NEGLIGIBLE_EFFECT
    if effect is not None:
        critique.append(
            f"Effect size d = {effect:.2f}" + (" (negligible in practice)." if negligible else ".")
        )

    ci = stats.get("confidence_interval_95")
    ci_has_zero = False
    if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(_to_float(v) is not None for v in ci):
        low, high = float(ci[0]), float(ci[1])
        ci_has_zero = low <= 0 <= high
        critique.append(f"95% confidence interval for the difference: [{low:.3g}, {high:.3g}].")
        if ci_has_zero and significant:
            critique.append("The interval contains zero, which contradicts the significant p-value.")

    summary = stats.get("summary")
    if isinstance(summary, str) and summary.strip():
        critique.append(f"Statistics engine summary: {summary.strip()}")

    critique.extend(llm_bullets)

    # ---- Verdict (decided by the checks, never by the LLM) ----
    if p_value is None:
        verdict = "NEEDS_REPLICATION"
        critique.append("Verdict reasoning: no statistical validation, so no conclusion can be drawn.")
    elif tested is False:
        verdict = "REJECTED"
        critique.append("Verdict reasoning: the experiment does not test the claim, so the result is not evidence for it.")
    elif not significant:
        if sample_ok is False:
            verdict = "NEEDS_REPLICATION"
            critique.append("Verdict reasoning: not significant, but the sample is too small to rule an effect out. Replicate with more data.")
        else:
            verdict = "REJECTED"
            critique.append("Verdict reasoning: p >= 0.05, so the finding is inconclusive and the hypothesis is not supported.")
    elif sample_ok is not True or negligible or ci_has_zero:
        verdict = "NEEDS_REPLICATION"
        critique.append("Verdict reasoning: significant, but sample size, effect size or interval is not convincing. Replicate before accepting.")
    else:
        verdict = "ACCEPTED"
        critique.append("Verdict reasoning: tests the claim, adequate sample, p < 0.05 and a meaningful effect.")

    return CriticReview(
        verdict=verdict,
        checks=Checks(hypothesis_tested=tested, sample_size_adequate=sample_ok, significant=significant),
        critique=critique,
        p_value=p_value,
    )


if __name__ == "__main__":
    demo_stats = {
        "p_value": 0.014,
        "is_significant": True,
        "confidence_interval_95": [12.4, 18.2],
        "effect_size_cohens_d": 0.82,
        "n_baseline": 40,
        "n_treatment": 40,
        "summary": "Statistically significant improvement over baseline with large effect size.",
    }
    demo = critique_experiment(
        "Dynamic token pruning reduces inference memory with small perplexity loss.",
        "Measured memory and perplexity for pruning vs full inference.",
        demo_stats,
    )
    print(demo.as_text())
