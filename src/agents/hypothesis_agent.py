"""HypothesisGenerator agent.

Turns a literature review (the JSON produced by ``LiteratureReviewer``) into
testable, AI-generated hypotheses with variables, metrics, methods and a
validation plan.

This module is independent: it does NOT import the other agents. It accepts a
plain ``dict`` (or JSON file) with the keys ``topic``, ``papers`` and
``analysis`` (``themes`` / ``limitations`` / ``research_gaps``).

Honesty rules enforced in code:

* ``evidence_based_observations`` (copied from the input) are kept separate
  from ``ai_generated_hypotheses``.
* Every hypothesis is tagged ``origin="ai_generated"`` and
  ``novelty_status="not_verified"``. Novelty is never claimed.
* Dataset names are *suggestions* that a human must verify.

External services / credentials: optional LLM provider via LLM_PROVIDER,
LLM_API_KEY, LLM_MODEL (and optional LLM_BASE_URL). Without them the
template-based fallback runs offline.

Run from the project root::

    python -m agents.hypothesis_generator            # built-in sample input
    python -m agents.hypothesis_generator review.json
    python -m agents.hypothesis_generator --grounded chunks.json   # RAG-grounded (needs LLM)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Hypotheses below are AI-generated proposals, not findings. Novelty has NOT "
    "been verified; a human must search the literature and review each "
    "hypothesis before running experiments. Suggested datasets are unverified "
    "suggestions."
)


# --------------------------------------------------------------------------
# LLM provider interface (configurable)
# --------------------------------------------------------------------------
class LLMError(RuntimeError):
    """Raised when the LLM provider call fails."""


class LLMClient(Protocol):
    """Anything with ``complete(prompt) -> str`` can act as an LLM provider."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class HTTPLLMClient:
    """Standard-library LLM client for Anthropic or OpenAI-compatible APIs."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        timeout: float = 90.0,
        max_tokens: int = 4000,
    ) -> None:
        self.provider = provider.lower()
        self._api_key = api_key  # never logged
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        """Send ``prompt`` and return the reply text."""
        if self.provider == "anthropic":
            url = (self.base_url or "https://api.anthropic.com") + "/v1/messages"
            headers = {
                "content-type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            }
            payload: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        elif self.provider == "openai":
            url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
            headers = {
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }

        elif self.provider == "gemini":
            url = (
                self.base_url
                or "https://generativelanguage.googleapis.com/v1beta/models"
            ) + f"/{self.model}:generateContent"

            headers = {
                "content-type": "application/json",
                "x-goog-api-key": self._api_key,
            }

            payload = {
                "contents": [
                    {"parts": [{"text": prompt}]}
                ],
                "generationConfig": {
                    "maxOutputTokens": self.max_tokens,
                    "responseMimeType": "application/json",
                    },
            }

        else:
            raise LLMError(f"Unsupported LLM_PROVIDER: {self.provider!r}")

        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if self.provider == "anthropic":
                return "".join(
                    b.get("text", "") for b in data["content"] if b.get("type") == "text"
                )
            if self.provider == "gemini":
                return "".join(
                    part.get("text", "")
                    for part in data["candidates"][0]["content"]["parts"]
                    if "text" in part
                )
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(
                f"LLM request failed: HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        
def llm_client_from_env() -> Optional[HTTPLLMClient]:
    """Build an LLM client from environment variables, or ``None``."""
    api_key, model = os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
    if not api_key or not model:
        return None
    return HTTPLLMClient(
        provider=os.getenv("LLM_PROVIDER", "anthropic"),
        api_key=api_key,
        model=model,
        base_url=os.getenv("LLM_BASE_URL") or None,
    )


def _extract_json(text: str) -> Any:
    """Parse the first JSON object found in ``text``."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(text[start : end + 1])


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------
class Observation(BaseModel):
    """An evidence-based observation taken from the supplied literature review."""

    kind: str  # "theme" | "limitation" | "research_gap"
    text: str
    source_ids: List[str] = Field(default_factory=list)
    label: str = "evidence_based"


class Hypothesis(BaseModel):
    """One AI-generated, testable hypothesis."""

    hypothesis_id: str
    statement: str
    motivation: str
    based_on_gap: str
    evidence_source_ids: List[str] = Field(default_factory=list)
    independent_variables: List[str]
    dependent_variables: List[str]
    suggested_datasets: List[str]
    evaluation_metrics: List[str]
    experiment_methods: List[str]
    validation_plan: List[str]
    origin: str = "ai_generated"
    novelty_status: str = "not_verified"


class HypothesisReport(BaseModel):
    """JSON-compatible output of ``HypothesisGenerator.generate``."""

    status: str  # "ok" | "insufficient_evidence"
    topic: str
    mode: str  # "fallback" | "llm"
    generated_at: str
    evidence_based_observations: List[Observation] = Field(default_factory=list)
    ai_generated_hypotheses: List[Hypothesis] = Field(default_factory=list)
    notes: str = ""
    disclaimer: str = DISCLAIMER


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _as_findings(items: Any, kind: str) -> List[Observation]:
    """Normalise review items (dicts or plain strings) into Observations."""
    out: List[Observation] = []
    for item in items or []:
        if isinstance(item, str):
            out.append(Observation(kind=kind, text=item))
        elif isinstance(item, dict) and item.get("text"):
            out.append(
                Observation(
                    kind=kind,
                    text=str(item["text"]),
                    source_ids=[str(i) for i in item.get("source_ids", [])],
                )
            )
    return out


# Generic datasets that ship with scikit-learn / can be generated offline,
# which matches the network-free sandbox used by CodeExecutor.
_PROXY_DATASETS = [
    "scikit-learn built-in: load_breast_cancer (verify relevance to the topic)",
    "scikit-learn built-in: load_wine (verify relevance to the topic)",
    "synthetic data from sklearn.datasets.make_classification (controllable noise)",
]


# --------------------------------------------------------------------------
# RAG-grounded hypotheses (Week 2)
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an academic researcher. Formulate novel, testable, and falsifiable "
    "scientific hypotheses that bridge research gaps between retrieved papers. "
    "Every hypothesis must cite valid Paper IDs and specific findings from the "
    "supplied evidence. Never invent papers, citations, datasets, methodologies, "
    "or experimental results. If the evidence is insufficient, clearly state "
    "this limitation."
)

# A falsifiable metric must say what is measured AND how it is compared/tested.
_TESTABLE_HINTS = (
    "than", "versus", " vs", "threshold", "at least", "at most", "below", "above",
    "exceed", "improv", "reduc", "increas", "decreas", "differ", "correlat",
    "p-value", "p <", "p<",
)


class GroundedHypothesis(BaseModel):
    """Strict schema for one RAG-grounded hypothesis."""

    title: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    proposed_solution: str = Field(min_length=1)
    cited_paper_ids: List[str] = Field(min_length=1)
    falsifiable_metric: str = Field(min_length=1)
    expected_code_modules: List[str] = Field(min_length=1)

    @field_validator("title", "problem_statement", "proposed_solution", "falsifiable_metric")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty or whitespace")
        return value

    @field_validator("cited_paper_ids", "expected_code_modules")
    @classmethod
    def _items_not_blank(cls, values: List[str]) -> List[str]:
        cleaned = [v.strip() for v in values]
        if any(not v for v in cleaned):
            raise ValueError("list items must not be empty or whitespace")
        return cleaned

    @field_validator("falsifiable_metric")
    @classmethod
    def _is_testable(cls, value: str) -> str:
        has_number = any(ch.isdigit() for ch in value)
        has_comparison = any(hint in value.lower() for hint in _TESTABLE_HINTS)
        if len(value) < 20 or not (has_number or has_comparison):
            raise ValueError(
                "must describe a measurable outcome and how it is compared or tested "
                "(e.g. 'macro-F1 higher than the baseline by at least 2 points over 10 seeds')"
            )
        return value


class SupportingFinding(BaseModel):
    """A finding quoted from one supplied paper."""

    paper_id: str = Field(min_length=1)
    finding: str = Field(min_length=1)


class GroundedHypothesisWithEvidence(GroundedHypothesis):
    """GroundedHypothesis plus the findings it relies on (kept out of the base schema)."""

    supporting_findings: List[SupportingFinding] = Field(min_length=1)
    origin: str = "ai_generated"
    novelty_status: str = "not_verified"


class EvidenceChunk(BaseModel):
    """One retrieved research-paper chunk, normalised."""

    paper_id: str
    title: str = ""
    text: str = ""
    findings: List[str] = Field(default_factory=list)
    methodology: str = ""

    def searchable_text(self) -> str:
        return " ".join([self.title, self.text, *self.findings, self.methodology])


class RejectedHypothesis(BaseModel):
    """A hypothesis the LLM produced that failed validation (never silently fixed)."""

    raw: Dict[str, Any]
    errors: List[str]


class GroundedReport(BaseModel):
    """JSON-compatible output of ``HypothesisGenerator.generate_grounded``."""

    status: Literal["ok", "partial", "insufficient_evidence", "failed"]
    topic: str
    mode: str = "llm"
    generated_at: str
    valid_paper_ids: List[str] = Field(default_factory=list)
    hypotheses: List[GroundedHypothesisWithEvidence] = Field(default_factory=list)
    rejected: List[RejectedHypothesis] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


_ID_KEYS = ("paper_id", "arxiv_id", "pmid", "id")
_TEXT_KEYS = ("text", "content", "chunk_text", "abstract", "summary")


def normalize_chunks(chunks: Any) -> Tuple[List[EvidenceChunk], List[str]]:
    """Convert retrieved chunks (dicts) to EvidenceChunks. Returns (chunks, skipped reasons)."""
    good: List[EvidenceChunk] = []
    skipped: List[str] = []
    for position, raw in enumerate(chunks or [], start=1):
        if not isinstance(raw, dict):
            skipped.append(f"chunk {position}: not a dict, skipped")
            continue
        paper_id = next((str(raw[k]).strip() for k in _ID_KEYS if raw.get(k)), "")
        text = next((str(raw[k]).strip() for k in _TEXT_KEYS if raw.get(k)), "")
        findings = raw.get("findings") or []
        if isinstance(findings, str):
            findings = [findings]
        findings = [str(f).strip() for f in findings if str(f).strip()]
        if not paper_id:
            skipped.append(f"chunk {position}: no paper id, skipped")
        elif not (text or findings):
            skipped.append(f"chunk {position} ({paper_id}): no text or findings, skipped")
        else:
            good.append(
                EvidenceChunk(
                    paper_id=paper_id,
                    title=str(raw.get("title") or "").strip(),
                    text=text,
                    findings=findings,
                    methodology=str(raw.get("methodology") or raw.get("methods") or "").strip(),
                )
            )
    return good, skipped


def load_chunks(path: str) -> List[Dict[str, Any]]:
    """Load chunks from a .json (list) or .jsonl file."""
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
    return data["chunks"] if isinstance(data, dict) and "chunks" in data else data


def _norm(text: str) -> str:
    """Lower-case, drop quote marks, collapse whitespace (for quote matching)."""
    return " ".join(re.sub(r"[\"'\u201c\u201d\u2018\u2019]", "", text.lower()).split())


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
class HypothesisGenerator:
    """Generate testable hypotheses from a literature review."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        max_hypotheses: int = 3,
        min_papers: int = 2,
        require_verbatim_quotes: bool = True,
        max_chunk_chars: int = 1500,
    ) -> None:
        self.llm = llm if llm is not None else llm_client_from_env()
        self.max_hypotheses = max(1, max_hypotheses)
        self.min_papers = max(1, min_papers)  # bridging gaps needs >= 2 papers by default
        self.require_verbatim_quotes = require_verbatim_quotes
        self.max_chunk_chars = max_chunk_chars

    # -- public API ---------------------------------------------------------
    def generate(self, review: Dict[str, Any]) -> HypothesisReport:
        """Create a ``HypothesisReport`` from review results (a dict)."""
        topic = str(review.get("topic", "unknown topic"))
        analysis = review.get("analysis") or {}
        observations = (
            _as_findings(analysis.get("themes"), "theme")
            + _as_findings(analysis.get("limitations"), "limitation")
            + _as_findings(analysis.get("research_gaps"), "research_gap")
        )
        gaps = [o for o in observations if o.kind == "research_gap"] or [
            o for o in observations if o.kind == "limitation"
        ]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if not gaps:
            return HypothesisReport(
                status="insufficient_evidence",
                topic=topic,
                mode="fallback",
                generated_at=now,
                evidence_based_observations=observations,
                notes="No research gaps or limitations were supplied, so no "
                "hypotheses were generated.",
            )

        gaps = gaps[: self.max_hypotheses]
        known_ids = {str(p.get("arxiv_id")) for p in review.get("papers", [])}
        mode, notes = "fallback", "Template-based fallback (no LLM configured)."
        hypotheses: List[Hypothesis] = []

        if self.llm is not None:
            try:
                hypotheses = self._llm_hypotheses(topic, observations, gaps, known_ids)
                mode, notes = "llm", "Generated by an LLM from the supplied evidence."
            except (LLMError, ValueError, ValidationError, KeyError, TypeError) as exc:
                logger.warning("LLM generation failed (%s); using fallback.", exc)
                notes = f"LLM generation failed ({exc}); used template fallback."
        if not hypotheses:
            hypotheses = self._fallback_hypotheses(topic, gaps)

        return HypothesisReport(
            status="ok",
            topic=topic,
            mode=mode,
            generated_at=now,
            evidence_based_observations=observations,
            ai_generated_hypotheses=hypotheses,
            notes=notes,
        )

    def generate_from_file(self, path: str) -> HypothesisReport:
        """Load a JSON review file (saved by LiteratureReviewer) and generate."""
        with open(path, "r", encoding="utf-8") as handle:
            return self.generate(json.load(handle))

    # -- RAG-grounded generation (Week 2) -------------------------------------
    def generate_grounded(
        self, chunks: List[Dict[str, Any]], topic: Optional[str] = None
    ) -> GroundedReport:
        """Generate hypotheses grounded in retrieved paper chunks.

        Never uses templates. Every hypothesis is validated: strict schema, cited
        Paper IDs must exist in ``chunks``, and each supporting finding must be a
        quote from a cited paper. Failures are reported, never silently repaired.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        evidence, notes = normalize_chunks(chunks)
        paper_text: Dict[str, str] = {}
        for chunk in evidence:  # several chunks of one paper are merged
            paper_text[chunk.paper_id] = (
                paper_text.get(chunk.paper_id, "") + " " + chunk.searchable_text()
            )
        valid_ids = sorted(paper_text)
        base: Dict[str, Any] = {
            "topic": topic or "retrieved evidence",
            "generated_at": now,
            "valid_paper_ids": valid_ids,
            "notes": notes,
        }

        if len(valid_ids) < self.min_papers:
            return GroundedReport(
                status="insufficient_evidence",
                errors=[
                    f"Insufficient evidence: {len(valid_ids)} usable paper(s) supplied, "
                    f"at least {self.min_papers} needed to bridge research gaps."
                ],
                **base,
            )
        if self.llm is None:
            return GroundedReport(
                status="failed",
                errors=["No LLM configured (set LLM_PROVIDER, LLM_API_KEY, LLM_MODEL). "
                        "Grounded generation never falls back to templates."],
                **base,
            )

        try:
            reply = self.llm.complete(self._grounded_prompt(base["topic"], evidence, valid_ids))
        except LLMError as exc:
            return GroundedReport(status="failed", errors=[str(exc)], **base)
        except Exception as exc:  # any provider/mocked failure becomes a safe result
            logger.warning("Unexpected LLM failure: %s", exc)
            return GroundedReport(
                status="failed", errors=[f"LLM call failed: {type(exc).__name__}: {exc}"], **base
            )

        try:
            data = _extract_json(reply)
        except ValueError as exc:
            return GroundedReport(
                status="failed", errors=[f"Malformed LLM response: {exc}"], **base
            )
        if not isinstance(data, dict):
            return GroundedReport(
                status="failed", errors=["Malformed LLM response: expected a JSON object."], **base
            )
        if data.get("insufficient_evidence"):
            reason = str(data.get("reason") or "no reason given")
            return GroundedReport(
                status="insufficient_evidence",
                errors=[f"The LLM reported insufficient evidence: {reason}"],
                **base,
            )
        items = data.get("hypotheses")
        if not isinstance(items, list) or not items:
            return GroundedReport(
                status="failed",
                errors=["Malformed LLM response: missing or empty 'hypotheses' list."],
                **base,
            )

        accepted: List[GroundedHypothesisWithEvidence] = []
        rejected: List[RejectedHypothesis] = []
        for raw in items[: self.max_hypotheses]:
            hypothesis, errors = self._validate_grounded(raw, paper_text)
            if hypothesis is not None:
                accepted.append(hypothesis)
            else:
                rejected.append(
                    RejectedHypothesis(raw=raw if isinstance(raw, dict) else {"value": str(raw)},
                                       errors=errors)
                )
        if not accepted:
            return GroundedReport(
                status="failed",
                rejected=rejected,
                errors=["Every hypothesis returned by the LLM failed validation."],
                **base,
            )
        return GroundedReport(
            status="ok" if not rejected else "partial",
            hypotheses=accepted,
            rejected=rejected,
            **base,
        )

    def _validate_grounded(
        self, raw: Any, paper_text: Dict[str, str]
    ) -> Tuple[Optional[GroundedHypothesisWithEvidence], List[str]]:
        """Return (hypothesis, []) if valid, else (None, error messages)."""
        if not isinstance(raw, dict):
            return None, ["hypothesis is not a JSON object"]
        candidate = dict(raw)
        candidate["origin"] = "ai_generated"  # enforced, never trusted from the LLM
        candidate["novelty_status"] = "not_verified"
        try:
            hypothesis = GroundedHypothesisWithEvidence(**candidate)
        except ValidationError as exc:
            return None, [
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            ]

        errors: List[str] = []
        unknown = sorted(set(hypothesis.cited_paper_ids) - set(paper_text))
        if unknown:
            errors.append(f"cites Paper ID(s) not in the supplied evidence: {unknown}")
        for finding in hypothesis.supporting_findings:
            if finding.paper_id not in hypothesis.cited_paper_ids:
                errors.append(
                    f"supporting finding is attributed to {finding.paper_id!r}, "
                    "which is not in cited_paper_ids"
                )
            elif finding.paper_id in paper_text and self.require_verbatim_quotes:
                quote = _norm(finding.finding.strip(" .\u2026"))
                if len(quote.split()) < 4:
                    errors.append(f"finding for {finding.paper_id} is too short to verify: "
                                  f"{finding.finding!r}")
                elif quote not in _norm(paper_text[finding.paper_id]):
                    errors.append(
                        f"finding for {finding.paper_id} is not a quote from that paper's "
                        f"evidence: {finding.finding!r}"
                    )
        cited_with_finding = {f.paper_id for f in hypothesis.supporting_findings}
        missing = sorted(set(hypothesis.cited_paper_ids) - cited_with_finding - set(unknown))
        if missing:
            errors.append(f"no supporting finding given for cited paper(s): {missing}")
        return (None, errors) if errors else (hypothesis, [])

    def _grounded_prompt(
        self, topic: str, evidence: List[EvidenceChunk], valid_ids: List[str]
    ) -> str:
        blocks = []
        for chunk in evidence:
            lines = [f"[PAPER_ID: {chunk.paper_id}]"]
            if chunk.title:
                lines.append(f"Title: {chunk.title}")
            if chunk.text:
                lines.append(f"Text: {chunk.text[: self.max_chunk_chars]}")
            if chunk.findings:
                lines.append("Findings: " + " | ".join(chunk.findings))
            if chunk.methodology:
                lines.append(f"Methodology: {chunk.methodology[: self.max_chunk_chars]}")
            blocks.append("\n".join(lines))
        return (
            f"{SYSTEM_PROMPT}\n\nTopic: {topic}\n\n"
            f"Evidence (the ONLY allowed sources). Valid Paper IDs: {', '.join(valid_ids)}\n\n"
            + "\n\n".join(blocks)
            + f"\n\nPropose up to {self.max_hypotheses} hypotheses that bridge gaps between "
            "these papers. Rules:\n"
            "- cited_paper_ids may only contain the valid Paper IDs listed above.\n"
            "- supporting_findings: for EVERY cited paper, copy a phrase of 4 to 30 words "
            "EXACTLY from that paper's evidence text. Do not paraphrase.\n"
            "- falsifiable_metric: state the measured quantity, the comparison, and the "
            "result that would refute the hypothesis.\n"
            "- expected_code_modules: short snake_case names of Python modules needed to "
            "run the experiment.\n"
            "- Use only datasets and methods that appear in the evidence.\n"
            '- If the evidence is insufficient reply {"insufficient_evidence": true, '
            '"reason": "..."}.\n\n'
            'Reply with JSON only: {"hypotheses":[{"title":"...","problem_statement":"...",'
            '"proposed_solution":"...","cited_paper_ids":["id"],'
            '"supporting_findings":[{"paper_id":"id","finding":"exact quote"}],'
            '"falsifiable_metric":"...","expected_code_modules":["..."]}]}'
        )

    # -- fallback -----------------------------------------------------------
    @staticmethod
    def _fallback_hypotheses(topic: str, gaps: List[Observation]) -> List[Hypothesis]:
        """Deterministic templates. Generic by design: a human should refine."""
        hypotheses = []
        for index, gap in enumerate(gaps, start=1):
            short_gap = gap.text if len(gap.text) <= 200 else gap.text[:197] + "..."
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"H{index}",
                    statement=(
                        "A model variant designed to address the issue below achieves "
                        "a higher mean cross-validated macro-F1 than an otherwise "
                        "identical baseline on a controlled proxy task. "
                        f"Issue: {short_gap}"
                    ),
                    motivation=(
                        f"The literature review for '{topic}' reported the issue "
                        "above (see evidence_source_ids). The hypothesis turns it "
                        "into a measurable comparison."
                    ),
                    based_on_gap=gap.text,
                    evidence_source_ids=gap.source_ids,
                    independent_variables=[
                        "model variant (baseline vs. modified)",
                        "noise level or training-set size (factor related to the issue)",
                    ],
                    dependent_variables=["macro-F1", "accuracy"],
                    suggested_datasets=list(_PROXY_DATASETS),
                    evaluation_metrics=[
                        "macro-F1 (mean and std over seeds)",
                        "accuracy",
                        "paired Wilcoxon signed-rank test p-value",
                    ],
                    experiment_methods=[
                        "Controlled A/B comparison using scikit-learn pipelines",
                        "Repeated stratified k-fold cross-validation with fixed seeds",
                        "Ablation: remove the modification and re-measure",
                    ],
                    validation_plan=[
                        "Run at least 10 random seeds and report mean and std.",
                        "Compare against a baseline with identical preprocessing.",
                        "Apply a paired significance test and report effect size.",
                        "Repeat on a second dataset to check consistency.",
                        "Manually search the literature to check novelty before "
                        "making any novelty claim.",
                        "Human review of code and results before drafting a paper.",
                    ],
                )
            )
        return hypotheses

    # -- LLM path -----------------------------------------------------------
    def _llm_hypotheses(
        self,
        topic: str,
        observations: List[Observation],
        gaps: List[Observation],
        known_ids: set,
    ) -> List[Hypothesis]:
        assert self.llm is not None
        evidence = "\n".join(
            f"- ({o.kind}) {o.text} [sources: {', '.join(o.source_ids) or 'n/a'}]"
            for o in observations
        )
        prompt = (
            f"Topic: {topic}\n\nEvidence from a literature review:\n{evidence}\n\n"
            f"Propose up to {len(gaps)} testable hypotheses that address the gaps. "
            "Do not claim novelty. Do not invent papers or results. Prefer "
            "experiments runnable offline with NumPy/pandas/scikit-learn.\n\n"
            'Reply with JSON only: {"hypotheses":[{"statement":"...",'
            '"motivation":"...","based_on_gap":"...","evidence_source_ids":["id"],'
            '"independent_variables":["..."],"dependent_variables":["..."],'
            '"suggested_datasets":["..."],"evaluation_metrics":["..."],'
            '"experiment_methods":["..."],"validation_plan":["..."]}]}'
        )
        data = _extract_json(self.llm.complete(prompt))
        results: List[Hypothesis] = []
        for index, raw in enumerate(data["hypotheses"][: self.max_hypotheses], start=1):
            raw = dict(raw)
            raw["hypothesis_id"] = f"H{index}"
            raw["origin"] = "ai_generated"  # enforced, never trusted from the LLM
            raw["novelty_status"] = "not_verified"
            raw["evidence_source_ids"] = [
                i for i in raw.get("evidence_source_ids", []) if i in known_ids
            ]
            results.append(Hypothesis(**raw))
        if not results:
            raise ValueError("LLM returned no hypotheses")
        return results


# --------------------------------------------------------------------------
# Sample input / CLI
# --------------------------------------------------------------------------
SAMPLE_REVIEW: Dict[str, Any] = {
    "topic": "example topic",
    "papers": [{"arxiv_id": "EXAMPLE-ID-1", "title": "(placeholder, not a real paper)"}],
    "analysis": {
        "themes": [{"text": "example theme", "source_ids": ["EXAMPLE-ID-1"]}],
        "limitations": [],
        "research_gaps": [
            {
                "text": "Possible gap: models degrade when training data is noisy.",
                "source_ids": ["EXAMPLE-ID-1"],
            }
        ],
    },
}


def main() -> None:
    """Command-line entry point (run from the project root)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generator = HypothesisGenerator()
    if len(sys.argv) > 2 and sys.argv[1] == "--grounded":
        grounded = generator.generate_grounded(load_chunks(sys.argv[2]))
        print(json.dumps(grounded.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return
    if len(sys.argv) > 1:
        report = generator.generate_from_file(sys.argv[1])
    else:
        report = generator.generate(SAMPLE_REVIEW)
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()



def generate_hypothesis(literature_summary: str) -> str:
    """Generate one hypothesis from a literature summary."""
    summary = (literature_summary or "").strip()

    if not summary:
        return "Insufficient evidence: literature summary is empty."

    generator = HypothesisGenerator()

    report = generator.generate({
        "topic": "Scientific Discovery",
        "analysis": {
            "research_gaps": [
                {
                    "text": summary[:200],
                    "source_ids": []
                }
            ]
        }
    })

    hypotheses = report.ai_generated_hypotheses

    if hypotheses:
        return f"Hypothesis: {hypotheses[0].statement}"

    return (
        "Insufficient evidence to generate a hypothesis. "
        "Please provide a research gap or limitation."
    )


def generate_all_hypotheses(literature_summary: str) -> list[str]:
    """Generate all hypotheses available from the supplied summary."""
    summary = (literature_summary or "").strip()

    if not summary:
        return []

    generator = HypothesisGenerator()

    report = generator.generate({
        "topic": "Scientific Discovery",
        "analysis": {
            "research_gaps": [
                {
                    "text": summary[:200],
                    "source_ids": []
                }
            ]
        }
    })

    return [
        f"Hypothesis: {hypothesis.statement}"
        for hypothesis in report.ai_generated_hypotheses
    ]
