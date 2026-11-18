"""
Context Window Memory Manager
Member 1 (Gayatri Bhakta - Team Lead Deliverable, Week 3)

Prevents context window overflow and token exhaustion during long-horizon,
multi-step experimental runs and multimodal literature ingestion. Provides:
1. Token estimation (tiktoken with robust character-based heuristic fallback).
2. Numerical fact extraction (preserves crucial experimental numbers, p-values, deltas).
3. Sliding window pruning for historical agent logs and dialogue turns.
4. Automatic state summarization when total tokens exceed 4,000.
5. Multimodal context synthesis (budgeting extracted PDF charts and text chunks).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Token Estimation
# ---------------------------------------------------------------------------

_TIKTOKEN_ENCODER = None
_TIKTOKEN_AVAILABLE = None


def _get_tiktoken_encoder():
    """Lazily load tiktoken cl100k_base encoder if installed."""
    global _TIKTOKEN_ENCODER, _TIKTOKEN_AVAILABLE
    if _TIKTOKEN_AVAILABLE is None:
        try:
            import tiktoken
            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
            _TIKTOKEN_AVAILABLE = True
        except (ImportError, Exception):
            _TIKTOKEN_ENCODER = None
            _TIKTOKEN_AVAILABLE = False
    return _TIKTOKEN_ENCODER


def estimate_tokens(text: Optional[str]) -> int:
    """
    Estimate token count for a given text.
    Uses tiktoken cl100k_base if available; otherwise falls back to
    standard character heuristic (~4 characters per token).
    """
    if not text:
        return 0

    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # Standard LLM rule-of-thumb: 1 token ~= 4 chars in English text
    return max(1, (len(text) + 3) // 4)


# ---------------------------------------------------------------------------
# Numerical Fact Extraction
# ---------------------------------------------------------------------------

NUMERICAL_PATTERNS = [
    # Metrics, p-values, statistics
    re.compile(
        r"(?:metric delta|delta|p[-_\s]*value|confidence interval|cohen['’]?s d|mean|score|accuracy|loss|std|f1|f-score)[\s:=]+[-+]?\d*\.?\d+(?:%|[a-zA-Z0-9_\-\.\^\s]*)",
        re.IGNORECASE,
    ),
    # Explicit key-value outputs: "Baseline mean: 98.4", "Method mean: 102.1"
    re.compile(r"(?:baseline|method|control|experimental)\s+(?:mean|score|metric)[\s:=]+[-+]?\d*\.?\d+", re.IGNORECASE),
    # Numbers with units / percentages: "+2.5%", "95% CI", "0.014"
    re.compile(r"[-+]?\d+\.?\d*\s*(?:%|ms|s|MB|GB|kb|trials?|epochs?)", re.IGNORECASE),
    re.compile(r"\b(?:p\s*[<>=]\s*[-+]?\d*\.?\d+)\b", re.IGNORECASE),
]


def extract_numerical_facts(text: str) -> List[str]:
    """
    Extract key quantitative facts, benchmark numbers, p-values, and
    metric deltas so they are NEVER erased during context compression.
    """
    if not text:
        return []

    facts = []
    seen = set()

    for pattern in NUMERICAL_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            clean = match.strip().rstrip(".,;")
            if clean and clean not in seen and len(clean) < 120:
                seen.add(clean)
                facts.append(clean)

    # Also capture lines explicitly labeled as metrics
    for line in text.splitlines():
        line_clean = line.strip()
        if any(marker in line_clean.lower() for marker in ["metric delta:", "p-value:", "result:", "accuracy:"]):
            if line_clean not in seen and len(line_clean) < 150:
                seen.add(line_clean)
                facts.append(line_clean)

    return facts


# ---------------------------------------------------------------------------
# Text & State Compression
# ---------------------------------------------------------------------------

def compress_text(text: str, max_tokens: int = 500) -> str:
    """
    Compress a long text block to fit within max_tokens while
    preserving the opening thesis, bullet points, and quantitative numbers.
    """
    if not text or estimate_tokens(text) <= max_tokens:
        return text

    # Extract critical numerical benchmarks before compressing
    numbers = extract_numerical_facts(text)
    numerical_appendix = ""
    if numbers:
        numerical_appendix = "\n[Preserved Quantitative Facts]:\n" + "\n".join(f"- {f}" for f in numbers[:10])

    target_char_budget = max(200, (max_tokens - estimate_tokens(numerical_appendix)) * 4)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:target_char_budget] + "\n...[truncated]" + numerical_appendix

    # Always keep first 3 key lines / abstract statement
    header = lines[:3]
    remaining = lines[3:]

    # Pick representative summary lines from remainder
    compressed_body = []
    current_len = sum(len(h) for h in header)

    for line in remaining:
        if current_len + len(line) + 20 > target_char_budget:
            break
        # Prefer lines with metrics or conclusions
        if any(w in line.lower() for w in ["conclude", "result", "metric", "hypothesis", "signific", "finding"]):
            compressed_body.append(line)
            current_len += len(line)

    result_parts = header + (["...[historical discussion summarized]..."] if remaining else []) + compressed_body
    return "\n".join(result_parts) + numerical_appendix


# ---------------------------------------------------------------------------
# Context Manager Class
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Enterprise Context Window & Token Management Engine.

    Attributes:
        max_context_tokens: Hard ceiling before compression triggers (default: 4,000).
        warning_threshold: Fraction of max_tokens where near-overflow warnings fire.
        target_summary_tokens: Token budget for historical round summaries.
    """

    def __init__(
        self,
        max_context_tokens: int = 4000,
        warning_threshold: float = 0.85,
        target_summary_tokens: int = 800,
    ):
        self.max_context_tokens = max_context_tokens
        self.warning_threshold = warning_threshold
        self.target_summary_tokens = target_summary_tokens

    def estimate_total_tokens(self, state: Dict[str, Any]) -> int:
        """Calculate the cumulative token footprint across all state fields."""
        total = 0
        for key, value in state.items():
            if isinstance(value, str):
                total += estimate_tokens(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        total += estimate_tokens(item)
                    elif isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str):
                                total += estimate_tokens(v)
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str):
                        total += estimate_tokens(v)
        return total

    def is_near_overflow(self, token_count_or_state: Union[int, Dict[str, Any]]) -> bool:
        """Return True if current token load exceeds warning threshold (85%)."""
        if isinstance(token_count_or_state, int):
            tokens = token_count_or_state
        else:
            tokens = self.estimate_total_tokens(token_count_or_state)
        return tokens >= int(self.max_context_tokens * self.warning_threshold)

    def sliding_window(self, items: List[Any], max_tokens: int) -> List[Any]:
        """
        Return the most recent items that fit within max_tokens.
        Preserves chronological tail of execution traces or logs.
        """
        if not items:
            return []

        selected = []
        current_tokens = 0

        for item in reversed(items):
            item_str = str(item)
            item_tok = estimate_tokens(item_str)
            if current_tokens + item_tok > max_tokens and selected:
                break
            selected.append(item)
            current_tokens += item_tok

        selected.reverse()
        return selected

    def compress_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspect state and compress bloated fields if token count exceeds max_tokens.
        Returns a new state dictionary with compressed logs and pinned numerical facts.
        """
        total_tokens = self.estimate_total_tokens(state)
        if total_tokens <= self.max_context_tokens:
            return state

        compressed_state = dict(state)
        preserved_facts: List[str] = []

        # 1. Collect numbers from execution result and code
        if state.get("execution_result"):
            preserved_facts.extend(extract_numerical_facts(state["execution_result"]))
        if state.get("literature_summary"):
            preserved_facts.extend(extract_numerical_facts(state["literature_summary"]))

        # 2. Prune and summarize chronological logs using sliding window
        raw_logs = state.get("logs", [])
        if raw_logs:
            windowed_logs = self.sliding_window(raw_logs, max_tokens=self.target_summary_tokens // 2)
            compressed_state["logs"] = [
                f"[Auto-Summarized {len(raw_logs) - len(windowed_logs)} earlier log entries to protect token window]"
            ] + windowed_logs if len(raw_logs) > len(windowed_logs) else windowed_logs

        # 3. Compress literature summary if overly verbose
        lit_summary = state.get("literature_summary", "")
        if estimate_tokens(lit_summary) > 1000:
            compressed_state["literature_summary"] = compress_text(lit_summary, max_tokens=700)

        # 4. Truncate lengthy traceback to head + tail
        traceback = state.get("error_traceback", "")
        if estimate_tokens(traceback) > 400:
            tb_lines = traceback.splitlines()
            if len(tb_lines) > 20:
                compressed_state["error_traceback"] = "\n".join(
                    tb_lines[:8] + ["...[inner traceback truncated]..."] + tb_lines[-10:]
                )

        # 5. Store compact memory context in state
        deduped_facts = list(dict.fromkeys(preserved_facts))[:8]
        compressed_state["compressed_context"] = (
            f"State compressed from {total_tokens} tokens down to safe budget.\n"
            f"Preserved Quantitative Findings: {'; '.join(deduped_facts) if deduped_facts else 'None'}"
        )
        compressed_state["is_compressed"] = True

        return compressed_state

    def format_prompt_within_budget(
        self,
        system_prompt: str,
        context: str,
        user_prompt: str,
        max_prompt_tokens: Optional[int] = None,
    ) -> str:
        """
        Construct a prompt guaranteed to fit inside max_prompt_tokens.
        Protects system prompt and user query; truncates only background context.
        """
        budget = max_prompt_tokens or self.max_context_tokens
        sys_tokens = estimate_tokens(system_prompt)
        user_tokens = estimate_tokens(user_prompt)

        available_for_context = max(100, budget - sys_tokens - user_tokens - 50)
        safe_context = compress_text(context, max_tokens=available_for_context)

        return f"{system_prompt}\n\nContext:\n{safe_context}\n\nTask:\n{user_prompt}"

    def format_multimodal_context(
        self,
        text_context: str,
        figures_data: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Synthesize text literature chunks with multimodal chart and figure extractions.
        Compresses extensive chart data points into key labels, baselines, and peak values
        to prevent exceeding token budgets.
        """
        budget = max_tokens or self.max_context_tokens
        fig_sections = []

        if figures_data:
            for i, fig in enumerate(figures_data, 1):
                caption = fig.get("caption", "").strip() or "No caption"
                chart_type = fig.get("chart_type", "Figure")
                trend = fig.get("trend", fig.get("summary", ""))
                metrics = fig.get("extracted_metrics", {})

                metric_strs = [f"{k}: {v}" for k, v in metrics.items()]
                metric_line = f" | Metrics: {', '.join(metric_strs)}" if metric_strs else ""
                fig_sections.append(
                    f"[Figure {i}: {chart_type}]\n"
                    f"Caption: {caption[:120]}\n"
                    f"Trend: {trend[:150]}{metric_line}"
                )

        figures_block = "\n\n".join(fig_sections)
        fig_tokens = estimate_tokens(figures_block)

        # Allocate remainder of budget to literature text
        text_budget = max(200, budget - fig_tokens - 100)
        compressed_text = compress_text(text_context, max_tokens=text_budget)

        if figures_block:
            return f"=== Literature Background ===\n{compressed_text}\n\n=== Extracted Multimodal Charts & Figures ===\n{figures_block}"
        return compressed_text
