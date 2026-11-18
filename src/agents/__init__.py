"""Agents package for the Autonomous Multimodal Scientific Discovery Agent.

Agent classes and functions are imported lazily to reduce circular imports
and avoid unnecessary module loading.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "LiteratureReviewer",
    "HypothesisGenerator",
    "CodeExecutor",
    "run_literature_review",
    "filter_papers_by_topic",
    "generate_hypothesis",
    "generate_all_hypotheses",
    "generate_and_run_code",
]

_EXPORTS = {
    "LiteratureReviewer": ".literature_agent",
    "run_literature_review": ".literature_agent",
    "filter_papers_by_topic": ".literature_agent",
    "HypothesisGenerator": ".hypothesis_agent",
    "generate_hypothesis": ".hypothesis_agent",
    "generate_all_hypotheses": ".hypothesis_agent",
    "CodeExecutor": ".code_executor_agent",
    "generate_and_run_code": ".code_executor_agent",
}

if TYPE_CHECKING:
    from .literature_agent import (
        LiteratureReviewer,
        run_literature_review,
        filter_papers_by_topic,
    )
    from .hypothesis_agent import (
        HypothesisGenerator,
        generate_hypothesis,
        generate_all_hypotheses,
    )
    from .code_executor_agent import (
        CodeExecutor,
        generate_and_run_code,
    )


def __getattr__(name: str) -> Any:
    """Import an exported agent class or function on first access."""
    if name not in _EXPORTS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module = importlib.import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)

    # Cache the resolved attribute to avoid repeated imports/lookups.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in module introspection."""
    return sorted(set(globals()) | set(__all__))