"""Checks that the agreed project structure on main is intact (Member 5)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PATHS = [
    "requirements.txt",
    "README.md",
    "data",
    "docs/role_specifications.md",
    "docs/ingestion.md",
    "docs/pr_review_standards.md",
    ".github/pull_request_template.md",
    "src/orchestrator.py",
    "src/agents/literature_agent.py",
    "src/agents/hypothesis_agent.py",
    "src/agents/code_executor_agent.py",
    "src/ingestion/arxiv_scraper.py",
    "src/ingestion/pubmed_scraper.py",
    "src/ingestion/pipeline.py",
    "src/ingestion/schema.py",
    "src/ingestion/state.py",
    "src/rag/vector_db.py",
    "tests",
]


@pytest.mark.parametrize("path", REQUIRED_PATHS)
def test_required_path_exists(path):
    assert (ROOT / path).exists(), f"Missing from project structure: {path}"


def test_requirements_lists_core_packages():
    text = (ROOT / "requirements.txt").read_text().lower()
    for package in ["pydantic", "langgraph", "chromadb", "pytest"]:
        assert package in text, f"{package} missing from requirements.txt"
