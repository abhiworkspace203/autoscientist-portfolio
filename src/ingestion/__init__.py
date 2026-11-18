"""Data ingestion pipelines (arXiv, PubMed) — owned by Member 3."""

from src.ingestion.schema import Paper, deduplicate, load_papers, save_papers

__all__ = ["Paper", "deduplicate", "load_papers", "save_papers"]
