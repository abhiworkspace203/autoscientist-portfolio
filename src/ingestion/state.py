"""
Ingestion state: remembers when the pipeline last ran and which papers it has
already seen, so `--incremental` runs only return NEW papers.

State file (default data/ingestion_state.json):
    {"last_run": "2026-10-12", "seen_ids": ["arxiv:2410.01234v1", "pubmed:39000001", ...]}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Set

from src.ingestion.schema import Paper

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("data/ingestion_state.json")


@dataclass
class IngestionState:
    last_run: Optional[date] = None
    seen_ids: Set[str] = field(default_factory=set)

    # ---- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str = DEFAULT_STATE_PATH) -> "IngestionState":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            last = raw.get("last_run")
            return cls(
                last_run=date.fromisoformat(last) if last else None,
                seen_ids=set(raw.get("seen_ids", [])),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Corrupt state file %s (%s) - starting fresh", path, exc)
            return cls()

    def save(self, path: Path | str = DEFAULT_STATE_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "seen_ids": sorted(self.seen_ids),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    # ---- logic -------------------------------------------------------------
    def filter_new(self, papers: Iterable[Paper]) -> List[Paper]:
        """Keep only papers whose paper_id has not been seen before."""
        return [p for p in papers if p.paper_id not in self.seen_ids]

    def mark_seen(self, papers: Iterable[Paper], run_date: date) -> None:
        self.seen_ids.update(p.paper_id for p in papers)
        self.last_run = run_date
