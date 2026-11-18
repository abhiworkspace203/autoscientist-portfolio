"""Tests for incremental ingestion (state file + pipeline filtering). Offline."""

from datetime import date
from unittest.mock import patch

from src.ingestion import pipeline
from src.ingestion.schema import Paper
from src.ingestion.state import IngestionState


def paper(pid: str) -> Paper:
    return Paper(pid, pid.split(":")[0], pid.split(":")[1], f"Title {pid}",
                 f"Abstract {pid}", ["A. Author"], "2026-10-01", "https://x")


def test_state_roundtrip(tmp_path):
    s = IngestionState(last_run=date(2026, 10, 11), seen_ids={"arxiv:1"})
    s.save(tmp_path / "state.json")
    loaded = IngestionState.load(tmp_path / "state.json")
    assert loaded.last_run == date(2026, 10, 11) and loaded.seen_ids == {"arxiv:1"}


def test_missing_or_corrupt_state_starts_fresh(tmp_path):
    assert IngestionState.load(tmp_path / "nope.json").last_run is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert IngestionState.load(bad).seen_ids == set()


def test_filter_new_and_mark_seen():
    s = IngestionState(seen_ids={"arxiv:1"})
    new = s.filter_new([paper("arxiv:1"), paper("pubmed:2")])
    assert [p.paper_id for p in new] == ["pubmed:2"]
    s.mark_seen(new, date(2026, 10, 12))
    assert s.seen_ids == {"arxiv:1", "pubmed:2"} and s.last_run == date(2026, 10, 12)


def test_resolve_start_date():
    today = date(2026, 10, 12)
    assert pipeline.resolve_start_date(today, 30, None) == date(2026, 9, 12)
    s = IngestionState(last_run=date(2026, 10, 10))
    assert pipeline.resolve_start_date(today, 30, s) == date(2026, 10, 10)


def test_incremental_run_skips_seen_papers(tmp_path):
    state_file = tmp_path / "state.json"
    batch = [paper("arxiv:1"), paper("pubmed:2")]
    with patch.object(pipeline.ArxivScraper, "search", return_value=batch[:1]), \
         patch.object(pipeline.PubMedScraper, "search", return_value=batch[1:]):
        first = pipeline.run_ingestion(["cs.AI"], "q", incremental=True, state_path=state_file)
        second = pipeline.run_ingestion(["cs.AI"], "q", incremental=True, state_path=state_file)
    assert len(first) == 2
    assert second == []          # nothing new the second time
