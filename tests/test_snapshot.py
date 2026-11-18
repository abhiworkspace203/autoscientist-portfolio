"""Tests for dated snapshots written to data/raw/. Offline."""

from datetime import datetime, timezone

from src.ingestion import pipeline
from src.ingestion.schema import Paper, load_papers


def paper(pid: str) -> Paper:
    return Paper(pid, "arxiv", pid, f"Title {pid}", f"Abstract {pid}",
                 ["A. Author"], "2026-10-01", "https://x")


WHEN = datetime(2026, 10, 13, 18, 30, 0, tzinfo=timezone.utc)


def test_snapshot_filename_has_date_and_time(tmp_path):
    assert pipeline.snapshot_path(tmp_path, WHEN).name == "papers_2026-10-13_183000.json"


def test_save_snapshot_writes_papers(tmp_path):
    path = pipeline.save_snapshot([paper("a"), paper("b")], tmp_path, WHEN)
    assert path.exists()
    assert [p.paper_id for p in load_papers(path)] == ["a", "b"]


def test_snapshot_never_overwrites_previous_run(tmp_path):
    first = pipeline.save_snapshot([paper("a")], tmp_path, WHEN)
    second = pipeline.save_snapshot([paper("b")], tmp_path, WHEN)   # same second
    assert first != second
    assert load_papers(first)[0].paper_id == "a"                   # old run intact
    assert len(list(tmp_path.glob("papers_*.json"))) == 2


def test_snapshot_creates_missing_folder(tmp_path):
    path = pipeline.save_snapshot([paper("a")], tmp_path / "raw" / "nested", WHEN)
    assert path.exists()
