"""
Scraper output checks (Member 5).

Checks the Week 1 deliverable data/sample_scraped_papers.json from the scrapers
(Member 3) and that it hands over cleanly to the vector database (Member 4).
All tests are offline. Mocked scraper tests live in tests/test_ingestion.py.
"""
import re
import time
from pathlib import Path

import pytest

from src.ingestion.schema import REQUIRED_FIELDS, load_papers
from src.rag.vector_db import ScientificVectorDB

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_scraped_papers.json"


@pytest.fixture(scope="module")
def papers():
    return load_papers(SAMPLE)


def test_sample_file_has_ten_papers(papers):
    assert len(papers) == 10


def test_sample_has_five_arxiv_and_five_pubmed(papers):
    sources = [p.source for p in papers]
    assert sources.count("arxiv") == 5
    assert sources.count("pubmed") == 5


def test_every_paper_has_required_fields(papers):
    for p in papers:
        assert p.is_valid(), f"{p.paper_id} is missing one of {REQUIRED_FIELDS}"


def test_dates_are_iso_format(papers):
    for p in papers:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.published_date), p.paper_id


def test_paper_ids_are_unique_and_prefixed(papers):
    ids = [p.paper_id for p in papers]
    assert len(ids) == len(set(ids))
    assert all(i.startswith(("arxiv:", "pubmed:")) for i in ids)


def test_arxiv_papers_have_pdf_links(papers):
    for p in papers:
        if p.source == "arxiv":
            assert p.pdf_url, p.paper_id


def test_scraped_papers_load_into_vector_db():
    db = ScientificVectorDB()
    count = db.load_from_json(str(SAMPLE))
    assert count in (10, 30)
    assert db.count() == 10


def test_vector_db_finds_a_paper_by_its_own_title(papers):
    db = ScientificVectorDB()
    db.load_from_json(str(SAMPLE))
    target = papers[0]
    hits = db.search(target.title, top_k=3)
    assert target.paper_id in [h["paper_id"] for h in hits]


def test_vector_db_query_is_fast():
    db = ScientificVectorDB()
    db.load_from_json(str(SAMPLE))
    start = time.perf_counter()
    db.search("machine learning", top_k=3)
    assert time.perf_counter() - start < 1.0
