"""
Tests for Scientific Vector Database (Member 4 Deliverable).
"""

import pytest
from src.rag.vector_db import ScientificVectorDB


@pytest.fixture
def db(tmp_path):
    return ScientificVectorDB(
        collection_name="scientific_papers_test",
        persist_directory=tmp_path / "vector_store",
    )


def test_vector_db_initialization(db):
    assert db.count() == 0


def test_vector_db_add_papers(db):
    papers = [
        {
            "paper_id": "test:001",
            "title": "Scaling Transformer Memory Systems",
            "abstract": "We investigate KV cache compression techniques.",
            "categories": ["cs.AI", "cs.LG"],
        },
        {
            "paper_id": "test:002",
            "title": "Autonomous Robotics Planning",
            "abstract": "Trajectory optimization using model predictive control.",
            "categories": ["cs.RO"],
        },
    ]
    added = db.add_papers(papers)
    assert added == 2
    assert db.count() == 2


def test_vector_db_search(db):
    papers = [
        {
            "paper_id": "test:001",
            "title": "Scaling Transformer Memory Systems",
            "abstract": "We investigate KV cache compression techniques.",
            "categories": ["cs.AI", "cs.LG"],
        },
        {
            "paper_id": "test:002",
            "title": "Autonomous Robotics Planning",
            "abstract": "Trajectory optimization using model predictive control.",
            "categories": ["cs.RO"],
        },
    ]
    db.add_papers(papers)
    results = db.search("Transformer Memory Cache", top_k=1)
    assert len(results) == 1
    assert results[0]["paper_id"] == "test:001"
    assert "similarity_score" in results[0]


def test_vector_db_load_from_sample_json(db):
    count = db.load_from_json("data/sample_papers.json")
    assert count == 30  # chunks stored from the 10 sample papers
    assert db.count() == 10  # distinct papers stored
    
def test_search_handles_invalid_input(tmp_path):
    db = ScientificVectorDB(persist_directory=tmp_path)

    assert db.search("   ") == []

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        db.search("science", top_k=0)