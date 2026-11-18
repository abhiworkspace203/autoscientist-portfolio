"""Tests for RAG grounding helpers (Member 3 - Week 2, Day 4). Offline, temp DB."""

import pytest
from pydantic import BaseModel

from src.rag import retriever as R
from src.rag.grounding import (
    GROUNDING_INSTRUCTION,
    NO_PAPERS_MESSAGE,
    build_grounded_prompt,
    check_hypothesis_citations,
    find_invalid_citations,
)
from src.rag.retriever import LiteratureRetriever
from src.rag.vector_db import ScientificVectorDB

PAPERS = [
    {"paper_id": "mock:drug", "source": "pubmed", "published_date": "2026-03-15",
     "title": "Graph Neural Networks Predict Drug Binding Affinity",
     "authors": ["Rosalind Franklin", "Linus Pauling"], "categories": ["Machine Learning"],
     "abstract": "A graph neural network learns molecular structure and predicts drug "
                 "binding affinity for kinase inhibitors, speeding up drug discovery."},
    {"paper_id": "mock:antibiotic", "source": "pubmed", "published_date": "2024-11-20",
     "title": "Antimicrobial Resistance Surveillance in Hospital Wastewater",
     "authors": ["Louis Pasteur"], "categories": ["Drug Resistance, Bacterial"],
     "abstract": "We sequence hospital wastewater to track antimicrobial resistance genes "
                 "carried by antibiotic resistant bacteria."},
]


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    # one shared throwaway store for the module (see note in test_retriever.py)
    store = ScientificVectorDB(collection_name="grounding_test",
                               persist_directory=tmp_path_factory.mktemp("vector_store"))
    store.add_papers(PAPERS)
    return store


@pytest.fixture(autouse=True)
def temp_retriever(db, monkeypatch):
    monkeypatch.setattr(R, "_default_retriever", LiteratureRetriever(db))


class GroundedHypothesis(BaseModel):          # mirrors the Week 2 spec schema
    title: str
    problem_statement: str
    proposed_solution: str
    cited_paper_ids: list[str]
    falsifiable_metric: str
    expected_code_modules: list[str]


def make(cited):
    return GroundedHypothesis(title="t", problem_statement="p", proposed_solution="s",
                              cited_paper_ids=cited, falsifiable_metric="m",
                              expected_code_modules=["train.py"])


def test_prompt_contains_instruction_context_and_allowed_ids():
    g = build_grounded_prompt("graph neural network drug binding affinity", top_k=2)
    assert g["valid_paper_ids"][0] == "mock:drug"
    assert GROUNDING_INSTRUCTION in g["prompt"]
    assert "[Paper 1: Graph Neural Networks Predict Drug Binding Affinity" in g["prompt"]
    assert "Allowed Paper IDs for cited_paper_ids: mock:drug" in g["prompt"]
    assert '"cited_paper_ids"' in g["prompt"]          # output schema included


def test_prompt_when_nothing_relevant():
    g = build_grounded_prompt("chocolate cake baking recipe")
    assert g["valid_paper_ids"] == [] and g["papers"] == []
    assert NO_PAPERS_MESSAGE in g["prompt"]
    assert "Allowed Paper IDs for cited_paper_ids: (none)" in g["prompt"]


def test_find_invalid_citations():
    assert find_invalid_citations(["a", "b"], ["a", "b", "c"]) == []
    assert find_invalid_citations(["a", "fake:1", "fake:1"], ["a"]) == ["fake:1"]


def test_grounded_hypothesis_passes():
    g = build_grounded_prompt("graph neural network drug binding affinity", top_k=2)
    assert check_hypothesis_citations(make(["mock:drug"]), g["valid_paper_ids"]) == []


def test_hallucinated_citation_is_flagged():
    g = build_grounded_prompt("graph neural network drug binding affinity", top_k=2)
    problems = check_hypothesis_citations(make(["mock:drug", "arxiv:9999.99999"]),
                                          g["valid_paper_ids"])
    assert problems == ["cites unknown paper ID: arxiv:9999.99999"]


def test_hypothesis_without_citations_is_flagged_and_dicts_work():
    assert check_hypothesis_citations({"cited_paper_ids": []}, ["mock:drug"]) == [
        "cites 0 paper(s); at least 1 required"]
    assert check_hypothesis_citations({"cited_paper_ids": ["mock:drug"]}, ["mock:drug"]) == []
