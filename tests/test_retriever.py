"""
Tests for the literature retriever (Member 3 - Week 2).

Builds a throwaway ChromaDB store (pytest temp folder) from a small mock corpus, then
runs mock queries with a known correct answer to measure retrieval accuracy.
The project's real store in data/vector_store is never touched.

Run:  python -m pytest tests/test_retriever.py -v
"""

import pytest

from src.rag import retriever as R
from src.rag.retriever import (
    LiteratureRetriever,
    distance_space,
    format_context,
    get_relevant_literature,
    matches_filters,
    retrieve_for_hypothesis,
    split_list,
    strip_title,
    to_cosine,
    valid_paper_ids,
)
from src.rag.vector_db import ScientificVectorDB

MOCK_PAPERS = [
    {"paper_id": "mock:transformer", "source": "arxiv", "published_date": "2026-09-01",
     "title": "Pruning Transformer Attention Heads for Memory-Efficient Inference",
     "authors": ["Ada Lovelace", "Alan Turing", "Grace Hopper"],
     "categories": ["cs.LG", "cs.AI"], "url": "https://arxiv.org/abs/mock1",
     "abstract": "Large transformer models need huge GPU memory at inference. We prune "
                 "redundant attention heads and compress the KV cache. Memory use drops "
                 "by 30 percent with less than 1 percent perplexity loss."},
    {"paper_id": "mock:robotics", "source": "arxiv", "published_date": "2025-05-10",
     "title": "Model Predictive Control for Legged Robot Locomotion",
     "authors": ["Marie Curie"], "categories": ["cs.RO"],
     "abstract": "Legged robots must plan footsteps on rough terrain. Model predictive "
                 "control optimises trajectories in real time for stable robot walking."},
    {"paper_id": "mock:drug", "source": "pubmed", "published_date": "2026-03-15",
     "title": "Graph Neural Networks Predict Drug Binding Affinity",
     "authors": ["Rosalind Franklin", "Linus Pauling"],
     "categories": ["Drug Discovery", "Machine Learning"],
     "abstract": "Predicting how strongly a drug molecule binds its protein target speeds "
                 "up drug discovery. A graph neural network learns molecular structure and "
                 "predicts binding affinity for kinase inhibitors."},
    {"paper_id": "mock:antibiotic", "source": "pubmed", "published_date": "2024-11-20",
     "title": "Antimicrobial Resistance Surveillance in Hospital Wastewater",
     "authors": ["Louis Pasteur"], "categories": ["Drug Resistance, Bacterial"],
     "abstract": "Antibiotic resistant bacteria spread through hospital wastewater. We "
                 "sequence wastewater samples to track antimicrobial resistance genes."},
    {"paper_id": "mock:sandbox", "source": "arxiv", "published_date": "2026-08-30",
     "title": "Sandboxing LLM-Generated Code with Docker Containers",
     "authors": ["Ken Thompson", "Dennis Ritchie"], "categories": ["cs.CR", "cs.AI"],
     "abstract": "Code written by language models can be unsafe. Running generated Python "
                 "code inside a locked Docker container with no network blocks malicious "
                 "commands. We measure container escape attempts and timeouts."},
    {"paper_id": "mock:climate", "source": "arxiv", "published_date": "2023-02-02",
     "title": "Deep Learning Downscaling of Climate Precipitation Forecasts",
     "authors": ["Svante Arrhenius"], "categories": ["physics.ao-ph", "cs.LG"],
     "abstract": "Global climate models are coarse. A convolutional network downscales "
                 "precipitation forecasts to kilometre resolution for regional rainfall."},
]

# (query, expected top-1 paper): the ground truth for accuracy
MOCK_QUERIES = [
    ("pruning transformer attention heads to save GPU memory", "mock:transformer"),
    ("legged robot locomotion with model predictive control", "mock:robotics"),
    ("graph neural network predicts drug binding affinity", "mock:drug"),
    ("antimicrobial resistance genes in hospital wastewater", "mock:antibiotic"),
    ("run LLM generated code safely in a docker container", "mock:sandbox"),
    ("climate precipitation forecast downscaling", "mock:climate"),
]


# All test stores live in ONE temp folder (different collection names). vector_db
# caches a Chroma client per store and never closes it; a new folder per test
# exhausts the OS open-file limit (macOS: 256) and Chroma fails with
# "unable to open database file".
@pytest.fixture(scope="module")
def store_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("vector_store")


@pytest.fixture(scope="module")
def db(store_dir):
    store = ScientificVectorDB(collection_name="retriever_test", persist_directory=store_dir)
    store.add_papers(MOCK_PAPERS)
    return store


@pytest.fixture(scope="module")
def retriever(db):
    return LiteratureRetriever(db)


# --------------------------------------------------------------- accuracy
def test_top1_retrieval_accuracy_on_mock_queries(retriever):
    correct = sum(
        1 for q, expected in MOCK_QUERIES
        if (hits := retriever.retrieve(q, top_k=1)) and hits[0]["paper_id"] == expected
    )
    print(f"\nTop-1 accuracy: {correct}/{len(MOCK_QUERIES)}")
    assert correct == len(MOCK_QUERIES)


def test_results_sorted_unique_and_ranked(retriever):
    hits = retriever.retrieve("deep learning neural network model", top_k=6, min_score=0.0)
    scores = [h["similarity"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({h["paper_id"] for h in hits}) == len(hits)          # one entry per paper
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))


def test_top_k_is_respected(retriever):
    assert len(retriever.retrieve("deep learning", top_k=2, min_score=0.0)) == 2
    assert retriever.retrieve("deep learning", top_k=0) == []


def test_result_fields_come_from_metadata(retriever):
    hit = retriever.retrieve("pruning transformer attention heads", top_k=1)[0]
    assert hit["paper_id"] == "mock:transformer"
    assert hit["authors"] == ["Ada Lovelace", "Alan Turing", "Grace Hopper"]
    assert hit["categories"] == ["cs.LG", "cs.AI"]
    assert hit["year"] == 2026 and hit["source"] == "arxiv"
    assert hit["url"] == "https://arxiv.org/abs/mock1"
    assert 0 < hit["similarity"] <= 1
    assert not hit["snippet"].startswith("Pruning Transformer")      # title not repeated


# -------------------------------------------------------------- threshold
def test_irrelevant_query_is_discarded_by_threshold(retriever):
    assert retriever.retrieve("chocolate cake baking recipe", top_k=5) == []


def test_higher_threshold_returns_fewer_papers(retriever):
    loose = retriever.retrieve("deep learning model", top_k=6, min_score=0.0)
    strict = retriever.retrieve("deep learning model", top_k=6, min_score=0.3)
    assert len(strict) < len(loose)
    assert all(h["similarity"] >= 0.3 for h in strict)


def test_empty_topic_returns_nothing(retriever):
    assert retriever.retrieve("   ") == []


def test_empty_store_returns_nothing(store_dir):
    empty = ScientificVectorDB(collection_name="empty_store", persist_directory=store_dir)
    assert LiteratureRetriever(empty).retrieve("anything") == []


# ------------------------------------------------------- metadata filters
def test_filter_by_year(retriever):
    hits = retriever.retrieve("deep learning", top_k=6, min_score=0.0, year_from=2026)
    assert hits and all(h["year"] >= 2026 for h in hits)
    hits = retriever.retrieve("deep learning", top_k=6, min_score=0.0, year_to=2024)
    assert hits and all(h["year"] <= 2024 for h in hits)


def test_filter_by_domain_tag(retriever):
    hits = retriever.retrieve("learning model code", top_k=6, min_score=0.0,
                              categories=["cs.AI"])
    assert {h["paper_id"] for h in hits} == {"mock:transformer", "mock:sandbox"}


def test_filter_tag_with_comma_inside(retriever):
    hits = retriever.retrieve("resistance", top_k=6, min_score=0.0,
                              categories=["Drug Resistance, Bacterial"])
    assert [h["paper_id"] for h in hits] == ["mock:antibiotic"]


def test_filter_by_source(retriever):
    hits = retriever.retrieve("drug resistance", top_k=6, min_score=0.0, sources=["pubmed"])
    assert hits and all(h["source"] == "pubmed" for h in hits)


def test_matches_filters_handles_missing_date():
    assert matches_filters({"published_date": ""}) is True
    assert matches_filters({"published_date": ""}, year_from=2020) is False


# ------------------------------------------------------- cosine conversion
def test_store_uses_l2_and_cosine_conversion(db):
    assert distance_space(db.collection) in {"l2", "cosine", "ip"}
    assert to_cosine(0.0, "l2") == 1.0          # identical vectors
    assert to_cosine(2.0, "l2") == 0.0          # orthogonal unit vectors
    assert to_cosine(0.25, "cosine") == 0.75
    assert to_cosine(9.0, "l2") == 0.0          # clamped


# -------------------------------------------------------------- formatting
def test_context_block_format(retriever):
    hits = retriever.retrieve("pruning transformer attention heads", top_k=1)
    text = format_context(hits)
    assert text.startswith("[Paper 1: Pruning Transformer Attention Heads for "
                           "Memory-Efficient Inference (Ada Lovelace et al., 2026)]")
    assert "ID: mock:transformer" in text
    assert "Abstract snippet: Large transformer models" in text


def test_context_respects_character_budget(retriever):
    hits = retriever.retrieve("deep learning model data", top_k=6, min_score=0.0)
    short = format_context(hits, max_chars=500)
    assert short.count("[Paper") < len(hits)


def test_valid_paper_ids_for_citation_checks(retriever):
    hits = retriever.retrieve("graph neural network drug binding", top_k=2)
    assert valid_paper_ids(hits)[0] == "mock:drug"


def test_long_paper_snippet_is_never_empty(store_dir):
    long_abstract = ("Antimicrobial resistance is rising worldwide. " * 30).strip()
    store = ScientificVectorDB(collection_name="long_paper", persist_directory=store_dir)
    store.add_papers([{"paper_id": "mock:long", "source": "pubmed",
                       "published_date": "2026-01-01", "authors": ["A. Author"],
                       "title": "Antimicrobial Resistance Review", "abstract": long_abstract}])
    assert store.collection.count() > 1            # title chunk + abstract chunks
    hit = LiteratureRetriever(store).retrieve("antimicrobial resistance review", top_k=1)[0]
    assert hit["snippet"].startswith("Antimicrobial resistance is rising")


def test_helpers():
    assert split_list("cs.AI, cs.LG") == ["cs.AI", "cs.LG"]
    assert split_list(["a", " b "]) == ["a", "b"]
    assert strip_title("My Title\n\nBody text", "My Title") == "Body text"


# ---------------------------------------------------------- public function
def test_get_relevant_literature_signature(monkeypatch, db):
    monkeypatch.setattr(R, "_default_retriever", LiteratureRetriever(db))
    hits = get_relevant_literature("legged robot locomotion", top_k=5)
    assert isinstance(hits, list) and hits[0]["paper_id"] == "mock:robotics"


def test_retrieve_for_hypothesis_bundle(monkeypatch, db):
    monkeypatch.setattr(R, "_default_retriever", LiteratureRetriever(db))
    bundle = retrieve_for_hypothesis("graph neural network drug binding affinity", top_k=2)
    assert set(bundle) == {"topic", "context", "valid_paper_ids", "papers"}
    assert bundle["valid_paper_ids"][0] == "mock:drug"
    assert "ID: mock:drug" in bundle["context"]


def test_works_on_real_sample_papers(store_dir):
    store = ScientificVectorDB(collection_name="sample_papers", persist_directory=store_dir)
    if store.load_from_json("data/sample_papers.json") == 0:
        pytest.skip("data/sample_papers.json not found")
    hits = LiteratureRetriever(store).retrieve("antimicrobial resistance", top_k=3)
    assert hits and hits[0]["paper_id"] == "pubmed:42827724"
