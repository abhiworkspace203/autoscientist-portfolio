"""
Scientific Vector Database Store
Member 4 Deliverable (Data Ingestion & Vector DB Integration)

Provides semantic embedding storage and similarity search for scientific papers
ingested from arXiv and PubMed APIs.
Supports ChromaDB with in-memory semantic fallback for resilient local testing.
"""

from __future__ import annotations
import hashlib
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERS_FILE = PROJECT_ROOT / "data" / "sample_papers.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

COLLECTION_NAME = "scientific_papers"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

_embedding_function = None
_collection_cache = {}
_client_cache = {}


class LocalHashEmbeddingFunction:
    """Small offline fallback used if sentence-transformers is unavailable."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = []

        for text in input:
            vector = [0.0] * EMBEDDING_SIZE

            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % EMBEDDING_SIZE
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign

            magnitude = math.sqrt(sum(value * value for value in vector))
            if magnitude:
                vector = [value / magnitude for value in vector]

            vectors.append(vector)

        return vectors

    def name(self) -> str:
        return "local_hash"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


def _get_embedding_function():
    """Create the pretrained embedder, or use a deterministic offline fallback."""
    global _embedding_function

    if _embedding_function is None:
        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            _embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=MODEL_NAME
            )
        except Exception as error:
            warnings.warn(
                "Could not load sentence-transformers; using a local "
                f"hash-based fallback instead. Details: {error}",
                RuntimeWarning,
            )
            _embedding_function = LocalHashEmbeddingFunction()

    return _embedding_function


def _get_or_create_collection(
    collection_name: str,
    persist_directory: str | Path,
):
    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)
    cache_key = (str(path.resolve()), collection_name)

    if cache_key not in _collection_cache:
        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=_get_embedding_function(),
        )
        _client_cache[cache_key] = client
        _collection_cache[cache_key] = collection

    return _collection_cache[cache_key]


def get_collection():
    """Open the project's persistent scientific_papers collection."""
    return _get_or_create_collection(COLLECTION_NAME, VECTOR_STORE_DIR)


def _add_papers_to_collection(
    papers: list[dict[str, Any]],
    collection,
) -> int:
    ids = []
    documents = []
    metadatas = []

    for paper in papers:
        paper_id = str(paper.get("paper_id") or paper.get("source_id") or "")
        title = str(paper.get("title") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        text = f"{title}\n\n{abstract}".strip()

        if not paper_id or not text:
            continue

        authors = paper.get("authors") or []
        categories = paper.get("categories") or []

        if not isinstance(authors, (list, tuple)):
            authors = [authors]
        if not isinstance(categories, (list, tuple)):
            categories = [categories]

        for chunk_number, chunk in enumerate(_splitter.split_text(text)):
            chunk_id = hashlib.sha256(
                f"{paper_id}:{chunk_number}".encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "source": str(paper.get("source") or "unknown"),
                    "authors": ", ".join(str(value) for value in authors),
                    "published_date": str(paper.get("published_date") or ""),
                    "url": str(paper.get("url") or ""),
                    "categories": ", ".join(
                        str(value) for value in categories
                    ),
                    "chunk_number": chunk_number,
                }
            )

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    return len(ids)


def add_papers(papers: list[dict[str, Any]]) -> int:
    """Split papers into chunks and store them in the persistent collection."""
    return _add_papers_to_collection(papers, get_collection())


def _load_papers_from_json(json_path: str | Path, collection) -> int:
    path = Path(json_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as file:
        papers = json.load(file)

    if not isinstance(papers, list):
        raise ValueError("The papers JSON file must contain a list of papers.")

    return _add_papers_to_collection(papers, collection)


def load_papers_from_json(json_path: str | Path = PAPERS_FILE) -> int:
    """Load Member 3's JSON paper list into the persistent collection."""
    return _load_papers_from_json(json_path, get_collection())


def _count_papers(collection) -> int:
    stored = collection.get(include=["metadatas"])
    paper_ids = {
        metadata.get("paper_id")
        for metadata in (stored.get("metadatas") or [])
        if metadata and metadata.get("paper_id")
    }
    return len(paper_ids)


def _search_collection(
    collection,
    query_text: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not query_text.strip():
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if collection.count() == 0:
        return []

    matches = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for document, metadata, distance in zip(
        matches["documents"][0],
        matches["metadatas"][0],
        matches["distances"][0],
    ):
        metadata = metadata or {}
        distance = float(distance)
        similarity_score = 1.0 / (1.0 + max(distance, 0.0))

        results.append(
            {
                "paper_id": metadata.get("paper_id"),
                "similarity_score": round(similarity_score, 6),
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return results


def query_papers(
    query_text: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search the persistent collection for relevant paper chunks."""
    return _search_collection(get_collection(), query_text, top_k)


class ScientificVectorDB:
    """Class interface used by the rest of the project and its tests."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str | Path | None = None,
    ):
        directory = persist_directory or VECTOR_STORE_DIR
        self.collection = _get_or_create_collection(collection_name, directory)

    def add_papers(self, papers: list[dict[str, Any]]) -> int:
        return _add_papers_to_collection(papers, self.collection)

    def load_from_json(self, json_path: str | Path = PAPERS_FILE) -> int:
        return _load_papers_from_json(json_path, self.collection)

    def count(self) -> int:
        return _count_papers(self.collection)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return _search_collection(self.collection, query, top_k)


if __name__ == "__main__":
    chunk_count = load_papers_from_json()
    print(f"Stored {chunk_count} paper chunks.")

    for result in query_papers(
        "AI time horizons and benchmarks",
        top_k=3,
    ):
        print(f"\nTitle: {result['metadata']['title']}")
        print(result["text"])
