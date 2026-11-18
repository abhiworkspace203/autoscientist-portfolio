# Member 3: Data Ingestion, Retrieval and Figures

Everything Member 3 (branch `afrah`) built across Weeks 1–3, in one page. Detailed guides: [retriever.md](retriever.md), [figure_extractor.md](figure_extractor.md).

## At a glance

| Week | Module | What it does |
| --- | --- | --- |
| 1 | `src/ingestion/arxiv_scraper.py` | Fetches recent arXiv papers by category and date range |
| 1 | `src/ingestion/pubmed_scraper.py` | Fetches PubMed abstracts via NCBI Entrez (esearch + efetch) |
| 1 | `src/ingestion/schema.py` | One shared `Paper` format for both sources, de-duplication, save/load JSON |
| 1 | `src/ingestion/pipeline.py` | Runs both scrapers; incremental mode (only new papers) and dated snapshots in `data/raw/` |
| 1 | `src/ingestion/state.py` | Remembers the last run and every paper already seen |
| 2 | `src/rag/retriever.py` | Searches the ChromaDB store: cosine similarity, score threshold, year/tag/source filters, prompt-ready context |
| 2 | `src/rag/grounding.py` | Builds the grounded RAG prompt for the Hypothesis agent and rejects citations of papers that were not retrieved |
| 3 | `src/ingestion/figure_extractor.py` | Extracts figures from paper PDFs (images and vector charts), drops logos, merges image strips, pairs captions |
| 3 | `examples/extract_sample_figures.py` | Downloads 3 sample arXiv papers and extracts their figures |

## How the pieces connect

```
arXiv / PubMed --> scrapers --> Paper JSON --> ChromaDB (vector_db, Member 4)
                                                   |
                                                   v
                         retriever --> grounding --> Hypothesis agent (Member 2)

paper PDFs --> figure_extractor --> data/extracted_figures/ --> chart reader / VLM (Member 2)
```

## Unified paper format

`paper_id, source, source_id, title, abstract, authors, published_date, url, pdf_url, doi, categories, journal, retrieved_at`

`paper_id` looks like `arxiv:2610.12445v1` or `pubmed:42827724`. Records missing a title, abstract, authors or date are skipped.

## Commands

```bash
# Week 1: collect papers
python -m src.ingestion.pipeline --arxiv-max 5 --pubmed-max 5
python -m src.ingestion.pipeline --incremental --out data/new_papers.json

# Week 2: retrieve literature for a topic
python -m src.rag.retriever "antimicrobial resistance" --top-k 3
python -m src.rag.retriever "agent safety" --categories cs.AI --year-from 2026

# Week 3: extract figures
python -m src.ingestion.figure_extractor paper.pdf --paper-id arxiv:2610.12452v1
python examples/extract_sample_figures.py
```

```python
from src.rag.grounding import build_grounded_prompt, check_hypothesis_citations
from src.ingestion.figure_extractor import load_extracted_figures

g = build_grounded_prompt("antimicrobial resistance", top_k=5)   # g["prompt"], g["valid_paper_ids"]
figures = load_extracted_figures()                               # path + caption of every figure
```

## Results

- **Week 1:** 10 live papers (5 arXiv, 5 PubMed) in `data/sample_scraped_papers.json`.
- **Week 2:** 100% top-1 accuracy on 6 mock queries with known answers; off-topic queries are rejected by the similarity threshold.
- **Week 3:** 3 real arXiv papers gave 23 figures, every one paired with its caption (`data/sample_figure_extractions.json`). On BrickBench, 17 loose image strips became 13 complete figures.

## Tests

```bash
python -m pytest tests/test_ingestion.py tests/test_state.py tests/test_snapshot.py \
                 tests/test_retriever.py tests/test_grounding.py tests/test_figure_extractor.py -q
```

All tests run offline: APIs are mocked, PDFs are generated on the fly, and retriever tests use a temporary ChromaDB folder, so `data/vector_store` is never touched.

## Setup notes

- `.env` holds `NCBI_EMAIL` (and optionally `NCBI_API_KEY`); it is never committed.
- On Intel Macs, `sentence-transformers` cannot load (PyTorch is capped at 2.2), so `vector_db` falls back to a hash embedding. Retrieval still works but matches words rather than meaning.
- `data/pdfs/` and `data/extracted_figures/` are git-ignored (large generated files).
