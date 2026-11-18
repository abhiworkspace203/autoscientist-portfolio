# Week 1 Integration Report

**Prepared by:** Member 5 (Code Executor Agent, Testing \& GitHub Integration)
**Branch:** `sachin`
**Date:** 15 October 2026
**Project:** Autonomous Multimodal Scientific Discovery Agent (AutoScientist AI)

## 1\. Summary

The Week 1 pipeline runs end to end: **Literature Reviewer -> Hypothesis Generator -> Code Executor**, coordinated by the LangGraph orchestrator. The arXiv/PubMed ingestion, the vector database and all three agents are in place, and the full test suite passes (**67 of 67 tests**, 81% line coverage of `src/`).

Week 1 delivers working skeletons. Several connections between components are not wired yet (see section 6), and the agents use rule-based placeholders instead of an LLM.

## 2\. Component status

|Component|Owner|Status|Notes|
|-|-|-|-|
|Orchestrator (LangGraph)|Member 1|Working|3 nodes, shared state, audit logs|
|Literature Reviewer agent|Member 2|Working (rule-based)|Reads `data/sample\_papers.json`, no LLM yet|
|Hypothesis Generator agent|Member 2|Working (placeholder)|Returns fixed templates|
|arXiv + PubMed scrapers|Member 3|Working|Unified `Paper` schema, dedup, incremental state, timeout handling|
|Scraped dataset|Member 3|Done|`data/sample\_scraped\_papers.json`: 10 papers (5 arXiv + 5 PubMed), all valid|
|Vector database|Member 4|Working (keyword scoring)|In-memory term overlap, not embeddings|
|Code Executor agent|Member 5|Working (skeleton)|Prompt template, script generation, subprocess runner with timeout|
|Tests, PR template, review standards|Member 5|Done|See sections 3 and 4|

## 3\. Test results

Command: `python -m pytest -v`

|Test file|Owner|Tests|Passed|Failed|
|-|-|-|-|-|
|tests/test\_agents.py|Member 1 / 2|5|5|0|
|tests/test\_code\_executor.py|Member 5|11|11|0|
|tests/test\_ingestion.py|Member 3|10|10|0|
|tests/test\_state.py|Member 3|5|5|0|
|tests/test\_orchestrator.py|Member 1|4|4|0|
|tests/test\_vector\_db.py|Member 4|4|4|0|
|tests/test\_scrapers.py|Member 5|9|9|0|
|tests/test\_project\_structure.py|Member 5|19|19|0|
|**Total**||**67**|**67**|**0**|

**Pass rate: 100%.**

What the Member 5 tests cover:

* `test\_code\_executor.py`: prompt template, generated code is valid Python, safe handling of quotes, error output instead of crashes, timeout on a hanging script, reproducible results.
* `test\_scrapers.py`: the Week 1 dataset (10 papers, 5 + 5, required fields, ISO dates, unique IDs, arXiv PDF links) and the handover to the vector DB (load, search by title, query under 1 second).
* `test\_project\_structure.py`: every file in the agreed structure exists.

### Coverage (`python -m pytest --cov=src`, needs `pip install pytest-cov`)

|Module|Coverage|
|-|-|
|ingestion/schema.py, ingestion/state.py|100%|
|agents/code\_executor\_agent.py|90%|
|agents/hypothesis\_agent.py|86%|
|agents/literature\_agent.py|85%|
|rag/vector\_db.py|81%|
|ingestion/pubmed\_scraper.py|80%|
|orchestrator.py|71%|
|ingestion/arxiv\_scraper.py, ingestion/pipeline.py|69%|
|**Total**|**81%**|

Lowest coverage is in the command-line `main()` functions and the live-network paths, which the offline tests intentionally do not call.

## 4\. GitHub and branch activity

|Item|Status|
|-|-|
|PR template|`.github/pull\_request\_template.md`|
|Review standards|`docs/pr\_review\_standards.md`|
|Branch protection on `main`|**Pending: needs the repo admin (Team Lead)**. Steps are in `docs/pr\_review\_standards.md`|
|Merge conflicts between `sachin` and `main`|Resolved: kept the Lead's `conftest.py` and `test\_agents.py`, kept Member 5's agent|

Commits per member this week (fill in from GitHub -> Insights -> Contributors, or `git shortlog -sn --all`):

|Member|Branch|Commits|Pull Requests|
|-|-|-|-|
|1||||
|2||||
|3||||
|4||||
|5|`sachin`|||

## 5\. Pipeline check

`python -m src.orchestrator` runs all three nodes with the real agents:

1. Literature Reviewer returns a summary built from the local papers.
2. Hypothesis Generator returns the top-ranked hypothesis.
3. Code Executor builds an experiment script, runs it, and returns `Metric delta: ...`.

## 6\. Issues and integration gaps found

|#|Issue|Where|Suggested owner|
|-|-|-|-|
|1|`requirements.txt` is missing `biopython`. On a fresh install, `test\_ingestion.py` and `test\_state.py` fail to import (`from Bio import Entrez`).|requirements.txt|Member 1 / 3|
|2|`docs/ingestion.md` tells users to install `requirements-ingestion.txt`, which does not exist in the repo.|docs/ingestion.md|Member 3|
|3|The vector DB uses in-memory keyword overlap. ChromaDB and embeddings (all-MiniLM-L6-v2) are in the plan and README, but `vector\_db.py` only creates unused `\_chroma\_\*` placeholders.|src/rag/vector\_db.py|Member 4|
|4|The Literature Reviewer reads `data/sample\_papers.json` directly and never queries the vector DB, so retrieval (RAG) is not connected to the agents yet.|src/agents/literature\_agent.py|Member 2 / 4|
|5|The Hypothesis Generator ignores the literature summary: it computes `concepts` but never uses it, and returns the same fixed templates every time.|src/agents/hypothesis\_agent.py|Member 2|
|6|No Pydantic schemas on agent inputs and outputs (agents pass plain strings), although `pydantic` is in `requirements.txt`.|src/agents/|Member 2 / 1|
|7|`data/sample\_papers.json` and `data/sample\_scraped\_papers.json` are identical copies. If one is regenerated, the other will drift.|data/|Member 3 / 4|
|8|The Code Executor runs generated scripts in a local subprocess with a timeout, not in Docker, and the experiment is a simulation. Safe for the current fixed template, not safe for LLM-written code.|src/agents/code\_executor\_agent.py|Member 5|
|9|Branch protection on `main` is not enabled yet.|GitHub settings|Member 1|

## 7\. Plan for Week 2

1. Connect the Literature Reviewer to the vector DB (issue 4) and agree a `search` contract.
2. Replace the keyword scoring with ChromaDB and a local embedding model (issue 3).
3. Make the Hypothesis Generator use the literature summary, ideally through an LLM with structured JSON output (issues 5 and 6).
4. Code Executor: send the prompt template to an LLM, add a retry loop that feeds errors back (max 3 tries), and move execution into a Docker sandbox (issue 8).
5. Fix the requirements and dataset duplication issues (issues 1, 2, 7).
6. Turn on branch protection and merge `sachin` into `main` through a reviewed Pull Request.

