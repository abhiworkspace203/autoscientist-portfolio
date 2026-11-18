# Evaluation & Benchmarking (Member 3, Week 4)

Grades the agent's generated research paper against recent human-written papers on 4 dimensions (0–10 each).
Full results: [evaluation_benchmark_report.md](evaluation_benchmark_report.md).

## Files

| File | What it does |
| --- | --- |
| `src/evaluation/holdout_curator.py` | Picks 9 recent arXiv papers (AI, Computational Biology, Physics; 2024–2026) and writes `data/holdout_papers/benchmark_metadata.json` |
| `src/evaluation/paper_structure.py` | Finds sections, figures, tables, references and abstract length in PDF text, LaTeX or Markdown |
| `src/evaluation/benchmark_evaluator.py` | Scores Novelty, Literature Grounding, Empirical Validity, Formatting (0–10) |
| `src/evaluation/benchmark_report.py` | Comparison tables, Agent vs Human radar chart, `docs/evaluation_benchmark_report.md` |

## Run it

```bash
python -m src.evaluation.holdout_curator
python -m src.evaluation.benchmark_evaluator --agent examples/sample_agent_output.json --save data/holdout_papers/benchmark_scores.json
python -m src.evaluation.benchmark_report
```

PDFs are saved in `data/holdout_papers/pdfs/` (git-ignored). Everything else runs offline.

## Scoring the real agent paper

Save the synthesizer agent's output as JSON with these keys (missing keys score 0), then pass it with `--agent`:

| Key | Type | Comes from |
| --- | --- | --- |
| `paper_id`, `title`, `domain` | string | synthesizer |
| `paper_text` | string (LaTeX or Markdown) | synthesizer (`paper_template.tex` filled in) |
| `hypothesis` | string | hypothesis agent |
| `abstract` | string | synthesizer |
| `cited_ids` | list of paper IDs | hypothesis agent citations |
| `retrieved_papers` | list of `{paper_id, title, abstract}` | retriever (`get_relevant_literature`) |
| `p_value`, `effect_size`, `ci` | number, number, `[low, high]` | stats validator / critic |

See `examples/sample_agent_output.json` for a complete example.

```python
from src.evaluation.benchmark_evaluator import evaluate_agent_output, run_benchmark
from src.evaluation.benchmark_report import generate

score = evaluate_agent_output(agent_output)     # PaperScore: .novelty .grounding .empirical_validity .formatting .overall
result = run_benchmark(agent_output)            # agent + every holdout paper + human average
generate(result)                                # tables, radar chart and report
```

## Tests

```bash
python -m pytest tests/test_holdout_curator.py tests/test_benchmark_evaluator.py tests/test_benchmark_report.py -q
```

34 tests, all offline (arXiv and PDF downloads are mocked, PDFs are generated on the fly).
