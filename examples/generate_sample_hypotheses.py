"""Generate data/sample_grounded_hypotheses.json from REAL retrieved paper chunks.

Run from the project root, with your LLM settings in the environment / .env
(LLM_PROVIDER, LLM_API_KEY, LLM_MODEL)::

    python examples/generate_sample_hypotheses.py --chunks path/to/chunks.json

``chunks.json`` (or .jsonl) is a list of chunks exported from your RAG /
ingestion step. Each needs a paper id (paper_id / arxiv_id / pmid / id) and
text (text / content / abstract), and may include title, findings, methodology.

The script refuses to write anything unless a real LLM call succeeded, so the
output file can never be mock data.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.agents.hypothesis_agent import (  # noqa: E402
    HypothesisGenerator,
    llm_client_from_env,
    load_chunks,
    normalize_chunks,
)


def build_output(report, llm, chunks, source):
    evidence, _ = normalize_chunks(chunks)
    seen, papers = set(), []
    for chunk in evidence:
        if chunk.paper_id not in seen:
            seen.add(chunk.paper_id)
            papers.append({"paper_id": chunk.paper_id, "title": chunk.title})
    data = report.model_dump(mode="json")
    return {
        "generated_by": {
            "type": "real_llm_call",
            "provider": getattr(llm, "provider", "unknown"),
            "model": getattr(llm, "model", "unknown"),
            "generated_at": report.generated_at,
        },
        "evidence_source": source,
        "evidence_papers": papers,
        **data,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", required=True, help="JSON/JSONL file of retrieved chunks")
    parser.add_argument("--out", default="data/sample_grounded_hypotheses.json")
    parser.add_argument("--topic", default=None)
    parser.add_argument("--count", type=int, default=3, help="hypotheses wanted (default 3)")
    parser.add_argument("--allow-fewer", action="store_true", help="write even if fewer than --count validate")
    args = parser.parse_args(argv)

    llm = llm_client_from_env()
    if llm is None:
        print("ERROR: set LLM_API_KEY and LLM_MODEL (and LLM_PROVIDER) first. Nothing written.")
        return 1
    chunks = load_chunks(args.chunks)
    report = HypothesisGenerator(llm=llm, max_hypotheses=args.count).generate_grounded(chunks, topic=args.topic)

    print(f"status: {report.status}; valid hypotheses: {len(report.hypotheses)}; rejected: {len(report.rejected)}")
    for message in report.errors + [e for r in report.rejected for e in r.errors]:
        print("  -", message)
    if report.status not in ("ok", "partial") or (len(report.hypotheses) < args.count and not args.allow_fewer):
        print("Not writing the output file (need real, validated hypotheses). Fix the issues above and retry.")
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(build_output(report, llm, chunks, args.chunks), handle, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
