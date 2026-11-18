"""Unit tests for RAG-grounded hypothesis generation.

All LLM calls are mocked, and every paper below is a TEST FIXTURE, not a real
paper. Passing these tests does NOT prove a real LLM integration works.
"""
import json

import pytest
from pydantic import ValidationError

from src.agents.hypothesis_agent import (
    SYSTEM_PROMPT,
    SAMPLE_REVIEW,
    GroundedHypothesis,
    GroundedReport,
    HypothesisGenerator,
    LLMError,
    generate_hypothesis,
    load_chunks,
    normalize_chunks,
)

# ---- fixtures -----------------------------------------------------------------
CHUNKS = [
    {"paper_id": "TEST-A", "title": "Fixture paper A",
     "text": "Noisy labels reduce the accuracy of graph models by about ten percent on benchmark data."},
    {"paper_id": "TEST-B", "title": "Fixture paper B",
     "text": "Contrastive pretraining improves robustness of encoders when training data is limited.",
     "findings": ["Pretraining raised macro-F1 under label noise"], "methodology": "Ablation on three benchmarks"},
    {"paper_id": "TEST-C", "title": "Fixture paper C",
     "text": "Ensembling small models lowers variance across random seeds in low-resource settings."},
]

GOOD = {
    "title": "Pretraining against label noise",
    "problem_statement": "Graph models lose accuracy under noisy labels.",
    "proposed_solution": "Add contrastive pretraining before supervised training.",
    "cited_paper_ids": ["TEST-A", "TEST-B"],
    "supporting_findings": [
        {"paper_id": "TEST-A", "finding": "Noisy labels reduce the accuracy of graph models"},
        {"paper_id": "TEST-B", "finding": "Contrastive pretraining improves robustness of encoders"},
    ],
    "falsifiable_metric": "Macro-F1 higher than the baseline by at least 2 points over 10 seeds at 20% label noise",
    "expected_code_modules": ["data_loader", "model_training", "evaluation"],
}


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def gen(reply=None, error=None, **kw):
    return HypothesisGenerator(llm=FakeLLM(reply, error), **kw)


def with_(**changes):
    item = json.loads(json.dumps(GOOD))
    item.update(changes)
    return item


# ---- 1. valid output ----------------------------------------------------------
def test_valid_grounded_hypothesis():
    llm = FakeLLM({"hypotheses": [GOOD]})
    report = HypothesisGenerator(llm=llm).generate_grounded(CHUNKS, topic="label noise")
    assert report.status == "ok" and report.rejected == [] and report.errors == []
    h = report.hypotheses[0]
    assert h.cited_paper_ids == ["TEST-A", "TEST-B"] and len(h.supporting_findings) == 2
    assert h.origin == "ai_generated" and h.novelty_status == "not_verified"
    assert report.valid_paper_ids == ["TEST-A", "TEST-B", "TEST-C"]
    assert llm.prompts[0].startswith(SYSTEM_PROMPT)
    assert "[PAPER_ID: TEST-A]" in llm.prompts[0]


def test_llm_cannot_override_origin_or_novelty():
    report = gen({"hypotheses": [with_(origin="human", novelty_status="verified_novel")]}).generate_grounded(CHUNKS)
    assert report.hypotheses[0].origin == "ai_generated"
    assert report.hypotheses[0].novelty_status == "not_verified"


# ---- 2. missing fields ---------------------------------------------------------
@pytest.mark.parametrize("field", list(GroundedHypothesis.model_fields))
def test_schema_requires_every_field(field):
    data = {k: v for k, v in GOOD.items() if k in GroundedHypothesis.model_fields}
    del data[field]
    with pytest.raises(ValidationError):
        GroundedHypothesis(**data)


def test_missing_field_in_llm_output_is_rejected_not_repaired():
    bad = {k: v for k, v in GOOD.items() if k != "proposed_solution"}
    report = gen({"hypotheses": [bad]}).generate_grounded(CHUNKS)
    assert report.status == "failed" and report.hypotheses == []
    assert any("proposed_solution" in e for e in report.rejected[0].errors)


# ---- 3. empty strings -----------------------------------------------------------
@pytest.mark.parametrize("field", ["title", "problem_statement", "proposed_solution", "falsifiable_metric"])
@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_required_strings_rejected(field, empty):
    report = gen({"hypotheses": [with_(**{field: empty})]}).generate_grounded(CHUNKS)
    assert report.status == "failed" and report.rejected[0].errors


def test_blank_items_in_lists_rejected():
    report = gen({"hypotheses": [with_(expected_code_modules=["ok", " "])]}).generate_grounded(CHUNKS)
    assert report.status == "failed"


def test_non_testable_metric_rejected():
    report = gen({"hypotheses": [with_(falsifiable_metric="It will work well in practice")]}).generate_grounded(CHUNKS)
    assert report.status == "failed" and "measurable" in report.rejected[0].errors[0]


# ---- 4. invalid paper ids ---------------------------------------------------------
def test_unknown_paper_id_is_flagged_and_not_removed_silently():
    item = with_(cited_paper_ids=["TEST-A", "FAKE-999"],
                 supporting_findings=GOOD["supporting_findings"][:1]
                 + [{"paper_id": "FAKE-999", "finding": "Invented result about everything"}])
    report = gen({"hypotheses": [item]}).generate_grounded(CHUNKS)
    assert report.status == "failed" and report.hypotheses == []
    assert any("FAKE-999" in e for e in report.rejected[0].errors)


def test_partial_status_when_only_some_hypotheses_are_valid():
    bad = with_(cited_paper_ids=["NOPE"], supporting_findings=[{"paper_id": "NOPE", "finding": "made up claim here"}])
    report = gen({"hypotheses": [GOOD, bad]}).generate_grounded(CHUNKS)
    assert report.status == "partial" and len(report.hypotheses) == 1 and len(report.rejected) == 1


def test_finding_must_be_quoted_from_cited_paper():
    item = with_(supporting_findings=[
        {"paper_id": "TEST-A", "finding": "Noisy labels reduce the accuracy of graph models"},
        {"paper_id": "TEST-B", "finding": "This sentence does not appear anywhere in the paper"}])
    report = gen({"hypotheses": [item]}).generate_grounded(CHUNKS)
    assert report.status == "failed" and "not a quote" in report.rejected[0].errors[0]


def test_quote_check_can_be_relaxed():
    item = with_(supporting_findings=[
        {"paper_id": "TEST-A", "finding": "a paraphrase of the noisy label result"},
        {"paper_id": "TEST-B", "finding": "a paraphrase of the pretraining result"}])
    assert gen({"hypotheses": [item]}, require_verbatim_quotes=False).generate_grounded(CHUNKS).status == "ok"


def test_finding_attributed_to_uncited_paper_rejected():
    item = with_(supporting_findings=GOOD["supporting_findings"]
                 + [{"paper_id": "TEST-C", "finding": "Ensembling small models lowers variance"}])
    assert gen({"hypotheses": [item]}).generate_grounded(CHUNKS).status == "failed"


def test_cited_paper_without_supporting_finding_rejected():
    item = with_(supporting_findings=GOOD["supporting_findings"][:1])
    report = gen({"hypotheses": [item]}).generate_grounded(CHUNKS)
    assert report.status == "failed" and "no supporting finding" in report.rejected[0].errors[0]


# ---- 5. empty citations ------------------------------------------------------------
def test_empty_citation_list_rejected():
    report = gen({"hypotheses": [with_(cited_paper_ids=[])]}).generate_grounded(CHUNKS)
    assert report.status == "failed" and any("cited_paper_ids" in e for e in report.rejected[0].errors)
    with pytest.raises(ValidationError):
        GroundedHypothesis(**{**{k: v for k, v in GOOD.items() if k in GroundedHypothesis.model_fields},
                              "cited_paper_ids": []})


# ---- 6. missing evidence -----------------------------------------------------------
@pytest.mark.parametrize("chunks", [[], None, [{"title": "no id", "text": "x"}], [{"paper_id": "P", "text": ""}]])
def test_missing_evidence_returns_safe_result_without_calling_llm(chunks):
    llm = FakeLLM({"hypotheses": [GOOD]})
    report = HypothesisGenerator(llm=llm).generate_grounded(chunks)
    assert report.status == "insufficient_evidence" and "Insufficient evidence" in report.errors[0]
    assert llm.prompts == [] and report.hypotheses == []


def test_single_paper_is_insufficient_by_default():
    assert gen({"hypotheses": [GOOD]}).generate_grounded(CHUNKS[:1]).status == "insufficient_evidence"
    assert gen({"hypotheses": [GOOD]}, min_papers=1).generate_grounded(CHUNKS[:1]).status != "insufficient_evidence"


def test_llm_can_report_insufficient_evidence():
    report = gen({"insufficient_evidence": True, "reason": "papers are unrelated"}).generate_grounded(CHUNKS)
    assert report.status == "insufficient_evidence" and "unrelated" in report.errors[0]


def test_no_llm_configured_never_uses_templates(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    report = HypothesisGenerator().generate_grounded(CHUNKS)
    assert report.status == "failed" and "No LLM configured" in report.errors[0] and report.hypotheses == []


# ---- 7. multiple chunks ---------------------------------------------------------------
def test_multiple_chunks_and_aliases_are_normalised():
    chunks = CHUNKS + [
        {"paper_id": "TEST-A", "text": "A second chunk of paper A mentions dropout regularisation results."},
        {"arxiv_id": "2401.00001", "abstract": "Abstract text for an arXiv style record."},
        "not a dict",
    ]
    evidence, skipped = normalize_chunks(chunks)
    assert [c.paper_id for c in evidence].count("TEST-A") == 2 and "2401.00001" in {c.paper_id for c in evidence}
    assert len(skipped) == 1
    item = with_(cited_paper_ids=["TEST-A", "TEST-B", "TEST-C"],
                 supporting_findings=GOOD["supporting_findings"]
                 + [{"paper_id": "TEST-C", "finding": "Ensembling small models lowers variance across random seeds"}])
    report = gen({"hypotheses": [item]}).generate_grounded(chunks)
    assert report.status == "ok" and report.valid_paper_ids.count("TEST-A") == 1
    assert report.notes  # skipped chunk is reported, not hidden


def test_finding_can_come_from_second_chunk_of_same_paper():
    chunks = CHUNKS + [{"paper_id": "TEST-A", "text": "Dropout regularisation recovers part of the lost accuracy."}]
    item = with_(supporting_findings=[
        {"paper_id": "TEST-A", "finding": "Dropout regularisation recovers part of the lost accuracy"},
        GOOD["supporting_findings"][1]])
    assert gen({"hypotheses": [item]}).generate_grounded(chunks).status == "ok"


def test_max_hypotheses_respected():
    assert len(gen({"hypotheses": [GOOD] * 5}, max_hypotheses=2).generate_grounded(CHUNKS).hypotheses) == 2


# ---- 8. LLM errors / malformed responses ---------------------------------------------------
@pytest.mark.parametrize("error", [LLMError("LLM request failed: timeout"), RuntimeError("boom"), ConnectionError("down")])
def test_llm_errors_give_safe_failed_result(error):
    report = gen(error=error).generate_grounded(CHUNKS)
    assert report.status == "failed" and report.hypotheses == [] and report.errors


@pytest.mark.parametrize("reply", ["not json at all", "[1, 2, 3]", '{"hypotheses": "oops"}', '{"hypotheses": []}',
                                   '{"other": 1}', "{broken json}", ""])
def test_malformed_llm_responses(reply):
    report = gen(reply).generate_grounded(CHUNKS)
    assert report.status == "failed" and report.hypotheses == [] and report.errors


def test_non_object_hypothesis_entries_rejected():
    report = gen({"hypotheses": ["just a string"]}).generate_grounded(CHUNKS)
    assert report.status == "failed" and report.rejected[0].errors


def test_json_wrapped_in_prose_is_accepted():
    reply = "Sure! Here you go:\n" + json.dumps({"hypotheses": [GOOD]}) + "\nHope that helps."
    assert gen(reply).generate_grounded(CHUNKS).status == "ok"


# ---- 9. serialisation ---------------------------------------------------------------------------
def test_report_round_trips_through_json(tmp_path):
    report = gen({"hypotheses": [GOOD]}).generate_grounded(CHUNKS, topic="label noise")
    path = tmp_path / "out.json"
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    again = GroundedReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert again == report
    data = json.loads(report.model_dump_json())
    assert {"title", "problem_statement", "proposed_solution", "cited_paper_ids", "falsifiable_metric",
            "expected_code_modules", "supporting_findings"} <= set(data["hypotheses"][0])
    assert "AI-generated" in data["disclaimer"]


def test_load_chunks_json_and_jsonl(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps(CHUNKS), encoding="utf-8")
    (tmp_path / "w.json").write_text(json.dumps({"chunks": CHUNKS}), encoding="utf-8")
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(c) for c in CHUNKS) + "\n", encoding="utf-8")
    assert len(load_chunks(str(tmp_path / "c.json"))) == len(load_chunks(str(tmp_path / "w.json"))) == 3
    assert len(load_chunks(str(tmp_path / "c.jsonl"))) == 3


# ---- backward compatibility (existing behaviour must not change) -----------------------------------
def test_existing_offline_behaviour_unchanged(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    report = HypothesisGenerator().generate(SAMPLE_REVIEW)
    assert report.status == "ok" and report.mode == "fallback" and report.ai_generated_hypotheses
    assert generate_hypothesis("").startswith("Insufficient evidence")
    assert generate_hypothesis("Models degrade on noisy data").startswith("Hypothesis:")
