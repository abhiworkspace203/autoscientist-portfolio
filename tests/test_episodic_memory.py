"""
Unit tests for Episodic Memory Engine (Member 1 / Team Lead Deliverable).
Verifies episode creation, disk persistence, metric extraction, filtering, and summarization.
"""

import pytest
from pathlib import Path
from src.memory.episodic_memory import Episode, EpisodicMemoryStore


@pytest.fixture
def temp_store(tmp_path):
    storage_file = tmp_path / "test_episodes.json"
    return EpisodicMemoryStore(storage_path=storage_file)


def test_episode_model_defaults():
    ep = Episode(topic="Drug Discovery", hypothesis="Compound X binds Target Y")
    assert ep.episode_id.startswith("ep_")
    assert ep.verdict == "PENDING"
    assert ep.iteration == 1
    assert ep.retry_count == 0


def test_add_and_retrieve_episodes(temp_store):
    ep1 = Episode(
        topic="Drug Discovery",
        hypothesis="Hypothesis A: target binds with high affinity",
        verdict="ACCEPTED",
        metric_delta=3.5,
        p_value=0.012,
        lessons_learned=["Affinity improves with ligand optimization"],
    )
    ep2 = Episode(
        topic="Quantum Neural Networks",
        hypothesis="Hypothesis B: tensor networks compress parameter space",
        verdict="REJECTED",
        lessons_learned=["Gradient vanishing observed at depth 6"],
    )

    temp_store.add_episode(ep1)
    temp_store.add_episode(ep2)

    # Retrieve all
    all_eps = temp_store.get_episodes()
    assert len(all_eps) == 2

    # Filter by topic
    drug_eps = temp_store.get_episodes(topic="Drug Discovery")
    assert len(drug_eps) == 1
    assert drug_eps[0].topic == "Drug Discovery"
    assert drug_eps[0].metric_delta == 3.5

    # Filter by verdict
    accepted_eps = temp_store.get_episodes(verdict="ACCEPTED")
    assert len(accepted_eps) == 1
    assert accepted_eps[0].verdict == "ACCEPTED"


def test_record_from_state_parses_deltas_and_p_values(temp_store):
    sample_state = {
        "research_topic": "Dynamic Pruning",
        "selected_hypothesis": "Hypothesis: pruning reduces memory footprint",
        "execution_status": "SUCCESS",
        "critic_verdict": "ACCEPTED",
        "execution_result": "Baseline mean: 100. Method mean: 104.5. Metric delta: +4.5.",
        "critic_critique": "Statistically significant improvement (p-value: 0.008, 95% CI [2.1, 6.9]).",
        "retry_count": 1,
        "experiment_iteration": 2,
    }

    recorded = temp_store.record_from_state(sample_state)
    assert recorded.topic == "Dynamic Pruning"
    assert recorded.metric_delta == 4.5
    assert recorded.p_value == 0.008
    assert recorded.verdict == "ACCEPTED"
    assert recorded.iteration == 2
    assert recorded.retry_count == 1
    assert any("Hypothesis validated" in l for l in recorded.lessons_learned)


def test_get_lessons_learned_deduplication(temp_store):
    ep1 = Episode(
        topic="Transformers",
        hypothesis="H1",
        lessons_learned=["Use AdamW with warmup", "Batch size 32 is optimal"],
    )
    ep2 = Episode(
        topic="Transformers",
        hypothesis="H2",
        lessons_learned=["Use AdamW with warmup", "Learning rate 1e-4 prevents divergence"],
    )

    temp_store.add_episode(ep1)
    temp_store.add_episode(ep2)

    lessons = temp_store.get_lessons_learned(topic="Transformers", max_items=5)
    assert len(lessons) == 3
    assert lessons.count("Use AdamW with warmup") == 1


def test_format_episodic_summary_within_budget(temp_store):
    for i in range(10):
        temp_store.add_episode(
            Episode(
                topic="Token Pruning",
                hypothesis=f"Hypothesis {i}: Pruning strategy variant {i} reduces cache memory",
                verdict="ACCEPTED" if i % 2 == 0 else "REJECTED",
                metric_delta=float(i) * 0.5,
                lessons_learned=[f"Lesson from iteration {i}"],
            )
        )

    summary = temp_store.format_episodic_summary("Token Pruning", max_tokens=150)
    assert "Prior Experimental History" in summary
    assert "Total past runs:" in summary
    assert "Key Lessons:" in summary
    assert len(summary) > 0


def test_clear_storage(temp_store):
    temp_store.add_episode(Episode(topic="Topic", hypothesis="Hyp"))
    assert len(temp_store.get_episodes()) == 1
    temp_store.clear()
    assert len(temp_store.get_episodes()) == 0
