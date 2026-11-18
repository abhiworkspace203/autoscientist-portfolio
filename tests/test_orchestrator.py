"""
Tests for the Multi-Agent Orchestrator (Member 1 / Team Lead Deliverable).
"""

import pytest
from src.orchestrator import (
    AgentGraphState,
    build_orchestrator,
    literature_reviewer_node,
    hypothesis_generator_node,
    code_executor_node,
)


def test_literature_reviewer_node_returns_summary_and_logs():
    state: AgentGraphState = {
        "research_topic": "Dynamic token pruning in transformers",
        "literature_summary": "",
        "generated_hypotheses": [],
        "selected_hypothesis": "",
        "generated_code": "",
        "execution_result": "",
        "logs": [],
    }
    result = literature_reviewer_node(state)
    assert "literature_summary" in result
    assert len(result["literature_summary"]) > 0
    assert "logs" in result
    assert any("Literature Reviewer" in log for log in result["logs"])


def test_hypothesis_generator_node_produces_hypotheses():
    state: AgentGraphState = {
        "research_topic": "Autonomous reasoning",
        "literature_summary": "Bottlenecks in agent inference latency and memory footprints.",
        "generated_hypotheses": [],
        "selected_hypothesis": "",
        "generated_code": "",
        "execution_result": "",
        "logs": [],
    }
    result = hypothesis_generator_node(state)
    assert "generated_hypotheses" in result
    assert len(result["generated_hypotheses"]) >= 1
    assert "selected_hypothesis" in result
    assert result["selected_hypothesis"].startswith("Hypothesis:")
    assert any("Hypothesis Generator" in log for log in result["logs"])


def test_code_executor_node_executes_code():
    state: AgentGraphState = {
        "research_topic": "Autonomous reasoning",
        "literature_summary": "Summary",
        "generated_hypotheses": [],
        "selected_hypothesis": "Hypothesis: Dynamic pruning reduces memory by 30%.",
        "generated_code": "",
        "execution_result": "",
        "logs": [],
    }
    result = code_executor_node(state)
    assert "generated_code" in result
    assert "execution_result" in result
    assert "Metric delta:" in result["execution_result"]
    assert any("Code Executor" in log for log in result["logs"])


def test_full_orchestrator_pipeline_end_to_end():
    app = build_orchestrator()
    initial_state = {"research_topic": "Neural Network Memory Optimization"}
    result = app.invoke(initial_state)

    assert result["research_topic"] == "Neural Network Memory Optimization"
    assert len(result["literature_summary"]) > 0
    assert len(result["generated_hypotheses"]) > 0
    assert len(result["selected_hypothesis"]) > 0
    assert len(result["generated_code"]) > 0
    assert "Metric delta:" in result["execution_result"]
    assert len(result["logs"]) >= 3
    assert result.get("execution_status") == "SUCCESS"
    assert "retry_count" in result


def test_debugger_node_increments_retry_count_and_patches_code():
    from src.orchestrator import debugger_node
    buggy_state: AgentGraphState = {
        "research_topic": "Test topic",
        "literature_summary": "Test summary",
        "retrieved_papers": [],
        "generated_hypotheses": ["Hypothesis: test"],
        "selected_hypothesis": "Hypothesis: test",
        "generated_code": "x = 10 / 0",
        "execution_result": "ZeroDivisionError: division by zero",
        "execution_status": "ERROR",
        "error_traceback": "ZeroDivisionError: division by zero",
        "retry_count": 0,
        "logs": [],
    }
    result = debugger_node(buggy_state)
    assert result["retry_count"] == 1
    assert "10 / 1" in result["generated_code"]
    assert any("Debugger Agent" in log for log in result["logs"])


def test_execution_router_conditional_routing():
    from src.orchestrator import execution_router
    from langgraph.graph import END

    # Case 1: Success -> END
    success_state: AgentGraphState = {
        "execution_status": "SUCCESS",
        "retry_count": 0,
    }  # type: ignore
    assert execution_router(success_state) == END

    # Case 2: Error with retry < 3 -> debugger_node
    retry_state: AgentGraphState = {
        "execution_status": "ERROR",
        "retry_count": 1,
    }  # type: ignore
    assert execution_router(retry_state) == "debugger_node"

    # Case 3: Error with retry >= 3 -> END (abort)
    exhausted_state: AgentGraphState = {
        "execution_status": "ERROR",
        "retry_count": 3,
    }  # type: ignore
    assert execution_router(exhausted_state) == END



def test_execution_router_with_critic_active():
    from src.orchestrator import execution_router
    critic_active_state: AgentGraphState = {
        "execution_status": "SUCCESS",
        "retry_count": 0,
        "enable_critic": True,
    }
    assert execution_router(critic_active_state) == "critic_node"


def test_critic_node_evaluates_output():
    from src.orchestrator import critic_node
    state: AgentGraphState = {
        "research_topic": "Dynamic Pruning",
        "selected_hypothesis": "Hypothesis: pruning saves 30% memory",
        "generated_code": "import random\n# Pruning memory simulation\nmemory_saved = 0.31",
        "execution_result": "Measured memory with and without pruning: memory dropped 31%. Baseline mean: 98. Method mean: 102. Metric delta: +4.0",
        "statistical_metrics": {
            "p_value": 0.003,
            "is_significant": True,
            "confidence_interval_95": [2.1, 6.4],
            "effect_size_cohens_d": 0.82,
            "n_baseline": 40,
            "n_treatment": 40,
            "summary": "Statistically significant improvement over baseline with large effect size.",
        },
        "experiment_iteration": 0,
        "logs": [],
    }
    result = critic_node(state)
    assert result["critic_verdict"] == "ACCEPTED"
    assert result["experiment_iteration"] == 1
    assert any("Critic Agent" in log for log in result["logs"])


def test_critic_router_conditional_routing():
    from src.orchestrator import critic_router
    from langgraph.graph import END

    # Case 1: Accepted -> END
    accepted_state: AgentGraphState = {
        "critic_verdict": "ACCEPTED",
        "experiment_iteration": 1,
    }
    assert critic_router(accepted_state) == END

    # Case 2: Rejected with iteration < 2 -> code_executor
    rejected_state: AgentGraphState = {
        "critic_verdict": "REJECTED",
        "experiment_iteration": 1,
    }
    assert critic_router(rejected_state) == "code_executor"

    # Case 3: Rejected with iteration >= 2 -> END
    exhausted_critique_state: AgentGraphState = {
        "critic_verdict": "REJECTED",
        "experiment_iteration": 2,
    }
    assert critic_router(exhausted_critique_state) == END


def test_full_orchestrator_pipeline_with_critic_enabled():
    app = build_orchestrator(enable_critic=True)
    initial_state = {
        "research_topic": "Quantum-Inspired Optimization",
        "enable_critic": True,
    }
    result = app.invoke(initial_state)

    assert result["research_topic"] == "Quantum-Inspired Optimization"
    assert result.get("critic_verdict") in ["ACCEPTED", "REJECTED", "NEEDS_REPLICATION"]
    assert "critic_critique" in result
    assert result.get("experiment_iteration", 0) >= 1
