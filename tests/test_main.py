"""
Unit and Integration Tests for Master Entrypoint (main.py)
Member 1 - Team Lead (Gayatri Bhakta)
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from main import (
    build_arg_parser,
    check_docker_environment,
    synthesize_latex_paper,
    run_pipeline,
    main,
)


def test_build_arg_parser_defaults():
    """Verify default CLI argument configurations."""
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.topic == "Autonomous Agent Reasoning"
    assert args.sandbox == "local"
    assert args.synthesize_latex is False
    assert args.enable_critic is True
    assert args.output is None
    assert args.quiet is False


def test_build_arg_parser_custom_flags():
    """Verify customized CLI flags are accurately parsed."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--topic", "Sparse Transformers",
        "--sandbox", "docker",
        "--synthesize-latex",
        "--no-critic",
        "--output", "test_out.json",
        "--quiet",
    ])
    assert args.topic == "Sparse Transformers"
    assert args.sandbox == "docker"
    assert args.synthesize_latex is True
    assert args.enable_critic is False
    assert args.output == "test_out.json"
    assert args.quiet is True


def test_check_docker_environment_fallback():
    """Ensure Docker check handles daemon unavailability gracefully."""
    with patch("src.sandbox.docker_runner.is_docker_available", return_value=False):
        assert check_docker_environment() is False

    with patch("src.sandbox.docker_runner.is_docker_available", return_value=True):
        assert check_docker_environment() is True


def test_synthesize_latex_paper_fallback(tmp_path: Path):
    """Verify built-in LaTeX synthesizer creates valid TeX and PDF artifacts."""
    sample_state = {
        "research_topic": "Transformer Optimization",
        "selected_hypothesis": "Sparse attention scales linearly with sequence length.",
        "critic_verdict": "ACCEPTED",
        "execution_result": "Baseline: 100\nMethod: 125\nMetric delta: +25\np-value: 0.001",
        "statistical_metrics": {
            "p_value": 0.001,
            "effect_size_cohens_d": 1.25,
            "is_significant": True,
        },
    }

    res = synthesize_latex_paper(sample_state, tmp_path)
    assert res["status"] == "SUCCESS"
    assert Path(res["tex_path"]).exists()
    assert Path(res["pdf_path"]).exists()

    tex_content = Path(res["tex_path"]).read_text(encoding="utf-8")
    assert "Transformer Optimization" in tex_content
    assert "Sparse attention scales linearly" in tex_content
    assert "p = 0.001" in tex_content
    assert "ACCEPTED" in tex_content


def test_run_pipeline_mocked_execution(tmp_path: Path):
    """Verify end-to-end execution flow with mocked LangGraph invocation."""
    mock_app = MagicMock()
    mock_result = {
        "research_topic": "Quantum Optimization",
        "selected_hypothesis": "Simulated annealing converges faster on dense graphs.",
        "execution_status": "SUCCESS",
        "retry_count": 0,
        "critic_verdict": "ACCEPTED",
        "statistical_metrics": {"p_value": 0.02, "is_significant": True},
        "logs": ["Step 1 completed", "Step 2 completed"],
    }
    mock_app.invoke.return_value = mock_result

    out_json = tmp_path / "result.json"

    with patch("main.build_orchestrator", return_value=mock_app):
        res = run_pipeline(
            topic="Quantum Optimization",
            sandbox_mode="local",
            synthesize_latex=True,
            enable_critic=True,
            output_path=str(out_json),
            quiet=True,
        )

    assert res["research_topic"] == "Quantum Optimization"
    assert res["execution_status"] == "SUCCESS"
    assert "latex_synthesis" in res
    assert "elapsed_seconds" in res
    assert out_json.exists()

    saved_data = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved_data["research_topic"] == "Quantum Optimization"
    assert saved_data["critic_verdict"] == "ACCEPTED"


def test_main_cli_entrypoint(monkeypatch):
    """Verify main() function parses sys.argv and executes cleanly."""
    monkeypatch.setattr("sys.argv", ["main.py", "--topic", "Test CLI", "--quiet"])

    mock_app = MagicMock()
    mock_app.invoke.return_value = {
        "research_topic": "Test CLI",
        "execution_status": "SUCCESS",
        "retry_count": 0,
        "critic_verdict": "ACCEPTED",
        "logs": ["Test log"],
    }

    with patch("main.build_orchestrator", return_value=mock_app):
        exit_code = main()
        assert exit_code == 0
