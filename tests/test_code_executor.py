"""Detailed tests for the Code Executor Agent (Member 5)."""
from src.agents import code_executor_agent as cea

HYP = "Dynamic token-level pruning reduces inference memory by 30%."


def test_prompt_contains_hypothesis():
    assert HYP in cea.build_prompt(HYP)


def test_prompt_has_no_unfilled_placeholder():
    assert "{hypothesis}" not in cea.build_prompt(HYP)


def test_prompt_strips_extra_whitespace():
    assert "Hypothesis:\nabc\n" in cea.build_prompt("   abc   ")


def test_generated_code_is_valid_python():
    compile(cea.generate_code(HYP), "<generated>", "exec")


def test_hypothesis_with_quotes_is_handled_safely():
    tricky = 'He said "hi" and it\'s fine'
    code = cea.generate_code(tricky)
    compile(code, "<generated>", "exec")
    assert tricky in cea.run_code(code)


def test_run_code_returns_stdout():
    assert cea.run_code("print('hello')") == "hello"


def test_run_code_reports_errors_instead_of_crashing():
    output = cea.run_code("raise ValueError('boom')")
    assert "Execution failed" in output
    assert "ValueError" in output


def test_run_code_stops_a_hanging_script():
    output = cea.run_code("while True: pass", timeout=1)
    assert "timed out" in output


def test_generate_and_run_code_returns_two_strings():
    code, output = cea.generate_and_run_code(HYP)
    assert isinstance(code, str) and code
    assert isinstance(output, str) and output


def test_output_reports_a_metric_delta():
    _, output = cea.generate_and_run_code(HYP)
    assert "Metric delta:" in output


def test_results_are_reproducible():
    assert cea.generate_and_run_code(HYP) == cea.generate_and_run_code(HYP)
