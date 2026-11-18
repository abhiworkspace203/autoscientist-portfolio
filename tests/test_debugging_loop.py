"""
Tests for the Debugger Agent and the Reflexion self-debugging loop (Member 5).

Three intentionally buggy scripts (ZeroDivisionError, IndexError, ImportError) must
fail on attempt 1, be fixed by the debugger, and run clean on attempt 2.
All tests are offline: no LLM, no network, no Docker.
"""
import pytest

from src.agents.code_executor_agent import run_code_detailed
from src.agents.debugger_agent import (
    build_reflexion_prompt,
    debug_and_run,
    debug_code,
    extract_code,
    parse_traceback,
)

HYP = "Hypothesis: pruning reduces memory."

BUGGY_SCRIPTS = {
    "ZeroDivisionError": (
        "values = [4, 8, 15]\n"
        "count = len([])\n"
        "average = sum(values) / count\n"
        'print("Average:", average)\n'
    ),
    "IndexError": (
        "data = [1, 2, 3]\n"
        'print("Fourth item:", data[3])\n'
    ),
    "ImportError": (
        "from collections import Mapping\n"
        'print("Is mapping:", isinstance({}, Mapping))\n'
    ),
}


class FakeLLM:
    """Stands in for a real LLM: returns a fixed reply and records the prompts it got."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# ---------------------------------------------------------------------------
# The three required buggy scripts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("error_type", BUGGY_SCRIPTS)
def test_script_fails_on_attempt_one(error_type):
    result = run_code_detailed(BUGGY_SCRIPTS[error_type])
    assert result["exit_code"] != 0
    assert error_type in result["stderr"]


@pytest.mark.parametrize("error_type", BUGGY_SCRIPTS)
def test_debugger_fixes_script_and_attempt_two_runs_clean(error_type):
    original = BUGGY_SCRIPTS[error_type]
    outcome = debug_and_run(original, HYP)

    assert outcome["execution_status"] == "SUCCESS"
    assert outcome["retry_count"] == 1
    assert [h["exit_code"] for h in outcome["history"]] == [1, 0]  # fails, then clean
    assert outcome["code"] != original
    assert outcome["output"]


# ---------------------------------------------------------------------------
# Prompt and parsing
# ---------------------------------------------------------------------------
def test_prompt_contains_code_traceback_and_hypothesis():
    prompt = build_reflexion_prompt("print(1/0)", "ZeroDivisionError: division by zero", HYP)
    assert "print(1/0)" in prompt
    assert "ZeroDivisionError: division by zero" in prompt
    assert HYP in prompt
    assert "line specified by the traceback" in prompt
    assert "{" not in prompt.replace("{}", "")  # no unfilled placeholders


def test_extract_code_from_fenced_block():
    assert extract_code("Here you go:\n```python\nprint('ok')\n```\nDone.") == "print('ok')"


def test_extract_code_from_plain_text():
    assert extract_code("print('ok')") == "print('ok')"


def test_parse_traceback_reads_type_and_line():
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/x.py", line 3, in <module>\n'
        "    average = sum(values) / count\n"
        "ZeroDivisionError: division by zero\n"
    )
    info = parse_traceback(tb)
    assert info["error_type"] == "ZeroDivisionError"
    assert info["line_no"] == 3
    assert info["message"] == "division by zero"


# ---------------------------------------------------------------------------
# LLM path and fallbacks
# ---------------------------------------------------------------------------
def test_llm_reply_is_used_when_available():
    llm = FakeLLM("```python\nprint('fixed by llm')\n```")
    fixed = debug_code("print(1/0)", "ZeroDivisionError: division by zero", HYP, llm=llm)
    assert fixed == "print('fixed by llm')"
    assert "print(1/0)" in llm.prompts[0]


def test_llm_failure_falls_back_to_rules():
    llm = FakeLLM(RuntimeError("API down"))
    outcome = debug_and_run(BUGGY_SCRIPTS["ZeroDivisionError"], HYP, llm=llm)
    assert outcome["execution_status"] == "SUCCESS"


def test_unfixable_error_returns_original_code():
    code = "raise RuntimeError('no rule for this')"
    tb = 'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\nRuntimeError: no rule for this'
    assert debug_code(code, tb, HYP) == code


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------
def test_working_code_needs_no_debugging():
    outcome = debug_and_run("print('fine')", HYP)
    assert outcome["execution_status"] == "SUCCESS"
    assert outcome["retry_count"] == 0
    assert len(outcome["history"]) == 1


def test_loop_stops_when_no_fix_is_found():
    outcome = debug_and_run("raise RuntimeError('x')", HYP)
    assert outcome["execution_status"] == "ERROR"
    assert len(outcome["history"]) == 1


def test_loop_stops_after_max_retries():
    # An LLM that keeps returning different broken code never succeeds.
    class AlwaysBroken:
        def __init__(self):
            self.n = 0

        def complete(self, prompt):
            self.n += 1
            return f"raise RuntimeError('still broken {self.n}')"

    outcome = debug_and_run("raise RuntimeError('start')", HYP, max_retries=3, llm=AlwaysBroken())
    assert outcome["execution_status"] == "ERROR"
    assert outcome["retry_count"] == 3
    assert len(outcome["history"]) == 4  # first run + 3 retries


def test_loop_accepts_a_sandbox_style_runner():
    """The Docker runner returns the same dict shape, so it can be passed in as `runner`."""
    calls = []

    def fake_sandbox(code, timeout):
        calls.append(code)
        if len(calls) == 1:
            return {"exit_code": 1, "stdout": "", "stderr": "Traceback\n"
                    '  File "x.py", line 1, in <module>\nZeroDivisionError: division by zero',
                    "timed_out": False}
        return {"exit_code": 0, "stdout": "ok\n", "stderr": "", "timed_out": False}

    outcome = debug_and_run("print(1 / 0)", HYP, runner=fake_sandbox)
    assert outcome["execution_status"] == "SUCCESS"
    assert len(calls) == 2
    assert "or 1" in calls[1]


def test_timeout_is_reported_to_the_llm():
    llm = FakeLLM("print('fast version')")

    def runner(code, timeout):
        if "fast version" in code:
            return {"exit_code": 0, "stdout": "done", "stderr": "", "timed_out": False}
        return {"exit_code": -1, "stdout": "", "stderr": "", "timed_out": True}

    outcome = debug_and_run("while True: pass", HYP, llm=llm, runner=runner)
    assert outcome["execution_status"] == "SUCCESS"
    assert "timed out" in llm.prompts[0]
