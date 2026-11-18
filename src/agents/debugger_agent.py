"""
Debugger Agent (Reflexion self-debugging loop)
Member 5 Deliverable (Week 2)

When generated code fails, the debugger takes the original code, the exact error
traceback and the hypothesis, and returns corrected code. A small loop then runs the
fixed code again (Reflexion: run -> read the error -> rewrite -> run).

Two ways to produce the fix:
  1. LLM: pass any object with ``complete(prompt) -> str`` (the same interface as
     ``LLMClient`` in hypothesis_agent.py, e.g. ``llm_client_from_env()``).
  2. Offline fallback (no LLM, no network): targeted rules for common bugs
     (ZeroDivisionError, IndexError, outdated imports). Heuristic, but deterministic.

Contract for the orchestrator (debugger_node):
    fixed_code = debug_code(original_code, error_traceback, hypothesis, llm=None)
    result     = debug_and_run(code, hypothesis, max_retries=3, llm=None, runner=None)
        result["execution_status"] -> "SUCCESS" or "ERROR"
        result["retry_count"]      -> number of fix attempts used
        result["code"], result["output"], result["history"]

The runner is any function ``runner(code, timeout) -> {"exit_code", "stdout",
"stderr", "timed_out"}``. The default is the local runner in code_executor_agent.py;
the Docker sandbox runner returns the same shape, so it can be passed in directly.
"""

import re
from typing import Any, Callable, Dict, List, Optional

REFLEXION_PROMPT = """You are the Debugger Agent in an autonomous scientific discovery system.

Hypothesis under test:
{hypothesis}

Here is your original code:
```python
{original_code}
```

Running it failed with this exact error traceback:
```
{error_traceback}
```

Diagnose the issue. Identify the bug on the line specified by the traceback.
Return only the corrected executable Python script inside one ```python code block.
Do not add explanations.
"""

DEFAULT_MAX_RETRIES = 3

# Names that moved from `collections` to `collections.abc` (removed in Python 3.10).
_COLLECTIONS_ABC = {
    "Mapping", "MutableMapping", "Iterable", "Iterator", "Sequence", "MutableSequence",
    "Callable", "Set", "MutableSet", "Hashable", "Sized", "Container", "Generator",
}

# Modules that were renamed or moved in common scientific libraries.
_MOVED_MODULES = {
    "sklearn.cross_validation": "sklearn.model_selection",
    "sklearn.grid_search": "sklearn.model_selection",
    "sklearn.externals.joblib": "joblib",
    "sklearn.preprocessing.imputation": "sklearn.impute",
}


# ---------------------------------------------------------------------------
# Prompt and reply handling
# ---------------------------------------------------------------------------
def build_reflexion_prompt(original_code: str, error_traceback: str, hypothesis: str = "") -> str:
    """Fill the Reflexion prompt with the code, the traceback and the hypothesis."""
    return REFLEXION_PROMPT.format(
        hypothesis=(hypothesis or "(not provided)").strip(),
        original_code=original_code.rstrip(),
        error_traceback=error_traceback.strip(),
    )


def extract_code(reply: str) -> str:
    """Return the Python code from an LLM reply (a fenced block, or the plain text)."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply, re.DOTALL)
    return (match.group(1) if match else reply).strip("\n")


def parse_traceback(error_traceback: str) -> Dict[str, Any]:
    """Pull the error type, message and failing line number out of a traceback."""
    lines = [ln for ln in error_traceback.strip().splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    error_type, _, message = last.partition(":")
    line_numbers = re.findall(r'File "[^"]+", line (\d+)', error_traceback)
    return {
        "error_type": error_type.strip(),
        "message": message.strip(),
        "line_no": int(line_numbers[-1]) if line_numbers else None,
    }


# ---------------------------------------------------------------------------
# Offline fallback: targeted fixes
# ---------------------------------------------------------------------------
_DIVISOR = r"\d+(?:\.\d+)?|[A-Za-z_][\w\.]*(?:\([^()]*\))?(?:\[[^\]]*\])?"
_DIVISION = re.compile(rf"(?P<op>//|/|%)\s*(?P<den>{_DIVISOR})")
_INDEX = re.compile(r"(?P<seq>[A-Za-z_][\w\.]*)\[(?P<idx>\d+|[A-Za-z_]\w*)\]")


def _fix_zero_division(line: str) -> Optional[str]:
    """Guard every divisor on the failing line: ``a / b`` -> ``a / (b or 1)``."""
    fixed = _DIVISION.sub(lambda m: f"{m.group('op')} ({m.group('den')} or 1)", line)
    return fixed if fixed != line else None


def _fix_index_error(line: str) -> Optional[str]:
    """Keep indexes in range: ``items[5]`` -> ``items[min(5, len(items) - 1)]``."""
    fixed = _INDEX.sub(
        lambda m: f"{m.group('seq')}[min({m.group('idx')}, len({m.group('seq')}) - 1)]", line
    )
    return fixed if fixed != line else None


def _fix_import_line(line: str, message: str) -> Optional[str]:
    """Repair outdated imports (collections ABCs, moved sklearn modules)."""
    for old, new in _MOVED_MODULES.items():
        if old in line:
            return line.replace(old, new)

    match = re.match(r"^(\s*)from collections import (.+)$", line)
    if match and "collections" in message:
        indent, names = match.group(1), [n.strip() for n in match.group(2).split(",")]
        abc = [n for n in names if n.split(" as ")[0] in _COLLECTIONS_ABC]
        rest = [n for n in names if n not in abc]
        if not abc:
            return None
        out = [f"{indent}from collections.abc import {', '.join(abc)}"]
        if rest:
            out.append(f"{indent}from collections import {', '.join(rest)}")
        return "\n".join(out)
    return None


def rule_based_fix(original_code: str, error_traceback: str) -> Optional[str]:
    """Fix the line named in the traceback using simple rules. Returns None if no rule applies."""
    info = parse_traceback(error_traceback)
    line_no = info["line_no"]
    code_lines = original_code.split("\n")
    if line_no is None or not (1 <= line_no <= len(code_lines)):
        return None

    line = code_lines[line_no - 1]
    error_type = info["error_type"]
    if error_type == "ZeroDivisionError":
        fixed = _fix_zero_division(line)
    elif error_type == "IndexError":
        fixed = _fix_index_error(line)
    elif error_type in ("ImportError", "ModuleNotFoundError"):
        fixed = _fix_import_line(line, info["message"])
    else:
        fixed = None

    if fixed is None:
        return None
    code_lines[line_no - 1] = fixed
    return "\n".join(code_lines)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
def debug_code(
    original_code: str,
    error_traceback: str,
    hypothesis: str = "",
    llm: Optional[Any] = None,
) -> str:
    """Return corrected code for a failed script.

    Uses the LLM when one is given; if the LLM is missing, fails, or returns nothing,
    the offline rules are tried. If nothing can fix it, the original code is returned
    unchanged (the loop treats that as "no progress" and stops).
    """
    if llm is not None:
        try:
            reply = llm.complete(build_reflexion_prompt(original_code, error_traceback, hypothesis))
            fixed = extract_code(reply)
            if fixed.strip():
                return fixed
        except Exception:  # LLM down, bad key, timeout: fall back to the rules
            pass

    fixed = rule_based_fix(original_code, error_traceback)
    return fixed if fixed is not None else original_code


def _traceback_from(result: Dict[str, Any], timeout: int) -> str:
    """Text to show the debugger for a failed run."""
    if result.get("timed_out"):
        return f"TimeoutError: execution timed out after {timeout} seconds"
    return (result.get("stderr") or "").strip() or f"Process exited with code {result.get('exit_code')}"


def debug_and_run(
    code: str,
    hypothesis: str = "",
    max_retries: int = DEFAULT_MAX_RETRIES,
    llm: Optional[Any] = None,
    runner: Optional[Callable[[str, int], Dict[str, Any]]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Run code; on failure, debug and rerun, up to ``max_retries`` fix attempts."""
    if runner is None:
        from src.agents.code_executor_agent import run_code_detailed as runner

    history: List[Dict[str, Any]] = []
    retry_count = 0

    while True:
        result = runner(code, timeout)
        succeeded = result.get("exit_code") == 0 and not result.get("timed_out")
        history.append({
            "attempt": len(history) + 1,
            "code": code,
            "exit_code": result.get("exit_code"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "timed_out": bool(result.get("timed_out")),
        })

        if succeeded:
            status, output = "SUCCESS", (result.get("stdout") or "").strip()
            break
        if retry_count >= max_retries:
            status, output = "ERROR", _traceback_from(result, timeout)
            break

        fixed = debug_code(code, _traceback_from(result, timeout), hypothesis, llm)
        if fixed.strip() == code.strip():  # no fix found: stop instead of looping forever
            status, output = "ERROR", _traceback_from(result, timeout)
            break
        code = fixed
        retry_count += 1

    return {
        "execution_status": status,
        "retry_count": retry_count,
        "code": code,
        "output": output,
        "history": history,
    }


if __name__ == "__main__":
    buggy = "values = [4, 8, 15]\ncount = len([])\nprint(sum(values) / count)\n"
    outcome = debug_and_run(buggy, "Demo hypothesis")
    print(outcome["execution_status"], "after", outcome["retry_count"], "fix attempt(s)")
    print(outcome["code"])
    print(outcome["output"])
