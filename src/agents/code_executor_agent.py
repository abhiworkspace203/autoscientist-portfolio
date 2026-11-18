"""
Code Executor Agent
Member 5 Deliverable

Turns a hypothesis into a Python experiment script, runs it, and returns
(code, output). The orchestrator calls generate_and_run_code().

Until the LLM is connected, the script is a real offline proxy experiment
(decision tree vs random forest, repeated cross-validated macro-F1).
The prompt template is ready for the LLM to be plugged in later, and the
subprocess runner will be replaced by a Docker sandbox in a later week.
"""

import hashlib
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Prompt template (to be sent to the LLM once an API key is set up)
# ---------------------------------------------------------------------------
CODE_EXECUTOR_PROMPT = """You are the Code Executor Agent in an autonomous scientific discovery system.

Hypothesis:
{hypothesis}

Task:
Write ONE self-contained Python script that tests this hypothesis.

Rules:
- Use only the Python standard library unless told otherwise.
- Set a random seed so results are reproducible.
- Compare a baseline against the proposed method over several trials.
- Print the average metric for each, then a final line: Metric delta: <number>
- Return only the code, with no explanations.
"""

# ---------------------------------------------------------------------------
# Offline proxy experiment (used until the LLM is connected)
#
# The fallback hypotheses (hypothesis_agent.py) claim that "a model variant ...
# achieves a higher mean cross-validated macro-F1 than an otherwise identical
# baseline on a controlled proxy task" and suggest scikit-learn proxy datasets.
# This script runs that comparison for real: a single decision tree (baseline)
# against a random-forest variant, scored with repeated stratified k-fold
# cross-validation on the same folds. All numbers are measured, never invented,
# so the statistics validator and the critic judge a real result.
#
# __HYPOTHESIS__, __DATASET__ and __SEED__ are filled in by generate_code().
# ---------------------------------------------------------------------------
PROXY_DATASETS = ("breast_cancer", "wine", "noisy_synthetic")

BASELINE_SCRIPT = '''import json
import statistics
import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

HYPOTHESIS = __HYPOTHESIS__
DATASET = __DATASET__
SEED = __SEED__
N_SPLITS, N_REPEATS = 5, 8          # 40 cross-validated scores per model


def load_proxy_task(name):
    if name == "breast_cancer":
        return load_breast_cancer(return_X_y=True)
    if name == "wine":
        return load_wine(return_X_y=True)
    return make_classification(n_samples=600, n_features=20, n_informative=6,
                               flip_y=0.08, random_state=SEED)


X, y = load_proxy_task(DATASET)
folds = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

baseline_scores, variant_scores = [], []
for train, test in folds.split(X, y):
    baseline_model = DecisionTreeClassifier(random_state=SEED)
    variant_model = RandomForestClassifier(n_estimators=50, random_state=SEED)
    for model, scores in ((baseline_model, baseline_scores), (variant_model, variant_scores)):
        model.fit(X[train], y[train])
        scores.append(f1_score(y[test], model.predict(X[test]), average="macro"))

print("Controlled proxy task:", DATASET, "| seed", SEED)
print("Baseline model: decision tree | Model variant: random forest (identical folds)")
print("Repeated stratified cross-validated macro-F1, n =", len(baseline_scores), "per model")
print("Baseline mean:", round(statistics.mean(baseline_scores), 4))
print("Method mean:", round(statistics.mean(variant_scores), 4))
print("Metric delta:", round(statistics.mean(variant_scores) - statistics.mean(baseline_scores), 4))
print("RESULTS_JSON: " + json.dumps({"baseline": baseline_scores, "treatment": variant_scores}))
'''


def build_prompt(hypothesis: str) -> str:
    """Fill the prompt template with the hypothesis."""
    return CODE_EXECUTOR_PROMPT.format(hypothesis=hypothesis.strip())


def _proxy_settings(hypothesis: str) -> Tuple[str, int]:
    """Pick a proxy dataset and seed from the hypothesis text (stable per topic)."""
    digest = int(hashlib.sha256(hypothesis.strip().encode("utf-8")).hexdigest(), 16)
    return PROXY_DATASETS[digest % len(PROXY_DATASETS)], digest % 10_000


def generate_code(hypothesis: str) -> str:
    """Create a Python experiment script for the hypothesis.

    Later: send build_prompt(hypothesis) to the LLM and return its code.
    """
    dataset, seed = _proxy_settings(hypothesis)
    return (BASELINE_SCRIPT
            .replace("__HYPOTHESIS__", repr(hypothesis.strip()))
            .replace("__DATASET__", repr(dataset))
            .replace("__SEED__", str(seed)))


def run_code_detailed(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Run the script in a separate process and return a structured result.

    Returns {"exit_code", "stdout", "stderr", "timed_out"}, the same shape the
    Docker sandbox runner returns, so the two are interchangeable (the debugger
    loop accepts either).
    Later: replace this with a Docker sandbox.
    """
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        def as_text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")

        return {
            "exit_code": -1,
            "stdout": as_text(exc.stdout),
            "stderr": as_text(exc.stderr),
            "timed_out": True,
        }
    finally:
        os.remove(path)


def run_code(code: str, timeout: int = 30) -> str:
    """Run the script and return its output as a single string."""
    result = run_code_detailed(code, timeout)
    if result["timed_out"]:
        return f"Execution failed: timed out after {timeout} seconds."
    if result["exit_code"] != 0:
        return f"Execution failed (exit code {result['exit_code']}):\n{result['stderr'].strip()}"
    return result["stdout"].strip()


def generate_and_run_code(hypothesis: str) -> Tuple[str, str]:
    """Entry point used by src/orchestrator.py. Returns (code, output)."""
    code = generate_code(hypothesis)
    output = run_code(code)
    return code, output

class CodeExecutor:
    """Convenience class wrapper for generate_and_run_code."""

    @staticmethod
    def run(hypothesis: str) -> Tuple[str, str]:
        return generate_and_run_code(hypothesis)


if __name__ == "__main__":
    demo = "Dynamic token-level pruning reduces inference memory by 30% with <1% perplexity loss."
    code, output = generate_and_run_code(demo)
    print(output)
