"""
Execution cost and latency tracker
Member 5 Deliverable (Week 4)

Logs prompt tokens, completion tokens, dollar cost and latency for every agent node,
and totals them per discovery run. The numbers feed the dashboard's cost widget
(dashboard/components/cost_analytics.py).

Honesty rules
  * Every record says whether its token counts are exact (reported by the provider) or
    estimated (about 4 characters per token). Estimates are never presented as exact.
  * A model with no known price is not silently counted as free: its cost is None and the
    call is counted in ``unpriced_calls``. Local models (Ollama, llama, ...) cost 0.
  * The default prices are editable placeholders, not an official price list. Check the
    provider's pricing page and pass your own table: CostTracker(pricing={"my-model": (in, out)}).

Simplest hook for the whole pipeline (wraps every graph node, nothing else is edited):
    tracker = CostTracker(label=topic)
    with track_pipeline(tracker):              # the graph must be BUILT inside this block
        app = build_orchestrator(enable_critic=True)
        result = app.invoke({"research_topic": topic})
    append_run(tracker)                         # the dashboard widget reads this file

How to hook it into the agents WITHOUT editing them
    tracker = CostTracker()
    llm = tracker.wrap_llm(llm_client_from_env(), agent="hypothesis_generator", model="gpt-4o")
    critic_node = tracker.wrap_node("critic", critic_node)       # latency of a graph node
    with tracker.experiment("exp-1"):                            # latency of one experiment
        ...
    tracker.save("output/cost_log.json")                          # or append_run(tracker, path)

Any object with ``complete(prompt) -> str`` can be wrapped. If it also sets ``last_usage =
{"prompt_tokens": int, "completion_tokens": int}`` after each call, those exact numbers are used.
"""

import functools
import importlib
import json
import math
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# USD per 1,000,000 tokens as (input, output). EDITABLE PLACEHOLDERS: verify against the provider's pricing page.
DEFAULT_PRICING_PER_MILLION: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "gemini-1.5-flash": (0.075, 0.30),
}
LOCAL_MODEL_PREFIXES = ("ollama", "local", "llama", "mistral", "qwen", "phi", "gemma")
CHARS_PER_TOKEN = 4  # rough rule of thumb for English text

DEFAULT_LOG_PATH = "output/cost_log.json"


def estimate_tokens(text: Optional[str]) -> int:
    """Rough token count: about 4 characters per token (0 for empty text)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


@dataclass
class UsageRecord:
    """One LLM call."""

    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: Optional[float]  # None = the model's price is unknown
    latency_s: float = 0.0
    estimated: bool = False    # True = token counts were estimated, not reported by the provider
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostTracker:
    """Collects UsageRecords for one discovery run and summarises them."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        pricing: Optional[Dict[str, Tuple[float, float]]] = None,
        label: str = "",
    ):
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.label = label
        self.pricing = dict(DEFAULT_PRICING_PER_MILLION)
        if pricing:
            self.pricing.update(pricing)
        self.records: List[UsageRecord] = []
        self.node_latency_s: Dict[str, float] = {}
        self.experiments: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    # ----- pricing -----
    def price_for(self, model: str) -> Optional[Tuple[float, float]]:
        """(input, output) USD per 1M tokens. (0, 0) for local models, None if unknown."""
        key = (model or "").lower()
        if key in self.pricing:
            return self.pricing[key]
        for name in sorted(self.pricing, key=len, reverse=True):  # longest prefix first
            if key.startswith(name.lower()):
                return self.pricing[name]
        if key.startswith(LOCAL_MODEL_PREFIXES):
            return (0.0, 0.0)
        return None

    def cost_for(self, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        price = self.price_for(model)
        if price is None:
            return None
        return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000

    # ----- recording -----
    def record(
        self,
        agent: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float = 0.0,
        estimated: bool = False,
    ) -> UsageRecord:
        """Log one LLM call with its token counts."""
        if prompt_tokens < 0 or completion_tokens < 0 or latency_s < 0:
            raise ValueError("Token counts and latency can not be negative.")
        rec = UsageRecord(
            agent=agent,
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            cost_usd=self.cost_for(model, int(prompt_tokens), int(completion_tokens)),
            latency_s=float(latency_s),
            estimated=estimated,
        )
        with self._lock:
            self.records.append(rec)
        return rec

    def record_text(self, agent: str, model: str, prompt: str, completion: str, latency_s: float = 0.0) -> UsageRecord:
        """Log one call when the provider gave no token counts: they are estimated from the text."""
        return self.record(agent, model, estimate_tokens(prompt), estimate_tokens(completion), latency_s, estimated=True)

    # ----- hooks -----
    def wrap_llm(self, llm: Any, agent: str, model: str) -> "TrackedLLM":
        return TrackedLLM(llm, self, agent, model)

    def wrap_node(self, agent: str, node: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a graph node so its wall-clock time is added to ``node_latency_s[agent]``."""

        @functools.wraps(node)
        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return node(*args, **kwargs)
            finally:
                self.add_node_latency(agent, time.perf_counter() - start)

        return timed

    def add_node_latency(self, agent: str, seconds: float) -> None:
        with self._lock:
            self.node_latency_s[agent] = self.node_latency_s.get(agent, 0.0) + seconds

    @contextmanager
    def experiment(self, name: str):
        """Time one experiment: ``with tracker.experiment("exp-1"): ...``"""
        start = time.perf_counter()
        try:
            yield
        finally:
            with self._lock:
                self.experiments.append({"name": name, "latency_s": time.perf_counter() - start})

    # ----- summaries -----
    def total_cost(self) -> float:
        """Sum of known costs. Calls with an unknown price are NOT included (see unpriced_calls)."""
        return sum(r.cost_usd for r in self.records if r.cost_usd is not None)

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def by_agent(self) -> Dict[str, Dict[str, Any]]:
        agents: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            a = agents.setdefault(r.agent, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                            "total_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0})
            a["calls"] += 1
            a["prompt_tokens"] += r.prompt_tokens
            a["completion_tokens"] += r.completion_tokens
            a["total_tokens"] += r.total_tokens
            a["cost_usd"] += r.cost_usd or 0.0
            a["latency_s"] += r.latency_s
        for name, seconds in self.node_latency_s.items():
            agents.setdefault(name, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                     "total_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0})["node_latency_s"] = seconds
        return agents

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "calls": len(self.records),
            "prompt_tokens": sum(r.prompt_tokens for r in self.records),
            "completion_tokens": sum(r.completion_tokens for r in self.records),
            "total_tokens": self.total_tokens(),
            "total_cost_usd": self.total_cost(),
            "llm_latency_s": sum(r.latency_s for r in self.records),
            "estimated_calls": sum(1 for r in self.records if r.estimated),
            "unpriced_calls": sum(1 for r in self.records if r.cost_usd is None),
            "experiments": list(self.experiments),
            "by_agent": self.by_agent(),
        }

    # ----- saving -----
    def to_dict(self) -> Dict[str, Any]:
        return {**self.summary(), "records": [asdict(r) for r in self.records]}

    def save(self, path: Union[str, Path]) -> Path:
        """Write this run alone to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runs": [self.to_dict()]}, indent=2), encoding="utf-8")
        return path


class TrackedLLM:
    """Wraps anything with ``complete(prompt) -> str`` and logs every call to a CostTracker."""

    def __init__(self, llm: Any, tracker: CostTracker, agent: str, model: str):
        self._llm, self._tracker, self.agent, self.model = llm, tracker, agent, model

    def complete(self, prompt: str) -> str:
        start = time.perf_counter()
        try:
            reply = self._llm.complete(prompt)
        except Exception:
            # A failed call still took time; no tokens were produced.
            self._tracker.record(self.agent, self.model, estimate_tokens(prompt), 0,
                                 time.perf_counter() - start, estimated=True)
            raise
        latency = time.perf_counter() - start
        usage = getattr(self._llm, "last_usage", None)
        if isinstance(usage, dict) and "prompt_tokens" in usage and "completion_tokens" in usage:
            self._tracker.record(self.agent, self.model, usage["prompt_tokens"], usage["completion_tokens"], latency)
        else:
            self._tracker.record_text(self.agent, self.model, prompt, reply, latency)
        return reply

    def __getattr__(self, name: str) -> Any:  # everything else (provider, model, ...) comes from the wrapped client
        return getattr(self._llm, name)


# Orchestrator node functions and the agent name they are reported under.
PIPELINE_NODES: Dict[str, str] = {
    "literature_reviewer_node": "literature_reviewer",
    "hypothesis_generator_node": "hypothesis_generator",
    "code_executor_node": "code_executor",
    "debugger_node": "debugger",
    "critic_node": "critic",
}


@contextmanager
def track_pipeline(tracker: CostTracker, module: Any = None, nodes: Optional[Dict[str, str]] = None):
    """Time every graph node of the orchestrator while the block runs; the originals are restored after.

    LangGraph captures the node functions when the graph is built, so build AND run the graph inside
    the ``with`` block. Nodes missing from the module are skipped.
    """
    module = module or importlib.import_module("src.orchestrator")
    originals: Dict[str, Callable[..., Any]] = {}
    for attr, agent in (nodes or PIPELINE_NODES).items():
        node = getattr(module, attr, None)
        if callable(node):
            originals[attr] = node
            setattr(module, attr, tracker.wrap_node(agent, node))
    try:
        yield tracker
    finally:
        for attr, node in originals.items():
            setattr(module, attr, node)


# ---------------------------------------------------------------------------
# A log file with many runs (what the dashboard reads)
# ---------------------------------------------------------------------------
def load_runs(path: Union[str, Path] = DEFAULT_LOG_PATH) -> List[Dict[str, Any]]:
    """Read all saved runs. A missing, empty or corrupt file gives an empty list."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    runs = data.get("runs") if isinstance(data, dict) else None
    return [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []


def append_run(tracker: CostTracker, path: Union[str, Path] = DEFAULT_LOG_PATH) -> Path:
    """Add this run to the log file (creating it if needed); earlier runs are kept."""
    path = Path(path)
    runs = load_runs(path)
    runs = [r for r in runs if r.get("run_id") != tracker.run_id] + [tracker.to_dict()]  # re-saving replaces
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    return path


def format_cost(value: Optional[float]) -> str:
    """Dollar amount for display; very small costs keep enough digits to be visible."""
    if value is None:
        return "unknown"
    return f"${value:.4f}" if value < 1 else f"${value:,.2f}"
