"""
Episodic Memory Engine for Multi-Day Scientific Discovery.
Member 1 (Gayatri Bhakta - Team Lead Deliverable, Week 3).

Maintains long-horizon coherence across multiple experimental runs and multi-day research cycles.
Stores structured episode records:
- research topic
- hypothesis tested
- code executed
- execution results & metric deltas
- statistical critic verdict & critique
- lessons learned

Provides:
- Disk persistence (data/experiments/episodes.json)
- Querying past episodes by topic, verdict, or recentness
- Distilling past experiments into a compact context summary for the next research loop
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.memory.context_manager import estimate_tokens, extract_numerical_facts


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class Episode(BaseModel):
    """Structured record of one completed scientific discovery iteration."""

    episode_id: str = Field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    topic: str
    hypothesis: str
    verdict: str = "PENDING"  # ACCEPTED, REJECTED, NEEDS_REPLICATION, ERROR
    metric_delta: Optional[float] = None
    p_value: Optional[float] = None
    execution_status: str = "SUCCESS"  # SUCCESS, ERROR
    summary: str = ""
    lessons_learned: List[str] = Field(default_factory=list)
    retry_count: int = 0
    iteration: int = 1


# ---------------------------------------------------------------------------
# Episodic Memory Store
# ---------------------------------------------------------------------------

DEFAULT_STORAGE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "experiments" / "episodes.json"


class EpisodicMemoryStore:
    """
    Persistent Long-Horizon Memory Store for the Multi-Agent Discovery Pipeline.
    Preserves historical discoveries, rejected hypotheses, and experimental benchmarks
    across multi-day research sessions.
    """

    def __init__(self, storage_path: Optional[str | Path] = None):
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_STORAGE_PATH
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Create parent directory if it does not exist."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_raw(self) -> List[Dict[str, Any]]:
        """Load raw episode dictionaries from disk."""
        if not self.storage_path.exists():
            return []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_raw(self, raw_episodes: List[Dict[str, Any]]) -> None:
        """Save raw episode dictionaries to disk."""
        self._ensure_storage_dir()
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(raw_episodes, f, indent=2, ensure_ascii=False)

    def add_episode(self, episode: Episode | Dict[str, Any]) -> Episode:
        """Add a new episode to persistent memory."""
        if isinstance(episode, dict):
            ep_obj = Episode(**episode)
        else:
            ep_obj = episode

        episodes = self._load_raw()
        episodes.append(ep_obj.model_dump())
        self._save_raw(episodes)
        return ep_obj

    def record_from_state(self, state: Dict[str, Any]) -> Episode:
        """
        Extract episode findings directly from an AgentGraphState dictionary
        and save to persistent episodic memory.
        """
        topic = state.get("research_topic", "General Scientific Discovery")
        hypothesis = state.get("selected_hypothesis", "No hypothesis selected")
        verdict = state.get("critic_verdict", "ACCEPTED" if state.get("execution_status") == "SUCCESS" else "ERROR")
        exec_status = state.get("execution_status", "SUCCESS")
        result = state.get("execution_result", "")
        critique = state.get("critic_critique", "")

        # Extract metric delta if present
        delta_val = None
        delta_match = re.search(r"Metric delta:\s*([-+]?\d*\.?\d+)", result, re.IGNORECASE)
        if delta_match:
            try:
                delta_val = float(delta_match.group(1))
            except ValueError:
                pass

        # Extract p-value if present
        p_val = None
        p_match = re.search(r"p[-_\s]*value[:=]?\s*(\d*\.?\d+)", result + " " + critique, re.IGNORECASE)
        if p_match:
            try:
                p_val = float(p_match.group(1))
            except ValueError:
                pass

        # Formulate lessons learned
        lessons: List[str] = []
        if verdict == "ACCEPTED":
            lessons.append(f"Hypothesis validated: '{hypothesis[:60]}...' succeeded with delta {delta_val}.")
        elif verdict == "REJECTED":
            lessons.append(f"Hypothesis rejected ({critique[:80]}...). Avoid repeating this exact formulation.")
        elif exec_status == "ERROR":
            lessons.append(f"Execution failed: {state.get('error_traceback', '')[:80]}...")

        episode = Episode(
            topic=topic,
            hypothesis=hypothesis,
            verdict=verdict,
            execution_status=exec_status,
            metric_delta=delta_val,
            p_value=p_val,
            summary=result[:200] if result else "",
            lessons_learned=lessons,
            retry_count=state.get("retry_count", 0),
            iteration=state.get("experiment_iteration", 1),
        )

        return self.add_episode(episode)

    def get_episodes(
        self,
        topic: Optional[str] = None,
        verdict: Optional[str] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """Query stored episodes, optionally filtering by topic or verdict."""
        raw_list = self._load_raw()
        filtered: List[Episode] = []

        for item in reversed(raw_list):
            if topic and topic.lower() not in item.get("topic", "").lower():
                continue
            if verdict and verdict.upper() != item.get("verdict", "").upper():
                continue
            filtered.append(Episode(**item))
            if len(filtered) >= limit:
                break

        return filtered

    def get_lessons_learned(self, topic: Optional[str] = None, max_items: int = 5) -> List[str]:
        """Extract historical lessons learned across relevant past episodes."""
        episodes = self.get_episodes(topic=topic, limit=20)
        all_lessons: List[str] = []
        seen = set()

        for ep in episodes:
            for lesson in ep.lessons_learned:
                clean = lesson.strip()
                if clean and clean not in seen:
                    seen.add(clean)
                    all_lessons.append(clean)
                if len(all_lessons) >= max_items:
                    return all_lessons

        return all_lessons

    def format_episodic_summary(self, topic: str, max_tokens: int = 300) -> str:
        """
        Produce a compact, token-budgeted memory briefing of previous experiments
        to prime new agent runs without causing context overflow.
        """
        episodes = self.get_episodes(topic=topic, limit=5)
        if not episodes:
            return ""

        lessons = self.get_lessons_learned(topic=topic, max_items=3)

        lines = [f"[Prior Experimental History for '{topic}'] (Total past runs: {len(episodes)}):"]
        if lessons:
            lines.append("Key Lessons:")
            for les in lessons:
                lines.append(f"  - {les}")

        lines.append("Recent Runs:")
        for ep in episodes:
            status_flag = "VALIDATED" if ep.verdict == "ACCEPTED" else ep.verdict
            metrics_str = []
            if ep.metric_delta is not None:
                metrics_str.append(f"delta={ep.metric_delta:+.2f}")
            if ep.p_value is not None:
                metrics_str.append(f"p={ep.p_value:.3f}")
            metric_tag = f" ({', '.join(metrics_str)})" if metrics_str else ""
            lines.append(f"• Run {ep.episode_id} [{status_flag}{metric_tag}]: {ep.hypothesis[:50]}...")

        summary = "\n".join(lines)
        # Verify token budget
        if estimate_tokens(summary) > max_tokens:
            char_budget = max_tokens * 4
            summary = summary[:char_budget] + "\n...[additional history truncated]"

        return summary

    def clear(self) -> None:
        """Clear the memory store (primarily for unit testing isolation)."""
        if self.storage_path.exists():
            try:
                os.remove(self.storage_path)
            except OSError:
                pass
