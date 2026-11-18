"""
Benchmark report (Member 3, Week 4).

Turns the scores from benchmark_evaluator.py into:
  - comparison tables (Agent vs Human average, per domain, per paper)
  - a radar (spider) chart of the 4 dimensions: docs/figures/benchmark_radar.svg
    (plus a .png when matplotlib is installed)
  - docs/evaluation_benchmark_report.md

    python -m src.evaluation.benchmark_report
    python -m src.evaluation.benchmark_report --agent examples/sample_agent_output.json
"""
from __future__ import annotations

import json
import math
from html import escape
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.benchmark_evaluator import DIMENSIONS, run_benchmark
from src.evaluation.holdout_curator import HOLDOUT_DIR

SCORES_FILE = HOLDOUT_DIR / "benchmark_scores.json"
REPORT_FILE = Path("docs/evaluation_benchmark_report.md")
FIGURE_DIR = Path("docs/figures")
LABELS = {"novelty": "Novelty", "grounding": "Literature Grounding",
          "empirical_validity": "Empirical Validity", "formatting": "Formatting & Structure",
          "overall": "Overall"}
AGENT_COLOR = "#0FA3B1"
HUMAN_COLOR = "#F2A541"


# ---------------------------------------------------------------- tables
def domain_averages(human: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in human:
        groups.setdefault(s.get("domain") or "Other", []).append(s)
    out = {}
    for domain, rows in sorted(groups.items()):
        avg = {d: round(sum(r[d] for r in rows) / len(rows), 2) for d in DIMENSIONS}
        avg["overall"] = round(sum(avg.values()) / len(DIMENSIONS), 2)
        avg["papers"] = len(rows)
        out[domain] = avg
    return out


def comparison_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per dimension: agent, human average, gap and who is ahead."""
    agent = result.get("agent")
    human = result["human_average"]
    rows = []
    for d in DIMENSIONS + ["overall"]:
        a = agent[d] if agent else None
        h = human[d]
        gap = round(a - h, 2) if a is not None else None
        leader = "-" if gap is None else "Agent" if gap > 0.25 else "Human" if gap < -0.25 else "Tie"
        rows.append({"dimension": LABELS[d], "agent": a, "human": h, "gap": gap, "leader": leader})
    return rows


def _fmt(x: Optional[float], signed: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def markdown_comparison(result: Dict[str, Any]) -> str:
    lines = ["| Dimension | Agent | Human average | Gap (Agent - Human) | Ahead |",
             "| --- | --- | --- | --- | --- |"]
    for r in comparison_rows(result):
        name = f"**{r['dimension']}**" if r["dimension"] == "Overall" else r["dimension"]
        lines.append(f"| {name} | {_fmt(r['agent'])} | {_fmt(r['human'])} | "
                     f"{_fmt(r['gap'], signed=True)} | {r['leader']} |")
    return "\n".join(lines)


def markdown_domains(result: Dict[str, Any]) -> str:
    lines = ["| Domain | Papers | Novelty | Grounding | Empirical | Formatting | Overall |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for domain, a in domain_averages(result["human"]).items():
        lines.append(f"| {domain} | {a['papers']} | {a['novelty']:.2f} | {a['grounding']:.2f} | "
                     f"{a['empirical_validity']:.2f} | {a['formatting']:.2f} | {a['overall']:.2f} |")
    return "\n".join(lines)


def markdown_papers(result: Dict[str, Any]) -> str:
    lines = ["| Paper | Domain | Title | Novelty | Grounding | Empirical | Formatting | Overall |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = list(result["human"]) + ([result["agent"]] if result.get("agent") else [])
    for s in rows:
        title = s["title"].replace("|", "/")
        title = title if len(title) <= 60 else title[:57] + "..."
        pid = s["paper_id"]
        if pid.startswith("arxiv:"):
            pid = f"[{pid}](https://arxiv.org/abs/{pid.split(':', 1)[1]})"
        if s["source"] == "agent":
            pid = f"**Agent** ({s['paper_id']})"
        lines.append(f"| {pid} | {s.get('domain', '')} | {title} | {s['novelty']:.2f} | "
                     f"{s['grounding']:.2f} | {s['empirical_validity']:.2f} | "
                     f"{s['formatting']:.2f} | {s['overall']:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- radar chart
def radar_points(values: Sequence[float], cx: float, cy: float, radius: float,
                 max_value: float = 10.0) -> List[tuple]:
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        angle = -math.pi / 2 + 2 * math.pi * i / n
        r = radius * max(0.0, min(v, max_value)) / max_value
        pts.append((round(cx + r * math.cos(angle), 1), round(cy + r * math.sin(angle), 1)))
    return pts


def radar_svg(series: Dict[str, Sequence[float]], labels: Sequence[str],
              colors: Optional[Dict[str, str]] = None, title: str = "Agent vs Human") -> str:
    """Dependency-free SVG radar chart (renders on GitHub)."""
    colors = colors or {}
    w, h, cx, cy, radius = 820, 580, 410, 300, 190
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}" font-family="Helvetica, Arial, sans-serif">',
             f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
             f'<text x="{cx}" y="34" text-anchor="middle" font-size="20" font-weight="bold" '
             f'fill="#1B2A49">{escape(title)}</text>']
    for ring in (2, 4, 6, 8, 10):
        pts = radar_points([ring] * len(labels), cx, cy, radius)
        parts.append(f'<polygon points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="none" '
                     f'stroke="#D5DCE4" stroke-width="1"/>')
        parts.append(f'<text x="{cx + 4}" y="{cy - radius * ring / 10 + 4}" font-size="11" '
                     f'fill="#8A97A8">{ring}</text>')
    for (x, y), label in zip(radar_points([10] * len(labels), cx, cy, radius), labels):
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#D5DCE4"/>')
        lx, ly = cx + (x - cx) * 1.16, cy + (y - cy) * 1.12
        anchor = "middle" if abs(lx - cx) < 10 else "start" if lx > cx else "end"
        parts.append(f'<text x="{lx}" y="{ly + 5}" text-anchor="{anchor}" font-size="14" '
                     f'fill="#1E293B">{escape(label)}</text>')
    for name, values in series.items():
        color = colors.get(name, "#3E7CB1")
        pts = radar_points(values, cx, cy, radius)
        parts.append(f'<polygon points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="{color}" '
                     f'fill-opacity="0.22" stroke="{color}" stroke-width="2.5"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
    for i, name in enumerate(series):
        y = h - 30
        x = cx - 110 + i * 150
        parts.append(f'<rect x="{x}" y="{y - 11}" width="14" height="14" '
                     f'fill="{colors.get(name, "#3E7CB1")}"/>')
        parts.append(f'<text x="{x + 20}" y="{y + 1}" font-size="14" fill="#1E293B">{escape(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def radar_png(series: Dict[str, Sequence[float]], labels: Sequence[str], path: Path,
              colors: Optional[Dict[str, str]] = None, title: str = "Agent vs Human") -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    colors = colors or {}
    n = len(labels)
    angles = [2 * math.pi * i / n for i in range(n)]
    fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    for name, values in series.items():
        vals = list(values) + [values[0]]
        ang = angles + [angles[0]]
        c = colors.get(name, "#3E7CB1")
        ax.plot(ang, vals, color=c, linewidth=2.5, label=name)
        ax.fill(ang, vals, color=c, alpha=0.22)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for ang, label in zip(angles, labels):
        x = math.sin(ang)
        ha = "center" if abs(x) < 0.1 else "left" if x > 0 else "right"
        ax.text(ang, 11.2, label, ha=ha, va="center", fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_title(title, fontsize=15, fontweight="bold", pad=44)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(series), frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return True


def build_charts(result: Dict[str, Any], figure_dir: Path = FIGURE_DIR) -> Dict[str, str]:
    labels = [LABELS[d] for d in DIMENSIONS]
    series = {}
    if result.get("agent"):
        series["Agent"] = [result["agent"][d] for d in DIMENSIONS]
    series["Human average"] = [result["human_average"][d] for d in DIMENSIONS]
    colors = {"Agent": AGENT_COLOR, "Human average": HUMAN_COLOR}
    figure_dir.mkdir(parents=True, exist_ok=True)
    svg_path = figure_dir / "benchmark_radar.svg"
    svg_path.write_text(radar_svg(series, labels, colors), encoding="utf-8")
    out = {"svg": svg_path.as_posix()}
    png_path = figure_dir / "benchmark_radar.png"
    if radar_png(series, labels, png_path, colors):
        out["png"] = png_path.as_posix()
    return out


# ---------------------------------------------------------------- report
def findings(result: Dict[str, Any]) -> List[str]:
    """Plain-language findings computed from the scores."""
    out: List[str] = []
    human = result["human"]
    avg = result["human_average"]
    agent = result.get("agent")
    if agent:
        rows = [r for r in comparison_rows(result) if r["dimension"] != "Overall"]
        best = max(rows, key=lambda r: r["gap"])
        worst = min(rows, key=lambda r: r["gap"])
        out.append(f"Overall the agent scored {agent['overall']:.2f} against a human average of "
                   f"{avg['overall']:.2f} ({agent['overall'] - avg['overall']:+.2f}).")
        if best["gap"] > 0:
            out.append(f"The agent's biggest lead is **{best['dimension']}** ({best['gap']:+.2f}): "
                       "every agent result passes through the statistics validator, while many "
                       "human papers do not report p-values or confidence intervals.")
        if worst["gap"] < 0:
            out.append(f"The agent's biggest gap is **{worst['dimension']}** ({worst['gap']:+.2f}): "
                       "its hypothesis stays close to the papers it retrieved, so it extends existing "
                       "work rather than proposing a clearly new direction.")
    weakest = min(DIMENSIONS, key=lambda d: avg[d])
    out.append(f"Across the {len(human)} human papers the weakest dimension is "
               f"**{LABELS[weakest]}** (average {avg[weakest]:.2f}).")
    domains = domain_averages(human)
    if len(domains) > 1:
        top = max(domains.items(), key=lambda kv: kv[1]["overall"])
        low = min(domains.items(), key=lambda kv: kv[1]["overall"])
        out.append(f"By domain, {top[0]} papers scored highest overall ({top[1]['overall']:.2f}) "
                   f"and {low[0]} lowest ({low[1]['overall']:.2f}).")
    if human:
        best_paper = max(human, key=lambda s: s["overall"])
        out.append(f"The highest-scoring human paper is {best_paper['paper_id']} "
                   f"({best_paper['overall']:.2f}).")
    return out


LIMITATIONS = [
    "**Different evidence for agent and humans.** The agent's empirical score uses the real p-value, "
    "effect size and confidence interval from the critic; human papers are scored on what their text "
    "reports, so a strong human paper that does not print p-values scores lower.",
    "**Novelty is word-based.** TF-IDF similarity measures shared vocabulary, not shared ideas. "
    "Human papers are compared with the other holdout papers, which come from different domains, so "
    "their novelty scores are high by construction.",
    "**Grounding for human papers is a proxy.** We count references instead of checking each one.",
    "**PDF text extraction is imperfect.** Two-column layouts and appendices can hide section "
    "headings or merge references, which lowers formatting and grounding scores slightly.",
    "**Small holdout set.** 9 papers (3 per domain) is enough for a roadmap, not for statistical "
    "claims about agents versus humans.",
]

NEXT_STEPS = [
    "Score the synthesizer agent's real generated paper instead of the sample.",
    "Run the agent on the same topics as the holdout papers for a like-for-like comparison.",
    "Replace TF-IDF with sentence embeddings (once the Mac PyTorch issue is fixed) for semantic novelty.",
    "Add an LLM-as-judge score next to the rule-based score and check how often they agree.",
    "Grow the holdout set to 30+ papers and report confidence intervals for each dimension.",
]


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def build_report(result: Dict[str, Any], charts: Dict[str, str],
                 report_path: Path = REPORT_FILE) -> str:
    rel = lambda p: Path(p).relative_to(report_path.parent).as_posix() if Path(p).is_relative_to(report_path.parent) else p  # noqa: E731
    chart = rel(charts.get("png") or charts["svg"])
    agent_note = ("The agent row uses **examples/sample_agent_output.json**, a sample paper. "
                  "Replace it with the synthesizer agent's real output for final numbers."
                  if result.get("agent") and str(result["agent"].get("paper_id", "")).startswith("agent:sample")
                  else "")
    return f"""# Evaluation Benchmark Report

Member 3 (Afrah) · Week 4 · generated {date.today().isoformat()} by `python -m src.evaluation.benchmark_report`

The agent's research paper is graded against a holdout set of {len(result['human'])} recent
human-written papers (2024–2026, AI, computational biology and physics) on 4 dimensions, each 0–10.

## Agent vs Human

![Agent vs Human radar chart]({chart})

{markdown_comparison(result)}

{agent_note}

## Human papers by domain

{markdown_domains(result)}

## Every paper

{markdown_papers(result)}

## Key findings

{_bullets(findings(result))}

## How each score is computed

| Dimension | Agent paper | Human paper |
| --- | --- | --- |
| Novelty | 10 × (1 − overlap), overlap = 0.7 × max + 0.3 × mean TF-IDF similarity to the retrieved papers and the holdout set | Same formula against the other holdout papers |
| Literature Grounding | 60% citations that were really retrieved + 40% relevance to the hypothesis, scaled down below 3 citations; made-up IDs are listed | Number of references (25 or more = 10) |
| Empirical Validity | p < 0.05 = 6, effect size ≥ 0.5 = +2, 95% CI excluding 0 = +2 (not significant: max 3) | Evidence in the text: p-values 3, statistical test 2, uncertainty (±, CI, std) 2, repeated runs 1, tables 1, figures 1 |
| Formatting & Structure | Required sections (Abstract, Introduction, Method, Results, Conclusion, References) 6, abstract 100–300 words 1.5, figures 1, tables 0.5, 5+ references 1 | Same |

Overall = mean of the 4 dimensions. Scoring is deterministic and offline, so every run gives the same numbers.

## Limitations

{_bullets(LIMITATIONS)}

## Next steps

{_bullets(NEXT_STEPS)}

## Reproduce

```bash
python -m src.evaluation.holdout_curator
python -m src.evaluation.benchmark_evaluator --agent examples/sample_agent_output.json --save data/holdout_papers/benchmark_scores.json
python -m src.evaluation.benchmark_report
```
"""


def generate(result: Dict[str, Any], report_path: Path = REPORT_FILE,
             figure_dir: Path = FIGURE_DIR) -> Dict[str, str]:
    charts = build_charts(result, figure_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(result, charts, report_path), encoding="utf-8")
    return {"report": report_path.as_posix(), **charts}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build benchmark tables, radar chart and report")
    ap.add_argument("--scores", default=str(SCORES_FILE),
                    help="scores JSON from benchmark_evaluator --save")
    ap.add_argument("--agent", default=None, help="re-score using this agent output JSON instead")
    ap.add_argument("--report", default=str(REPORT_FILE))
    ap.add_argument("--figures", default=str(FIGURE_DIR))
    args = ap.parse_args()

    if args.agent:
        result = run_benchmark(json.loads(Path(args.agent).read_text(encoding="utf-8")))
    elif Path(args.scores).is_file():
        result = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    else:
        print(f"No scores at {args.scores}. Run: python -m src.evaluation.benchmark_evaluator "
              f"--agent examples/sample_agent_output.json --save {args.scores}")
        return
    if not result.get("human"):
        print("No human papers in the scores. Run the holdout curator first.")
        return
    paths = generate(result, Path(args.report), Path(args.figures))
    print(markdown_comparison(result))
    print("\nWritten:")
    for p in paths.values():
        print(f"  {p}")


if __name__ == "__main__":
    main()
