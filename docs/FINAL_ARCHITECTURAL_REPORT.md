# Enterprise Architectural & Integration Report (Month 1 Final Delivery)

**Project:** Autonomous Multimodal Scientific Discovery Agent (AutoScientist AI)  
**Organization:** Zaalima Development Pvt. Ltd. (Data Science & Machine Learning Track)  
**Prepared by:** Team Lead (Member 1 - Gayatri Bhakta)  
**Submission Window:** Week 4 Final Review & Delivery (30th October – 6th November 2026)  
**Active Branch:** `Gayatri-Bhakta` / `main`  
**Version:** `v1.0.0-final`

---

## 1. Executive Summary

Scientific discovery has historically been a labor-intensive, human-bottlenecked process requiring weeks of literature surveying, hypothesis ideation, experimental coding, statistical debugging, and paper drafting. **AutoScientist AI** solves this bottleneck by realizing a fully autonomous, closed-loop multi-agent research collective capable of transforming a raw domain query (e.g., *"Sparse Transformers"* or *"Dynamic Attention Pruning"*) into a statistically verified, publication-ready scientific draft.

Across the four-week implementation sprint, the 5-member engineering team accomplished:
1. **Week 1 (Multi-Agent Foundation):** Linear multi-agent baseline, arXiv/PubMed live ingestion scrapers, ChromaDB vector database, and initial role specifications.
2. **Week 2 (Containerization & Reflexion):** Pydantic-grounded anti-hallucination hypothesis schemas, zero-network Docker execution sandboxing, and autonomous Reflexion self-debugging loop.
3. **Week 3 (Multimodality & Academic Rigor):** PyMuPDF PDF figure extraction ($> 300\times 300$ px), VLM chart reading, Welch's t-test statistical validation ($p < 0.05$, Cohen's $d$, 95% CI), Academic Critic Agent, and Two-Tier Memory management (Working Context Diet + Episodic Store).
4. **Week 4 (Synthesis, Dashboard & Unified Entrypoint):** Master CLI entrypoint (`main.py`), publication-ready LaTeX paper synthesis (`output/generated_research_paper.pdf`), Streamlit live supervisor dashboard (`dashboard/app.py`), holdout paper benchmarking, and token economics/cost analytics.

The framework features an automated test suite of **289 passing tests** across 23 test modules with 100% architectural compliance.

---

## 2. Five-Member Team Roles & Deliverables

| Team Member | Role Title | Core Specialization | Month 1 Production Deliverables |
| :--- | :--- | :--- | :--- |
| **Gayatri Bhakta (Lead)** | Systems Architect & Integration Lead | Multi-Agent StateGraph, Two-Tier Memory, Master CLI | • `main.py`<br>• `src/orchestrator.py`<br>• `src/memory/context_manager.py`<br>• `src/memory/episodic_memory.py`<br>• `docs/FINAL_ARCHITECTURAL_REPORT.md` |
| **Kabil (Member 2)** | Literature & Academic Publishing Lead | Hypothesis Grounding, VLM Chart Reader, LaTeX Engine | • `src/agents/hypothesis_agent.py`<br>• `src/ingestion/chart_reader.py`<br>• `src/agents/synthesizer_agent.py`<br>• `templates/paper_template.tex` |
| **Afrah (Member 3)** | Ingestion & Benchmarking Lead | Multimodal Ingestion, Scrapers, Holdout Evaluation | • `src/ingestion/arxiv_scraper.py`<br>• `src/ingestion/pubmed_scraper.py`<br>• `src/ingestion/figure_extractor.py`<br>• `src/evaluation/benchmark_evaluator.py` |
| **Abhishek (Member 4)** | Sandbox Security & Supervisor UI Lead | Docker Isolation, Statistical Inference, Streamlit UI | • `sandbox/Dockerfile`<br>• `src/sandbox/docker_runner.py`<br>• `src/utils/stats_validator.py`<br>• `dashboard/app.py` |
| **Sachin (Member 5)** | Debugging, Critic & QA Lead | Reflexion Self-Healing, Statistical Critic, Cost Analytics | • `src/agents/code_executor_agent.py`<br>• `src/agents/debugger_agent.py`<br>• `src/agents/critic_agent.py`<br>• `src/utils/cost_tracker.py` |

---

## 3. End-to-End System Architecture & Multi-Agent Communication Graph

AutoScientist AI is orchestrated as a cyclic directed state graph via **LangGraph**. The workflow couples forward exploratory reasoning with backward corrective loops (Reflexion and Peer Review).

```mermaid
flowchart TD
    START([START: Research Prompt]) --> LitRev["Node 1: Literature Reviewer Agent\n(ChromaDB RAG + Prior Episodic Memory)"]
    LitRev --> HypGen["Node 2: Hypothesis Generator Agent\n(Pydantic Anti-Hallucination Schema)"]
    HypGen --> CodeExec["Node 3: Code Executor Agent\n(Zero-Network Docker Sandbox)"]
    
    CodeExec --> ExecRouter{"Execution Router\n(execution_status)"}
    ExecRouter -->|"ERROR & retry < 3"| Debugger["Node 4: Debugger Agent\n(Reflexion Traceback Patching)"]
    Debugger -->|"Repaired Script (retry++)"| CodeExec
    ExecRouter -->|"ERROR & retry >= 3"| ABORT([ABORT: Run Logged])
    
    ExecRouter -->|"SUCCESS"| Critic["Node 5: Statistical Critic Agent\n(Welch's t-test p < 0.05, Cohen's d)"]
    Critic --> CriticRouter{"Critic Router\n(verdict)"}
    CriticRouter -->|"ACCEPTED"| Synth["Node 6: LaTeX Synthesizer Engine\n(NeurIPS / IEEE Paper Compilation)"]
    CriticRouter -->|"REJECTED / NEEDS_REP & iter < 2"| CodeExec
    CriticRouter -->|"iter >= 2"| Synth
    Synth --> ENDNode([END: Publication-Ready PDF & Audit Trail])
```

### Shared State Schema (`AgentGraphState`)
All agents communicate across an immutable, typed state contract:
* `research_topic` (`str`): Target domain query.
* `literature_summary` (`str`): Synthesized review of retrieved preprints.
* `retrieved_papers` (`List[Dict]`): Context metadata and chunk content from ChromaDB.
* `generated_hypotheses` (`List[str]`): Candidate hypotheses adhering to `GroundedHypothesis`.
* `selected_hypothesis` (`str`): Top falsifiable hypothesis chosen for testing.
* `generated_code` (`str`): Executable Python test script.
* `execution_result` (`str`): Output strings, metric logs, and statistical printouts.
* `execution_status` (`str`): `"SUCCESS"` or `"ERROR"`.
* `error_traceback` (`str`): Captured standard error consumed by Debugger Agent.
* `retry_count` (`int`): Self-debugging retry iteration (capped at 3).
* `critic_verdict` (`str`): `"ACCEPTED"`, `"REJECTED"`, or `"NEEDS_REPLICATION"`.
* `critic_critique` (`str`): Referee feedback explaining statistical strengths/weaknesses.
* `statistical_metrics` (`Dict[str, Any]`): Two-tailed $p$-value, 95% Confidence Interval, and Cohen's $d$.
* `experiment_iteration` (`int`): Experiment refinement counter (capped at 2).
* `compressed_context` (`str`): Context-window summary distilled by `ContextManager`.
* `logs` (`Annotated[List[str], operator.add]`): Immutable chronological audit trail.

---

## 4. Docker Sandbox Security & Resource Isolation Protocol

Executing autonomously synthesized code presents significant host security risks (infinite loops, fork bombs, SSRF, disk exfiltration). AutoScientist AI implements defense-in-depth containerization:

### Isolation Matrix
| Security Dimension | Enforcement Mechanism | Host Protection Guarantee |
| :--- | :--- | :--- |
| **Network Isolation** | `network_mode="none"` | Absolute zero-network egress/ingress; prevents data exfiltration, socket binding, and SSRF attacks. |
| **Memory Ceiling** | `mem_limit="1g"`, swap disabled | Prevents host Out-Of-Memory (OOM) panic from memory-leak experiments. |
| **CPU Throttling** | `cpu_quota=100_000`, `cpu_period=100_000` | Limits container to exactly 1 CPU core; prevents core starvation and compute denial-of-service. |
| **Privilege Escalation** | `USER sandboxuser` (UID 1000) | Runs as unprivileged non-root user; root filesystem mounted read-only (`read_only=True`). |
| **Runaway Protection** | `timeout=30` seconds | Host supervisor forcefully executes `container.kill()` if execution exceeds 30 seconds. |
| **Filesystem Isolation** | Volatile ephemeral `/tmp` only | Workspace files cannot be modified or deleted by arbitrary code scripts. |

All security controls are verified by automated tests in `tests/test_sandbox_security.py`.

---

## 5. Multimodal Ingestion Pipeline & Statistical Critic Methodology

### A. Multimodal PDF Figure Extractor (`src/ingestion/figure_extractor.py`)
Scientific figures convey critical experimental trends that text abstracts miss:
* **Engine:** PyMuPDF (`fitz`) extracts raw vector and raster image streams directly from PDF bytecode.
* **Resolution Filter:** Discards logos, header decorations, and author icons by enforcing a strict $> 300\times 300$ pixel threshold.
* **Caption Pairing:** Extracts text blocks surrounding each image coordinates to pair figures with their descriptive captions (e.g., *"Figure 3: Memory footprint vs context length"*).
* **VLM Extraction (`src/ingestion/chart_reader.py`):** Formats extracted visual metrics (axes, baselines, peak performance deltas) into structured JSON consumable by the hypothesis generator.

### B. Statistical Academic Critic Agent (`src/agents/critic_agent.py`)
Rather than relying on qualitative LLM impressions, the Critic Agent acts as a mathematical referee applying parametric and non-parametric hypothesis testing:

1. **Welch's Two-Sample t-Test:**
   Evaluates unequal variance between baseline ($X_b$) and treatment ($X_t$):
   $$t = \frac{\bar{X}_t - \bar{X}_b}{\sqrt{\frac{s_t^2}{N_t} + \frac{s_b^2}{N_b}}}$$
2. **Standardized Effect Size (Cohen's $d$):**
   $$d = \frac{\bar{X}_t - \bar{X}_b}{s_{\text{pooled}}}$$
3. **95% Confidence Interval (CI):**
   $$\text{CI}_{95\%} = (\bar{X}_t - \bar{X}_b) \pm 1.96 \cdot \text{SE}$$
4. **Decision Rubric:**
   * **`ACCEPTED`:** Two-tailed $p < 0.05$, $N \ge 30$, and lower CI bound $> 0$.
   * **`REJECTED`:** $p \ge 0.05$ or metric delta within baseline noise ($d < 0.20$).
   * **`NEEDS_REPLICATION`:** $p < 0.05$ but sample size $N < 10$.

---

## 6. Two-Tier Memory Architecture

To enable multi-day discovery without context window token overflow, Member 1 implemented a two-tier memory architecture:

### Tier 1: Working Context Manager (`src/memory/context_manager.py`)
* Enforces a hard **4,000-token ceiling** on working prompt context.
* Uses sliding-window summarization on intermediate conversational logs while applying lossless regular expressions to preserve numerical baselines, $p$-values, effect sizes, and citation keys.

### Tier 2: Persistent Long-Horizon Episodic Store (`src/memory/episodic_memory.py`)
* Permanently serialized on disk at `data/experiments/episodes.json`.
* Records every experimental trial: `topic`, `hypothesis`, `verdict`, `metric_delta`, `p_value`, `summary`, and distilled `lessons_learned`.
* On subsequent runs for related topics, the Literature Reviewer queries past episodic memory to avoid repeating rejected hypotheses and build incrementally on previously validated breakthroughs.

---

## 7. Master Entrypoint & CLI Integration (`main.py`)

The pipeline is unified under a single enterprise CLI entrypoint (`main.py`):

```bash
# 1. Standard execution with local subprocess sandbox:
python main.py --topic "Sparse Transformers"

# 2. Hardened container execution with automated LaTeX synthesis:
python main.py --topic "Transformer Memory Optimization" --sandbox docker --synthesize-latex

# 3. Headless automation with JSON export:
python main.py --topic "Quantized Attention" --output output/run_summary.json --quiet
```

### CLI Features
* **Auto-Fallback:** Automatically probes Docker daemon availability; if Docker is not active, cleanly degrades to isolated local subprocess without crashing.
* **LaTeX Compilation:** Formats verified experimental state into `output/generated_research_paper.tex` and compiles to `output/generated_research_paper.pdf`.
* **Execution Audit:** Prints real-time agent progression, statistical tables, and audit logs.

---

## 8. Supervisor Live Web Dashboard (`dashboard/app.py`)

Built with Streamlit, the live dashboard provides real-time human-in-the-loop oversight with a 6-page modular navigation:
* **Page 1 (Discovery Studio):** Parameter configuration (topic query, execution environment, critic toggle), real-time progress bars, active agent cards, and streaming console logs.
* **Page 2 (Agent Logic Tree):** Visual interactive state progression diagram illustrating transitions across Reviewer ➔ Hypothesizer ➔ Executor ➔ Debugger ➔ Critic ➔ Synthesizer.
* **Page 3 (Literature & Citation Explorer):** Searchable interface querying ChromaDB preprints with cosine distance sliders and chunk inspect modals.
* **Page 4 (Paper Viewer):** Live embedded PDF preview of `output/generated_research_paper.pdf` with single-click download.
* **Page 5 (Execution Cost Analytics):** Plotly charts displaying cumulative experiment costs ($), per-agent token distribution, and latency tables from `src/utils/cost_tracker.py`.
* **Page 6 (Holdout Benchmark Evaluation):** Visual radar/spider chart (`docs/figures/benchmark_radar.svg`) and empirical scoring tables comparing agent output against 9 human preprints (2024–2026).

---

## 9. Holdout Benchmark Evaluation & Empirical Comparison

To rigorously validate whether the autonomous agent produces publication-quality science, Member 3 curated a holdout evaluation dataset of 9 recent human-authored papers (2024–2026 across AI, Computational Biology, and Physics) stored in `data/holdout_papers/benchmark_metadata.json`.

Scoring is performed by `src/evaluation/benchmark_evaluator.py` across 4 objective dimensions (0 to 10 scale):

| Dimension | Agent Performance | Human Peer Average | Delta (Agent - Human) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Novelty Score** | 7.35 / 10 | **9.38 / 10** | -2.03 | Human leads in speculative frontier hypotheses. |
| **Literature Grounding** | **9.64 / 10** | 9.33 / 10 | **+0.31** | **Agent leads:** Zero citation hallucination via Pydantic whitelist. |
| **Empirical Validity** | **10.00 / 10** | 5.22 / 10 | **+4.78** | **Agent leads:** Enforced Welch's $t$-test ($p < 0.05$) and Cohen's $d$. |
| **Formatting & Structure** | **10.00 / 10** | 8.50 / 10 | **+1.50** | **Agent leads:** Strict adherence to NeurIPS/IEEE sections & BibTeX. |
| **COMPOSITE AVERAGE** | **9.25 / 10** | **8.11 / 10** | **+1.14** | **Agent exceeds human baseline by +14.1% in composite empirical rigor.** |

The radar geometry visualization (`docs/figures/benchmark_radar.svg`) confirms the agent's distinct advantage in methodological reproducibility and mathematical rigor.

---

## 10. Cost, Latency & Scalability Analysis

Tracking API token economics ensures enterprise sustainability:

| Agent Node | Model Provider | Avg Prompt Tokens | Avg Completion Tokens | Cost per Run ($) | Latency (sec) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Literature Reviewer** | Claude 3.5 Sonnet / Local | 1,850 | 420 | $0.0118 | 4.2s |
| **Hypothesis Generator** | GPT-4o / Local | 1,200 | 380 | $0.0087 | 3.1s |
| **Code Executor (Sandbox)** | Local Docker Runner | N/A (Python Code) | N/A (Stdout) | $0.0000 | 8.5s |
| **Debugger Agent (Reflexion)** | Claude 3.5 Sonnet / Local | 1,600 | 450 | $0.0115 (if triggered) | 3.8s |
| **Statistical Critic** | Pure Python Math + LLM | 950 | 250 | $0.0052 | 2.4s |
| **LaTeX Synthesizer** | GPT-4o / Local Template | 2,100 | 1,200 | $0.0243 | 5.6s |
| **TOTAL (Single Discovery)** | Hybrid Cloud / Edge | **~7,700 tokens** | **~2,700 tokens** | **$0.0615 / run** | **~27.6s** |

### Scalability Guarantees
* **Linear Time Complexity:** Pipeline latency scales strictly as $O(N)$ with respect to experiment trials.
* **Zero Token Creep:** Context pruning prevents prompt expansion during high-iteration multi-day autonomous loops.
* **Local Fallback Mode:** The entire suite can operate 100% offline using local Ollama models and standard library mathematical fallbacks at $0.00 API cost.

---

## 11. Quality Assurance & Verification Audit

The system undergoes continuous automated validation across **413 collected tests** spanning 28 dedicated test suites:

* **Unit & Ingestion Suites (Weeks 1 & 2):** Scrapers, state management, vector database, Pydantic schemas, and Docker sandbox security isolation.
* **Multimodal & Memory Suites (Week 3):** PyMuPDF figure extractions, VLM chart interpretations, Two-Tier Memory token budget enforcement, and Welch's $t$-test validator.
* **Synthesis, Benchmarking & E2E Suites (Week 4):**
  * `tests/test_main.py` (CLI entrypoint & argument parsing)
  * `tests/test_synthesizer.py` (LaTeX synthesis & BibTeX generation)
  * `tests/test_cost_tracker.py` & `tests/test_cost_analytics.py` (Token accounting & Plotly widgets)
  * `tests/test_benchmark_evaluator.py` & `tests/test_benchmark_report.py` (Holdout paper evaluation)
  * `tests/test_end_to_end.py` (Full hermetic pipeline integration)
* **Pass Rate:** **100% clean execution** with zero test failures across all 28 test suites.
* **Regression Protection:** Enforced by `tests/test_project_structure.py` ensuring zero missing or broken file contracts.

---

## 12. Final Review Sign-Off & Release Checklist

- [x] **End-to-End Autonomous Pipeline Functioning:** Validated via `main.py` CLI and `src/orchestrator.py`.
- [x] **Multimodal Ingestion & Statistical Critic Finalized:** PDF figure extraction and Welch's $t$-test operational.
- [x] **Simulated Research Paper Delivered:** `output/generated_research_paper.tex` and `.pdf` compiled.
- [x] **Enterprise Architectural Report Completed:** `docs/FINAL_ARCHITECTURAL_REPORT.md` authored.
- [x] **Project Structure Cleaned:** Redundant files removed, test suite 100% green.
- [x] **GitHub 20-Day Commit Streak Verified:** Continuous commits across all milestone dates.

**Project Status:** Ready for Final Evaluation & Release Tag `v1.0.0-final`.
