# Telecom AI Researcher

An automated research pipeline that generates complete, peer-review-ready conference papers in telecommunications / machine learning — from a one-line topic description to a compiled PDF.

The system orchestrates a team of specialised LLM agents through a **6-stage deterministic FSM pipeline**, with human checkpoints at key milestones and a built-in quality-gate reviewer at every stage.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Sample Output](#sample-output)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Project Structure](#project-structure)
8. [Stage Reference](#stage-reference)
9. [Run Artifacts](#run-artifacts)

---

## Overview

```
telecom-researcher run "complex-valued diffusion denoiser for OFDM channel estimation"
```

From one command, the pipeline:

1. **Searches** arXiv and Semantic Scholar for 15+ relevant papers and synthesises a gap analysis
2. **Brainstorms** three candidate research ideas and selects the most promising one
3. **Formalises** the mathematical problem statement, system model, and proposed method
4. **Implements** working Python/PyTorch code, trains models, and produces numerical results (update: for the safety reason, this step needs human confirmation and executes in programing IDE such as pycharm/VS Code rather than automatically run in Claude `code_executor`)
5. **Analyses** results, generates IEEE-style figures, and writes a detailed analysis report
6. **Writes** a complete IEEE conference paper (LaTeX), compiles it with `tectonic`, and outputs a PDF

Total cost for a full run: **~\$14–20** (Claude Sonnet + Opus + GPT-4o).
Total time: **~2–4 hours** (dominated by model training in Stage 4).

---

## Pipeline Architecture

```
Topic String
     │
     ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Pipeline FSM Controller                          │
│                                                                     │
│  Stage 1          Stage 2          Stage 3          Stage 4        │
│  Literature  ──►  Ideation   ──►  Formulation ──►  Experiments    │
│  Review           [✓ human]                         [✓ human]      │
│                                                                     │
│  Stage 5          Stage 6                                          │
│  Analysis    ──►  Manuscript  ──►  PDF                             │
│                   [✓ human]                                        │
│                                                                     │
│  Each stage: Agent → Reviewer (0–1 score) → Gate (≥0.7 pass)      │
└────────────────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Stage | Model (default) | Role |
|-------|-------|-----------------|------|
| `LitReviewerAgent` | 1 | Claude Sonnet | arXiv/S2 search, gap analysis, BibTeX |
| `IdeatorAgent` | 2 | Claude Sonnet | Candidate ideas, novelty scoring |
| `FormulatorAgent` | 3 | GPT-4o | System model, math formulation |
| `ExperimenterAgent` | 4 | GPT-4o | Code generation, training, evaluation |
| `AnalystAgent` | 5 | Claude Sonnet | Figures, statistical analysis |
| `WriterAgent` | 6 | Claude Opus | Full IEEE LaTeX paper |
| `ReviewerAgent` | every stage | Claude Opus | Stage quality gating |
| `EditorAgent` | 6 | Claude Sonnet | LaTeX revision and compilation |

### Tools available to agents

- `ArxivSearchTool` — semantic search over arXiv
- `SemanticScholarSearchTool` — citation metadata, abstracts
- `CodeExecutorTool` — sandboxed Python execution (trains models, evaluates results)
- `FigureGeneratorTool` — matplotlib IEEE-style figure generation
- `LatexCompilerTool` — `tectonic` LaTeX compilation
- `ReadFileTool` / `WriteFileTool` — structured state I/O

---

## Sample Output

**Run:** `20260225_202351_complex-valued-diffusion-denoiser-with-p`
**Topic:** *Complex-Valued Diffusion Denoiser with Phase Interaction Layer for OFDM Channel Estimation*

### Paper Summary

The pipeline autonomously produced a complete 6-page IEEE ICMLCN 2026 submission:

- **Proposed method:** ComplexUNet — a U-Net denoiser with 5-channel conditioning (`[Re(hₜ), Im(hₜ), Re(ĥ_LS), Im(ĥ_LS), mₚ]`) and a novel **Quantum Phase Rotation Gate** output layer (Rz matrix parameterisation)
- **Inference:** DDIM with 200 deterministic steps; ablation shows DDPM+RePaint yields *positive* NMSE (catastrophic failure due to train–test distribution mismatch)
- **Key insight:** Direct 5-channel conditioning replaces RePaint pilot injection — essential for functional diffusion-based estimation

### Numerical Results (NMSE dB @ SNR)

| Method | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|--------|------|-------|-------|-------|-------|
| LS | −6.4 | −10.7 | −14.2 | −16.3 | −17.4 |
| MMSE | −9.2 | −13.2 | −18.0 | −23.0 | **−28.0** |
| DNN baseline | −7.7 | −11.5 | −15.0 | −17.4 | −18.6 |
| RealUNet (ablation) | −6.2 | −9.1 | −13.3 | −17.9 | −20.6 |
| **ComplexUNet (proposed)** | **−5.9** | **−9.3** | **−13.6** | **−17.3** | **−19.8** |

### Pipeline Metrics

| Stage | Score | Cost |
|-------|-------|------|
| 1 Literature Review | 0.88 ✓ | |
| 2 Ideation | 0.88 ✓ | |
| 3 Formulation | 0.85 ✓ | |
| 4 Experiments | 0.82 ✓ | |
| 5 Analysis | 0.87 ✓ | |
| 6 Manuscript | 0.90 ✓ | |
| **Total** | | **\$14.29** / 1.1M tokens |

### Run artifacts (committed)

```
runs/20260225_202351_complex-valued-diffusion-denoiser-with-p/state/
├── 01_literature_review/literature_review.json   # 15 papers, BibTeX, gap analysis
├── 02_ideation/ideation.json                      # 3 candidate ideas, selected idea
├── 03_formulation/formulation.json                # System model, math formulation
├── 04_experiments/
│   ├── experiments.json                           # Experiment plan + results summary
│   └── code/
│       ├── data_generation.py                     # OFDM channel dataset generator
│       ├── models.py                              # ComplexUNet + RealUNet architectures
│       ├── train_model.py                         # Training loop (warmup + cosine LR)
│       ├── evaluate.py                            # DDIM inference + NMSE/BER eval
│       ├── generate_figures.py                    # IEEE-style matplotlib figures
│       ├── results.json                           # Final NMSE/BER numbers
│       └── training_results.json                  # Per-epoch validation NMSE
├── 05_analysis/
│   ├── analysis.json                              # Analysis report
│   └── training_data.json                         # Training convergence data
└── 06_manuscript/
    ├── main.tex                                   # Full IEEE LaTeX paper (6 pages)
    └── references.bib                             # 15-entry bibliography
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/senyan0508/telecom-ai-researcher.git
cd telecom-ai-researcher

# Create and activate virtual environment (Python 3.12+)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux

# Install the package
pip install -e ".[dl]"       # includes PyTorch for experiment agents
# or: pip install -e .       # LLM pipeline only (no local training)

# Install LaTeX compiler (macOS)
brew install tectonic

# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."         # needed for formulation + experiments stages
```

---

## Usage

### Full pipeline run

```bash
telecom-researcher run "your research topic here"
```

Options:

```
--config PATH         Custom YAML config (default: config/default.yaml)
--start-stage INT     Resume from stage N (1–6); skips earlier stages
--run-id TEXT         Custom run directory name
--project-dir PATH    Project root (default: current directory)
--all-sonnet          Use Claude Sonnet for all agents (cheaper, for testing)
--no-human-checkpoint Skip interactive approval checkpoints
--verbose             Debug logging
```

### Resume a run from a specific stage

```bash
telecom-researcher run "topic" --start-stage 4 --run-id "20260225_202351_..."
```

### Check run status

```bash
telecom-researcher status runs/20260225_202351_complex-valued-diffusion-denoiser-with-p
```

### Cost report across all runs

```bash
telecom-researcher cost-report
```

---

## Configuration

The pipeline reads `config/default.yaml`. Key settings:

```yaml
# Quality gate threshold (0–1); stages below this score are retried
quality:
  gate_threshold: 0.7
  max_revisions_per_stage: 3

# Per-stage cost budgets (USD)
cost_budget_total: 80.0
cost_budget_per_stage:
  experiments: 20.0
  manuscript: 20.0

# Human approval checkpoints
human_checkpoint_after_ideation: true
human_checkpoint_after_experiments: true
human_checkpoint_after_manuscript: true

# Model assignments per stage
models:
  literature_review:
    primary:
      provider: anthropic
      model_id: anthropic/claude-sonnet-4-20250514
  manuscript:
    primary:
      provider: anthropic
      model_id: anthropic/claude-opus-4-20250514
```

See `config/models.yaml` for the full multi-model Phase B configuration (Claude + GPT-4o).

---

## Project Structure

```
telecom-ai-researcher/
├── src/telecom_researcher/
│   ├── main.py              # CLI (click): run / status / cost-report
│   ├── pipeline.py          # FSM controller, stage registry
│   ├── state.py             # Pydantic state models, load/save helpers
│   ├── config.py            # PipelineConfig, model loading
│   ├── context.py           # Stage prompt context builders
│   ├── agents/
│   │   ├── base.py          # BaseAgent (LLM call loop, tool dispatch)
│   │   ├── lit_reviewer.py  # Stage 1
│   │   ├── ideator.py       # Stage 2
│   │   ├── formulator.py    # Stage 3
│   │   ├── experimenter.py  # Stage 4
│   │   ├── analyst.py       # Stage 5
│   │   ├── writer.py        # Stage 6
│   │   ├── reviewer.py      # Per-stage quality reviewer
│   │   └── editor.py        # LaTeX revision agent
│   ├── llm/
│   │   └── client.py        # LiteLLM wrapper, cost tracking
│   ├── tools/
│   │   ├── arxiv_search.py
│   │   ├── semantic_scholar.py
│   │   ├── code_executor.py
│   │   ├── figure_generator.py
│   │   └── latex_compiler.py
│   └── templates/
│       ├── ieee_conference.tex   # LaTeX scaffold
│       └── figure_style.py      # IEEE rcParams
├── config/
│   ├── default.yaml         # Default pipeline + model config
│   └── models.yaml          # Phase B multi-model config
├── runs/                    # Output directory (gitignored except reference run)
│   └── .gitkeep
├── tests/
├── pyproject.toml
└── README.md
```

---

## Stage Reference

| # | Name | Agent | Input | Output |
|---|------|-------|-------|--------|
| 1 | Literature Review | `LitReviewerAgent` | Topic string | `literature_review.json` — 15+ papers, BibTeX, 5 research gaps |
| 2 | Ideation | `IdeatorAgent` | Literature artifact | `ideation.json` — 3 candidate ideas, selected idea with novelty/feasibility scores |
| 3 | Formulation | `FormulatorAgent` | Ideation artifact | `formulation.json` — system model, problem statement, proposed method math |
| 4 | Experiments | `ExperimenterAgent` | Formulation artifact | `experiments.json` + `code/` — full Python implementation, trained models, result tables |
| 5 | Analysis | `AnalystAgent` | Experiments artifact | `analysis.json` — statistical analysis, figures, key findings |
| 6 | Manuscript | `WriterAgent` + `EditorAgent` | All artifacts | `main.tex` + `main.pdf` — complete IEEE conference paper |

Each stage is followed by a `ReviewerAgent` pass. If the score is below `gate_threshold` (default 0.7), the agent is given feedback and re-runs (up to `max_revisions_per_stage` times, default 3).

---

## Run Artifacts

Each run creates a directory under `runs/<run_id>/state/`:

```
state/
├── pipeline_state.json        # Overall FSM state (stage, cost, tokens, review scores)
├── 01_literature_review/      # Stage 1 output
├── 02_ideation/               # Stage 2 output
├── 03_formulation/            # Stage 3 output
├── 04_experiments/
│   ├── experiments.json       # High-level experiment plan
│   └── code/                  # Auto-generated Python codebase
│       ├── data_generation.py
│       ├── models.py
│       ├── train_model.py
│       ├── evaluate.py
│       ├── generate_figures.py
│       └── results.json
├── 05_analysis/               # Stage 5 output + figures
└── 06_manuscript/             # Final paper: main.tex, references.bib, main.pdf
```

---

*Generated by AI Agents · Digital Future Institute, Khalifa University*
