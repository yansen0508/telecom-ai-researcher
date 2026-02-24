"""Pipeline state and artifact dataclasses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# --- Stage Metadata ---

class StageMetadata(BaseModel):
    """Metadata for any pipeline stage execution."""

    stage_name: str
    model_used: str
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    revision_number: int = 0

    def mark_complete(self) -> None:
        self.completed_at = datetime.now(timezone.utc).isoformat()


# --- Stage 1: Literature Review ---

class PaperSummary(BaseModel):
    """Summary of a single research paper."""

    title: str
    authors: list[str]
    year: int
    venue: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    key_methods: list[str] = Field(default_factory=list)
    key_results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0  # 0-1


class ResearchGap(BaseModel):
    """An identified gap in the literature."""

    description: str
    supporting_evidence: list[str] = Field(default_factory=list)
    opportunity_score: float = 0.0  # 0-1


class LiteratureArtifact(BaseModel):
    """Output of Stage 1: Literature Review."""

    topic: str
    search_queries: list[str] = Field(default_factory=list)
    papers: list[PaperSummary] = Field(default_factory=list)
    gap_analysis: list[ResearchGap] = Field(default_factory=list)
    taxonomy: dict[str, list[str]] = Field(default_factory=dict)  # category -> paper titles
    bibtex_entries: list[str] = Field(default_factory=list)
    metadata: StageMetadata | None = None

    def to_summary(self) -> str:
        """Compressed summary for context passing to later stages."""
        lines = [f"Topic: {self.topic}", f"Papers reviewed: {len(self.papers)}", ""]
        lines.append("Key Gaps:")
        for gap in self.gap_analysis[:5]:
            lines.append(f"  - {gap.description} (score: {gap.opportunity_score:.2f})")
        lines.append("")
        lines.append("Top Papers:")
        for paper in sorted(self.papers, key=lambda p: p.relevance_score, reverse=True)[:10]:
            lines.append(f"  - [{paper.year}] {paper.title}")
        return "\n".join(lines)


# --- Stage 2: Ideation ---

class ResearchIdea(BaseModel):
    """A candidate research idea."""

    title: str
    abstract_sketch: str = ""
    approach: str = ""
    expected_contribution: str = ""
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    significance_score: float = 0.0


class IdeationArtifact(BaseModel):
    """Output of Stage 2: Ideation."""

    candidate_ideas: list[ResearchIdea] = Field(default_factory=list)
    selected_idea: ResearchIdea | None = None
    novelty_assessment: str = ""
    feasibility_assessment: str = ""
    scope_definition: str = ""
    metadata: StageMetadata | None = None

    def to_summary(self) -> str:
        if not self.selected_idea:
            return "No idea selected yet."
        idea = self.selected_idea
        return (
            f"Selected Idea: {idea.title}\n"
            f"Approach: {idea.approach}\n"
            f"Contribution: {idea.expected_contribution}\n"
            f"Scope: {self.scope_definition}"
        )


# --- Stage 3: Problem Formulation ---

class Metric(BaseModel):
    """An evaluation metric."""

    name: str
    formula: str = ""  # LaTeX formula
    description: str = ""


class FormulationArtifact(BaseModel):
    """Output of Stage 3: Problem Formulation."""

    system_model: str = ""  # LaTeX-formatted
    notation_table: dict[str, str] = Field(default_factory=dict)  # symbol -> meaning
    problem_statement: str = ""  # LaTeX-formatted optimization/learning problem
    theoretical_analysis: str = ""
    evaluation_metrics: list[Metric] = Field(default_factory=list)
    baseline_methods: list[str] = Field(default_factory=list)
    latex_equations: list[str] = Field(default_factory=list)
    metadata: StageMetadata | None = None

    def to_summary(self) -> str:
        lines = [
            "Problem Formulation:",
            f"  System model defined: {'yes' if self.system_model else 'no'}",
            f"  Notation symbols: {len(self.notation_table)}",
            f"  Metrics: {', '.join(m.name for m in self.evaluation_metrics)}",
            f"  Baselines: {', '.join(self.baseline_methods)}",
        ]
        return "\n".join(lines)


# --- Stage 4: Experiments ---

class ExperimentConfig(BaseModel):
    """Configuration for a simulation experiment."""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class DebugIteration(BaseModel):
    """Record of a code debug cycle."""

    error_message: str
    fix_description: str
    success: bool


class ExperimentArtifact(BaseModel):
    """Output of Stage 4: Experiments."""

    experiment_configs: list[ExperimentConfig] = Field(default_factory=list)
    simulation_code: str = ""
    code_path: str = ""
    raw_results: dict[str, Any] = Field(default_factory=dict)
    result_tables: list[dict[str, Any]] = Field(default_factory=list)
    execution_logs: list[str] = Field(default_factory=list)
    debug_history: list[DebugIteration] = Field(default_factory=list)
    metadata: StageMetadata | None = None

    def to_summary(self) -> str:
        lines = [
            "Experiments:",
            f"  Configs: {len(self.experiment_configs)}",
            f"  Code generated: {'yes' if self.simulation_code else 'no'}",
            f"  Debug iterations: {len(self.debug_history)}",
            f"  Results tables: {len(self.result_tables)}",
        ]
        return "\n".join(lines)


# --- Stage 5: Analysis ---

class FigureSpec(BaseModel):
    """Specification for a generated figure."""

    filename: str
    caption: str = ""
    latex_include: str = ""  # \includegraphics{} command


class TableSpec(BaseModel):
    """Specification for a generated table."""

    latex_code: str
    caption: str = ""


class AnalysisArtifact(BaseModel):
    """Output of Stage 5: Analysis."""

    figures: list[FigureSpec] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    analysis_narrative: str = ""
    figure_generation_code: str = ""
    metadata: StageMetadata | None = None

    def to_summary(self) -> str:
        lines = [
            "Analysis:",
            f"  Figures: {len(self.figures)}",
            f"  Tables: {len(self.tables)}",
            "  Key Findings:",
        ]
        for finding in self.key_findings[:5]:
            lines.append(f"    - {finding}")
        return "\n".join(lines)


# --- Stage 6: Manuscript ---

class ManuscriptArtifact(BaseModel):
    """Output of Stage 6: Manuscript Writing."""

    latex_source: str = ""
    bibtex_source: str = ""
    figures_dir: str = ""
    compiled_pdf_path: str = ""
    section_drafts: dict[str, str] = Field(default_factory=dict)  # section_name -> latex
    editor_comments: list[str] = Field(default_factory=list)
    metadata: StageMetadata | None = None

    def to_summary(self) -> str:
        sections = list(self.section_drafts.keys())
        return (
            f"Manuscript: {len(sections)} sections written\n"
            f"Sections: {', '.join(sections)}\n"
            f"PDF compiled: {'yes' if self.compiled_pdf_path else 'no'}"
        )


# --- Review ---

class ReviewResult(BaseModel):
    """Result of a gate review."""

    stage: int
    score: float = 0.0  # 0-1
    passed: bool = False
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    feedback: str = ""
    critical_issues: list[str] = Field(default_factory=list)


class PeerReviewResult(BaseModel):
    """Result of the final mock peer review."""

    novelty_score: int = 0  # 1-5
    soundness_score: int = 0
    significance_score: int = 0
    presentation_score: int = 0
    experimental_score: int = 0
    overall_recommendation: str = ""  # Strong Accept / Accept / ... / Reject
    detailed_comments: str = ""
    required_revisions: list[str] = Field(default_factory=list)


# --- Pipeline State ---

class PipelineState(BaseModel):
    """Overall pipeline execution state. Persisted to disk for checkpoint/resume."""

    run_id: str
    topic: str
    current_stage: int = 0  # 0 = not started, 1-6 = stage number
    stage_status: str = "pending"  # pending | running | review | complete | failed | paused
    artifacts: dict[int, str] = Field(default_factory=dict)  # stage_num -> artifact file path
    review_results: dict[int, ReviewResult] = Field(default_factory=dict)
    revision_count: dict[int, int] = Field(default_factory=dict)  # stage_num -> count
    peer_review: PeerReviewResult | None = None
    total_cost: float = 0.0
    total_tokens: int = 0
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def update(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


# --- Persistence helpers ---

def save_state(state: PipelineState, run_dir: Path) -> None:
    """Save pipeline state to disk."""
    state.update()
    state_file = run_dir / "state" / "pipeline_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(state.model_dump_json(indent=2))


def load_state(run_dir: Path) -> PipelineState:
    """Load pipeline state from disk."""
    state_file = run_dir / "state" / "pipeline_state.json"
    return PipelineState.model_validate_json(state_file.read_text())


def save_artifact(artifact: BaseModel, run_dir: Path, stage_num: int, stage_name: str) -> str:
    """Save a stage artifact to disk. Returns the relative path."""
    artifact_dir = run_dir / "state" / f"{stage_num:02d}_{stage_name}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = artifact_dir / f"{stage_name}.json"
    artifact_file.write_text(artifact.model_dump_json(indent=2))
    return str(artifact_file.relative_to(run_dir))


def load_artifact(run_dir: Path, artifact_path: str, artifact_class: type[BaseModel]) -> BaseModel:
    """Load a stage artifact from disk."""
    full_path = run_dir / artifact_path
    return artifact_class.model_validate_json(full_path.read_text())
