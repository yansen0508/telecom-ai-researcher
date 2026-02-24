"""Context builder for assembling LLM prompts with progressive summarization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from telecom_researcher.state import (
    AnalysisArtifact,
    ExperimentArtifact,
    FormulationArtifact,
    IdeationArtifact,
    LiteratureArtifact,
    ManuscriptArtifact,
    PipelineState,
    load_artifact,
)

# Mapping from stage number to artifact class
STAGE_ARTIFACT_MAP: dict[int, tuple[str, type[BaseModel]]] = {
    1: ("literature_review", LiteratureArtifact),
    2: ("ideation", IdeationArtifact),
    3: ("formulation", FormulationArtifact),
    4: ("experiments", ExperimentArtifact),
    5: ("analysis", AnalysisArtifact),
    6: ("manuscript", ManuscriptArtifact),
}

STAGE_NAMES = {
    1: "Literature Review",
    2: "Ideation",
    3: "Problem Formulation",
    4: "Experiments",
    5: "Analysis",
    6: "Manuscript Writing",
}


def build_stage_context(
    stage_num: int,
    state: PipelineState,
    run_dir: Path,
) -> str:
    """Build context string for a stage, respecting token budgets.

    Strategy:
    - Always include: topic
    - Full artifact from the immediately preceding stage
    - Compressed summaries of earlier stages
    - Stage-specific full inclusions (e.g., Writer needs full formulation for equations)
    """
    parts: list[str] = []

    # Always include the research topic
    parts.append(f"# Research Topic\n\n{state.topic}")

    # Include compressed summaries of earlier stages (except the immediately preceding one)
    for earlier_stage in range(1, stage_num - 1):
        if earlier_stage in state.artifacts:
            _, artifact_cls = STAGE_ARTIFACT_MAP[earlier_stage]
            artifact = load_artifact(run_dir, state.artifacts[earlier_stage], artifact_cls)
            if hasattr(artifact, "to_summary"):
                parts.append(
                    f"## Stage {earlier_stage} ({STAGE_NAMES[earlier_stage]}) - Summary\n\n"
                    f"{artifact.to_summary()}"
                )

    # Include full artifact from immediately preceding stage
    prev_stage = stage_num - 1
    if prev_stage >= 1 and prev_stage in state.artifacts:
        _, artifact_cls = STAGE_ARTIFACT_MAP[prev_stage]
        artifact = load_artifact(run_dir, state.artifacts[prev_stage], artifact_cls)
        parts.append(
            f"## Stage {prev_stage} ({STAGE_NAMES[prev_stage]}) - Full Output\n\n"
            f"{artifact.model_dump_json(indent=2)}"
        )

    # Stage-specific full inclusions
    if stage_num == 6:
        # Writer needs full formulation for equations and notation
        if 3 in state.artifacts:
            form_artifact = load_artifact(
                run_dir, state.artifacts[3], FormulationArtifact
            )
            parts.append(
                "## Problem Formulation (Full - for equations and notation)\n\n"
                f"{form_artifact.model_dump_json(indent=2)}"
            )
        # Writer also needs full analysis for figures and tables
        if 5 in state.artifacts:
            analysis_artifact = load_artifact(
                run_dir, state.artifacts[5], AnalysisArtifact
            )
            parts.append(
                "## Analysis (Full - for figures and tables)\n\n"
                f"{analysis_artifact.model_dump_json(indent=2)}"
            )

    if stage_num == 5:
        # Analyst needs the formulation to interpret results in context
        if 3 in state.artifacts:
            form_artifact = load_artifact(
                run_dir, state.artifacts[3], FormulationArtifact
            )
            parts.append(
                "## Problem Formulation (for context)\n\n"
                f"{form_artifact.model_dump_json(indent=2)}"
            )

    return "\n\n---\n\n".join(parts)


def build_review_context(
    stage_num: int,
    state: PipelineState,
    run_dir: Path,
    artifact_json: str,
) -> str:
    """Build context for the reviewer agent."""
    parts = [
        f"# Review Task: Stage {stage_num} ({STAGE_NAMES[stage_num]})",
        f"## Research Topic\n\n{state.topic}",
        f"## Output to Review\n\n{artifact_json}",
    ]

    # Include relevant prior context for cross-checking
    if stage_num >= 2 and 1 in state.artifacts:
        lit = load_artifact(run_dir, state.artifacts[1], LiteratureArtifact)
        parts.append(f"## Literature Review Summary (for novelty check)\n\n{lit.to_summary()}")

    if stage_num >= 4 and 3 in state.artifacts:
        form = load_artifact(run_dir, state.artifacts[3], FormulationArtifact)
        parts.append(f"## Formulation Summary (for consistency check)\n\n{form.to_summary()}")

    return "\n\n---\n\n".join(parts)
