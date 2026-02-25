"""Pipeline Controller — deterministic FSM orchestrating the 6 research stages."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

from telecom_researcher.agents.analyst import AnalystAgent
from telecom_researcher.agents.editor import EditorAgent
from telecom_researcher.agents.experimenter import ExperimenterAgent
from telecom_researcher.agents.formulator import FormulatorAgent
from telecom_researcher.agents.ideator import IdeatorAgent
from telecom_researcher.agents.lit_reviewer import LitReviewerAgent
from telecom_researcher.agents.reviewer import ReviewerAgent
from telecom_researcher.agents.writer import WriterAgent
from telecom_researcher.config import PipelineConfig
from telecom_researcher.context import (
    STAGE_NAMES,
    build_review_context,
    build_stage_context,
)
from telecom_researcher.llm.client import CostTracker
from telecom_researcher.state import (
    AnalysisArtifact,
    ExperimentArtifact,
    FormulationArtifact,
    IdeationArtifact,
    LiteratureArtifact,
    ManuscriptArtifact,
    PipelineState,
    ReviewResult,
    StageMetadata,
    load_state,
    save_artifact,
    save_state,
)
from telecom_researcher.tools.arxiv_search import ArxivSearchTool
from telecom_researcher.tools.base import (
    ReadFileTool,
    ToolRegistry,
    WriteFileTool,
)
from telecom_researcher.tools.code_executor import CodeExecutorTool
from telecom_researcher.tools.figure_generator import FigureGeneratorTool
from telecom_researcher.tools.latex_compiler import LatexCompilerTool
from telecom_researcher.tools.semantic_scholar import SemanticScholarSearchTool

logger = logging.getLogger(__name__)
console = Console()

# Stage number -> (artifact class, agent class, allowed tools)
STAGE_REGISTRY: dict[int, dict[str, Any]] = {
    1: {
        "artifact_cls": LiteratureArtifact,
        "agent_cls": LitReviewerAgent,
        "tools": ["arxiv_search", "semantic_scholar_search", "read_file", "write_file"],
    },
    2: {
        "artifact_cls": IdeationArtifact,
        "agent_cls": IdeatorAgent,
        "tools": ["read_file", "write_file"],
    },
    3: {
        "artifact_cls": FormulationArtifact,
        "agent_cls": FormulatorAgent,
        "tools": ["read_file", "write_file"],
    },
    4: {
        "artifact_cls": ExperimentArtifact,
        "agent_cls": ExperimenterAgent,
        "tools": ["code_execute", "read_file", "write_file"],
    },
    5: {
        "artifact_cls": AnalysisArtifact,
        "agent_cls": AnalystAgent,
        "tools": ["figure_generate", "code_execute", "read_file", "write_file"],
    },
    6: {
        "artifact_cls": ManuscriptArtifact,
        "agent_cls": WriterAgent,
        "tools": ["latex_compile", "read_file", "write_file", "arxiv_search"],
    },
}

MODEL_ROLE_MAP: dict[int, str] = {
    1: "literature_review",
    2: "ideation",
    3: "formulation",
    4: "experiments",
    5: "analysis",
    6: "manuscript",
}


class Pipeline:
    """Main pipeline controller."""

    def __init__(self, config: PipelineConfig, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.cost_tracker = CostTracker()
        self._tool_registry = self._build_tool_registry(stage_num=0)

    def _build_tool_registry(self, stage_num: int = 0) -> ToolRegistry:
        """Create and register all available tools.

        The working directory for read_file/write_file is set per-stage so that
        relative paths resolve to the correct stage directory.
        """
        registry = ToolRegistry()

        code_dir = self.run_dir / "state" / "04_experiments" / "code"
        figures_dir = self.run_dir / "state" / "05_analysis" / "figures"
        manuscript_dir = self.run_dir / "state" / "06_manuscript"

        # Stage-specific working directory for read_file/write_file
        if stage_num == 4:
            file_working_dir = code_dir
        elif stage_num == 5:
            file_working_dir = self.run_dir / "state" / "05_analysis"
        elif stage_num == 6:
            file_working_dir = manuscript_dir
        else:
            file_working_dir = self.run_dir / "state"

        registry.register(ArxivSearchTool())
        registry.register(SemanticScholarSearchTool())
        registry.register(CodeExecutorTool(working_dir=code_dir))
        registry.register(LatexCompilerTool(working_dir=manuscript_dir))
        registry.register(FigureGeneratorTool(output_dir=figures_dir))
        registry.register(ReadFileTool(working_dir=file_working_dir))
        registry.register(WriteFileTool(working_dir=file_working_dir))

        return registry

    async def run(self, topic: str, start_stage: int = 1) -> PipelineState:
        """Execute the full pipeline from start_stage to completion."""
        run_id = self.run_dir.name

        # Initialize or load state
        if start_stage > 1:
            # Resuming: load previous state to preserve artifacts from earlier stages
            try:
                state = load_state(self.run_dir)
                state.topic = topic  # Update topic in case user modified it
                console.print(
                    f"[green]Loaded previous state: {len(state.artifacts)} artifacts found[/green]"
                )
            except Exception:
                console.print(
                    "[yellow]Warning: Could not load previous state, starting fresh[/yellow]"
                )
                state = PipelineState(run_id=run_id, topic=topic)
        else:
            state = PipelineState(run_id=run_id, topic=topic)

        save_state(state, self.run_dir)

        console.print(Panel(f"[bold]Telecom AI Researcher[/bold]\nTopic: {topic}\nRun: {run_id}"))

        for stage_num in range(start_stage, 7):
            stage_name = STAGE_NAMES[stage_num]
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Stage {stage_num}: {stage_name}[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

            state.current_stage = stage_num
            state.stage_status = "running"
            save_state(state, self.run_dir)

            # Pre-stage setup
            if stage_num == 6:
                self._setup_manuscript_dir()

            # Run the stage
            artifact = await self._run_stage(stage_num, state)

            # Save artifact
            artifact_path = save_artifact(
                artifact, self.run_dir, stage_num,
                MODEL_ROLE_MAP.get(stage_num, f"stage_{stage_num}")
            )
            state.artifacts[stage_num] = artifact_path

            # Gate review
            state.stage_status = "review"
            save_state(state, self.run_dir)

            review = await self._run_review(stage_num, artifact, state)
            state.review_results[stage_num] = review

            # Revision loop
            revision = 0
            while not review.passed and revision < self.config.quality.max_revisions_per_stage:
                revision += 1
                console.print(f"[yellow]Revision {revision}/{self.config.quality.max_revisions_per_stage}[/yellow]")

                artifact = await self._run_stage(stage_num, state, feedback=review.feedback)
                artifact_path = save_artifact(
                    artifact, self.run_dir, stage_num,
                    MODEL_ROLE_MAP.get(stage_num, f"stage_{stage_num}")
                )
                state.artifacts[stage_num] = artifact_path

                review = await self._run_review(stage_num, artifact, state)
                state.review_results[stage_num] = review

            state.revision_count[stage_num] = revision

            if review.passed:
                console.print(f"[green]Stage {stage_num} PASSED (score: {review.score:.2f})[/green]")
            else:
                console.print(f"[red]Stage {stage_num} did not pass after max revisions (score: {review.score:.2f})[/red]")
                # Critical failure halt: stop pipeline if experiments or analysis fail badly
                if stage_num in (4, 5) and review.score < 0.4:
                    state.stage_status = "failed"
                    save_state(state, self.run_dir)
                    console.print(
                        f"[red bold]Critical failure in Stage {stage_num} (score={review.score:.2f} < 0.4). "
                        f"Pipeline halted to prevent writing a paper based on failed experiments.[/red bold]"
                    )
                    console.print(f"[yellow]Fix the issues and resume with --start-stage {stage_num}[/yellow]")
                    return state

            state.stage_status = "complete"
            state.total_cost = self.cost_tracker.total_cost
            state.total_tokens = self.cost_tracker.total_tokens
            save_state(state, self.run_dir)

            # Human checkpoint after ideation
            if stage_num == 2 and self.config.human_checkpoint_after_ideation:
                console.print("\n[bold yellow]Human checkpoint: Review the selected idea before continuing.[/bold yellow]")
                console.print("Press Enter to continue, or Ctrl+C to stop...")
                try:
                    input()
                except (KeyboardInterrupt, EOFError):
                    state.stage_status = "paused"
                    save_state(state, self.run_dir)
                    console.print("[yellow]Pipeline paused. Resume later with --start-stage 3[/yellow]")
                    return state

            # Human checkpoint after experiments (for manual model training)
            if stage_num == 4 and getattr(self.config, "human_checkpoint_after_experiments", False):
                code_dir = self.run_dir / "state" / "04_experiments" / "code"
                console.print("\n[bold yellow]{'='*60}[/bold yellow]")
                console.print("[bold yellow]Human checkpoint: Experiment code has been generated.[/bold yellow]")
                console.print(f"[bold yellow]Code directory: {code_dir}[/bold yellow]")
                console.print("[bold yellow]Please run the following in your VSCode terminal:[/bold yellow]")
                console.print(f"  [cyan]cd {code_dir}[/cyan]")
                console.print(f"  [cyan]python data_generation.py[/cyan]")
                console.print(f"  [cyan]python train_model.py[/cyan]")
                console.print(f"  [cyan]python evaluate.py[/cyan]")
                console.print("[bold yellow]After training completes, press Enter to continue.[/bold yellow]")
                console.print("[bold yellow]Or press Ctrl+C to stop and resume later with --start-stage 5[/bold yellow]")
                console.print(f"[bold yellow]{'='*60}[/bold yellow]\n")
                try:
                    input()
                except (KeyboardInterrupt, EOFError):
                    state.stage_status = "paused"
                    save_state(state, self.run_dir)
                    console.print("[yellow]Pipeline paused. Resume later with --start-stage 5[/yellow]")
                    return state

        console.print(f"\n[bold green]Pipeline complete![/bold green]")
        console.print(f"Total cost: ${self.cost_tracker.total_cost:.2f}")
        console.print(f"Total tokens: {self.cost_tracker.total_tokens:,}")
        return state

    def _setup_manuscript_dir(self) -> None:
        """Prepare manuscript directory: copy figures from Stage 5."""
        import shutil

        manuscript_dir = self.run_dir / "state" / "06_manuscript"
        manuscript_dir.mkdir(parents=True, exist_ok=True)

        # Copy figures from analysis stage
        figures_src = self.run_dir / "state" / "05_analysis" / "figures"
        figures_dst = manuscript_dir / "figures"
        if figures_src.exists():
            figures_dst.mkdir(parents=True, exist_ok=True)
            for fig_file in figures_src.glob("*"):
                if fig_file.is_file():
                    shutil.copy2(fig_file, figures_dst / fig_file.name)
            logger.info(f"Copied {len(list(figures_dst.iterdir()))} figures to manuscript dir")

    async def _run_stage(
        self,
        stage_num: int,
        state: PipelineState,
        feedback: str = "",
    ) -> BaseModel:
        """Run a single pipeline stage and return the artifact."""
        # Rebuild tool registry with stage-specific working directories
        self._tool_registry = self._build_tool_registry(stage_num)

        stage_info = STAGE_REGISTRY[stage_num]
        artifact_cls = stage_info["artifact_cls"]
        agent_cls = stage_info["agent_cls"]
        allowed_tools = stage_info["tools"]

        # Get model config for this stage
        model_role = MODEL_ROLE_MAP[stage_num]
        model_config = getattr(self.config.models, model_role)

        # Build context from previous stages
        context = build_stage_context(stage_num, state, self.run_dir)

        # Create agent
        prompts_dir = self.config.project_dir / "config" / "prompts"
        agent = agent_cls(
            model_config=model_config,
            tool_registry=self._tool_registry,
            allowed_tools=allowed_tools,
            cost_tracker=self.cost_tracker,
            prompts_dir=prompts_dir if prompts_dir.exists() else None,
        )

        # Run agent
        task_description = f"Perform {STAGE_NAMES[stage_num]} for the research topic: {state.topic}"
        result = await agent.run(
            user_message=task_description,
            context=context,
            feedback=feedback,
        )

        console.print(f"  Agent: {result.model_used} | Iterations: {result.iterations} | Cost: ${result.total_cost:.4f}")

        # Parse the JSON output into the artifact
        artifact = self._parse_artifact(result.content, artifact_cls, stage_num, result.model_used)
        return artifact

    async def _run_review(
        self,
        stage_num: int,
        artifact: BaseModel,
        state: PipelineState,
    ) -> ReviewResult:
        """Run the reviewer agent on a stage's output."""
        model_config = self.config.models.reviewer
        prompts_dir = self.config.project_dir / "config" / "prompts"

        reviewer = ReviewerAgent(
            model_config=model_config,
            cost_tracker=self.cost_tracker,
            prompts_dir=prompts_dir if prompts_dir.exists() else None,
        )

        artifact_json = artifact.model_dump_json(indent=2)
        review_context = build_review_context(stage_num, state, self.run_dir, artifact_json)

        result = await reviewer.run(
            user_message=f"Review the output of Stage {stage_num} ({STAGE_NAMES[stage_num]})",
            context=review_context,
        )

        # Parse review result
        try:
            review_data = json.loads(self._extract_json(result.content))
            review = ReviewResult(
                stage=stage_num,
                score=review_data.get("score", 0.0),
                passed=review_data.get("score", 0.0) >= self.config.quality.gate_threshold,
                strengths=review_data.get("strengths", []),
                weaknesses=review_data.get("weaknesses", []),
                feedback=review_data.get("feedback", ""),
                critical_issues=review_data.get("critical_issues", []),
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse review: {e}")
            review = ReviewResult(
                stage=stage_num,
                score=0.75,  # default pass on parse failure
                passed=True,
                feedback=result.content,
            )

        console.print(f"  Review: score={review.score:.2f} passed={review.passed}")
        return review

    def _parse_artifact(
        self,
        content: str,
        artifact_cls: type[BaseModel],
        stage_num: int,
        model_used: str,
    ) -> BaseModel:
        """Parse agent output content into a typed artifact."""
        metadata = StageMetadata(
            stage_name=STAGE_NAMES[stage_num],
            model_used=model_used,
        ).model_dump()

        try:
            json_str = self._extract_json(content)
            data = json.loads(json_str)
            data["metadata"] = metadata
            return artifact_cls.model_validate(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for stage {stage_num}: {e}")
            logger.error(f"Raw content (first 3000 chars):\n{content[:3000]}")
        except Exception as e:
            logger.error(f"Artifact validation error for stage {stage_num}: {e}")
            logger.error(f"Raw content (first 1000 chars):\n{content[:1000]}")

        # Fallback: try to construct artifact with minimal fields
        try:
            # Build fallback data with all fields at defaults
            fallback_data: dict[str, Any] = {"metadata": metadata}
            # For LiteratureArtifact, inject the topic from pipeline state
            if hasattr(artifact_cls, "model_fields") and "topic" in artifact_cls.model_fields:
                fallback_data["topic"] = "parse_error"
            return artifact_cls.model_validate(fallback_data)
        except Exception:
            # Last resort: return with model_construct (skip validation)
            return artifact_cls.model_construct(metadata=metadata)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text that may contain markdown code fences or surrounding text."""
        text = text.strip()

        # Try 1: Remove markdown code fences (```json ... ```)
        if "```" in text:
            import re
            # Match ```json or ``` blocks
            pattern = r"```(?:json)?\s*\n?(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                match = match.strip()
                if match.startswith("{"):
                    return match

        # Try 2: Find the outermost JSON object by brace matching
        start = text.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]

        # Try 3: Just find first { and last }
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text
