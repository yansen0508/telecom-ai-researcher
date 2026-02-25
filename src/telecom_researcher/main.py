"""CLI entry point for telecom-ai-researcher."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from telecom_researcher.config import PipelineConfig, load_config, make_all_sonnet_config

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("arxiv").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@click.group()
def cli() -> None:
    """Telecom AI Researcher - Automated research paper generation pipeline."""
    pass


@cli.command()
@click.argument("topic")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--run-id", default=None, help="Custom run ID (default: auto-generated)")
@click.option("--start-stage", default=1, type=int, help="Start from this stage (1-6)")
@click.option("--project-dir", type=click.Path(path_type=Path), default=None)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--no-human-checkpoint", is_flag=True, default=False, help="Skip human checkpoints")
@click.option("--all-sonnet", is_flag=True, default=False, help="Use Sonnet for all agents (cheaper testing)")
def run(
    topic: str,
    config_path: Path | None,
    run_id: str | None,
    start_stage: int,
    project_dir: Path | None,
    verbose: bool,
    no_human_checkpoint: bool,
    all_sonnet: bool,
) -> None:
    """Run the full research pipeline for a given TOPIC."""
    setup_logging(verbose)

    # Load config — auto-discover default.yaml if --config not specified
    if config_path is None:
        default_config = (project_dir or Path.cwd()) / "config" / "default.yaml"
        if default_config.exists():
            config_path = default_config

    overrides = {}
    if project_dir:
        overrides["project_dir"] = str(project_dir)
    if no_human_checkpoint:
        overrides["human_checkpoint_after_ideation"] = False
        overrides["human_checkpoint_after_experiments"] = False
        overrides["human_checkpoint_after_manuscript"] = False

    config = load_config(config_path, overrides)
    if verbose:
        config.verbose = True
    if all_sonnet:
        config.models = make_all_sonnet_config()
        console.print("[yellow]Using Claude Sonnet for all agents (test mode)[/yellow]")

    # Determine project dir
    proj_dir = project_dir or Path.cwd()
    config.project_dir = proj_dir

    # Create run directory
    if not run_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = topic[:40].lower().replace(" ", "-").replace("/", "-")
        run_id = f"{timestamp}_{slug}"

    run_dir = proj_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Run directory:[/bold] {run_dir}")

    # Import and run pipeline
    from telecom_researcher.pipeline import Pipeline

    pipeline = Pipeline(config=config, run_dir=run_dir)
    state = asyncio.run(pipeline.run(topic=topic, start_stage=start_stage))

    # Print summary
    console.print(f"\n[bold]Final state:[/bold] stage={state.current_stage} status={state.stage_status}")
    console.print(f"[bold]Cost:[/bold] ${state.total_cost:.2f}")
    console.print(f"[bold]Tokens:[/bold] {state.total_tokens:,}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
def status(run_dir: Path) -> None:
    """Show the status of a pipeline run."""
    from telecom_researcher.state import load_state

    state = load_state(run_dir)
    console.print(f"[bold]Run:[/bold] {state.run_id}")
    console.print(f"[bold]Topic:[/bold] {state.topic}")
    console.print(f"[bold]Stage:[/bold] {state.current_stage} ({state.stage_status})")
    console.print(f"[bold]Cost:[/bold] ${state.total_cost:.2f}")
    console.print(f"[bold]Tokens:[/bold] {state.total_tokens:,}")

    for stage_num, review in state.review_results.items():
        status_icon = "[green]PASS[/green]" if review.passed else "[red]FAIL[/red]"
        console.print(f"  Stage {stage_num}: {status_icon} (score: {review.score:.2f})")


@cli.command()
def cost_report() -> None:
    """Show cost report across all runs."""
    runs_dir = Path.cwd() / "runs"
    if not runs_dir.exists():
        console.print("No runs directory found.")
        return

    from telecom_researcher.state import load_state

    total = 0.0
    for run_path in sorted(runs_dir.iterdir()):
        state_file = run_path / "state" / "pipeline_state.json"
        if state_file.exists():
            state = load_state(run_path)
            console.print(f"  {state.run_id}: ${state.total_cost:.2f} ({state.total_tokens:,} tokens)")
            total += state.total_cost

    console.print(f"\n[bold]Total:[/bold] ${total:.2f}")


if __name__ == "__main__":
    cli()
