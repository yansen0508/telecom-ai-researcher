"""Configuration models for the research pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Specification for a single model."""

    provider: str = "anthropic"  # anthropic, openai, google, ollama, openai-compatible
    model_id: str = "anthropic/claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 8192
    base_url: str | None = None  # custom endpoint (e.g., vLLM, Ollama)
    api_key_env: str | None = None  # env var name for API key


class AgentModelConfig(BaseModel):
    """Model assignment for a specific agent role."""

    primary: ModelSpec
    fallback: ModelSpec | None = None


class ModelsConfig(BaseModel):
    """All model assignments."""

    literature_review: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(
            primary=ModelSpec(model_id="anthropic/claude-sonnet-4-20250514"),
        )
    )
    ideation: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(
            primary=ModelSpec(
                model_id="anthropic/claude-opus-4-0-20250514",
                max_tokens=16384,
            ),
        )
    )
    formulation: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(
            primary=ModelSpec(
                model_id="anthropic/claude-opus-4-0-20250514",
                max_tokens=16384,
            ),
        )
    )
    experiments: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(
            primary=ModelSpec(model_id="anthropic/claude-sonnet-4-20250514"),
        )
    )
    analysis: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(
            primary=ModelSpec(model_id="anthropic/claude-sonnet-4-20250514"),
        )
    )
    writing: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(
            primary=ModelSpec(
                model_id="anthropic/claude-opus-4-0-20250514",
                max_tokens=16384,
            ),
        )
    )
    reviewer: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(
            primary=ModelSpec(
                model_id="anthropic/claude-opus-4-0-20250514",
                max_tokens=16384,
            ),
        )
    )
    editor: AgentModelConfig = Field(
        default_factory=lambda: AgentModelConfig(
            primary=ModelSpec(model_id="anthropic/claude-sonnet-4-20250514"),
        )
    )


class QualityConfig(BaseModel):
    """Quality control settings."""

    gate_threshold: float = 0.7  # minimum score to pass gate review (0-1)
    max_revisions_per_stage: int = 3
    max_pipeline_reruns: int = 1
    self_critique_enabled: bool = True
    mock_peer_review_enabled: bool = True


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    # Run settings
    project_dir: Path = Field(default_factory=lambda: Path.cwd())
    run_id: str | None = None  # auto-generated if None

    # Model assignments
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    # Quality control
    quality: QualityConfig = Field(default_factory=QualityConfig)

    # Cost budget (USD)
    cost_budget_total: float = 50.0
    cost_budget_per_stage: dict[str, float] = Field(
        default_factory=lambda: {
            "literature_review": 5.0,
            "ideation": 5.0,
            "formulation": 5.0,
            "experiments": 10.0,
            "analysis": 5.0,
            "writing": 15.0,
            "review": 5.0,
        }
    )

    # Human checkpoints
    human_checkpoint_after_ideation: bool = True
    human_checkpoint_after_manuscript: bool = True

    # Logging
    log_all_llm_calls: bool = True
    verbose: bool = False


def load_config(config_path: Path | None = None, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    """Load pipeline configuration from YAML file with optional overrides.

    Priority: overrides > YAML file > defaults
    """
    config_data: dict[str, Any] = {}

    if config_path and config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}

    # Apply environment variable substitutions
    config_data = _resolve_env_vars(config_data)

    if overrides:
        config_data = _deep_merge(config_data, overrides)

    return PipelineConfig(**config_data)


def _resolve_env_vars(data: Any) -> Any:
    """Recursively resolve ${ENV_VAR} references in config values."""
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_var = data[2:-1]
        return os.environ.get(env_var, data)
    elif isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, with override taking priority."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
