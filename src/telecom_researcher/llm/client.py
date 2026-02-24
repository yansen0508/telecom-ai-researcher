"""Unified LLM client with retry, fallback, and cost tracking via LiteLLM."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import acompletion, completion_cost

from telecom_researcher.config import AgentModelConfig, ModelSpec

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    is_fallback: bool = False
    tool_calls: int = 0


@dataclass
class CostTracker:
    """Tracks cumulative token usage and cost across all calls."""

    records: list[LLMCallRecord] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def add(self, record: LLMCallRecord) -> None:
        self.records.append(record)
        logger.debug(
            f"LLM call: model={record.model} tokens={record.input_tokens}+{record.output_tokens} "
            f"cost=${record.cost_usd:.4f}"
        )

    def summary(self) -> dict[str, Any]:
        by_model: dict[str, dict[str, float]] = {}
        for r in self.records:
            if r.model not in by_model:
                by_model[r.model] = {"calls": 0, "tokens": 0, "cost": 0.0}
            by_model[r.model]["calls"] += 1
            by_model[r.model]["tokens"] += r.input_tokens + r.output_tokens
            by_model[r.model]["cost"] += r.cost_usd
        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_calls": len(self.records),
            "by_model": by_model,
        }


def _build_litellm_params(spec: ModelSpec) -> dict[str, Any]:
    """Build parameters for a litellm call from a ModelSpec."""
    params: dict[str, Any] = {
        "model": spec.model_id,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }

    # Custom base URL (for vLLM, Ollama, or proxied endpoints)
    if spec.base_url:
        params["api_base"] = spec.base_url
    elif spec.provider == "anthropic" and os.environ.get("ANTHROPIC_BASE_URL"):
        params["api_base"] = os.environ["ANTHROPIC_BASE_URL"]

    # API key: explicit env var > provider-specific env var > litellm auto-detect
    if spec.api_key_env:
        api_key = os.environ.get(spec.api_key_env)
        if api_key:
            params["api_key"] = api_key
    elif spec.provider == "anthropic":
        # Try ANTHROPIC_API_KEY first, fall back to CLAUDE_CODE_OAUTH_TOKEN
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        if api_key:
            params["api_key"] = api_key
    elif spec.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            params["api_key"] = api_key
    elif spec.provider == "google":
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            params["api_key"] = api_key

    return params


async def call_llm(
    model_config: AgentModelConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    cost_tracker: CostTracker | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call an LLM with retry and fallback logic.

    Returns the raw response dict with keys: content, tool_calls, finish_reason, model.
    """
    specs_to_try = [("primary", model_config.primary)]
    if model_config.fallback:
        specs_to_try.append(("fallback", model_config.fallback))

    last_error: Exception | None = None

    for label, spec in specs_to_try:
        params = _build_litellm_params(spec)
        params["messages"] = messages
        if tools:
            params["tools"] = tools

        for attempt in range(max_retries):
            try:
                response = await acompletion(**params)
                choice = response.choices[0]

                # Track cost
                record = LLMCallRecord(
                    model=spec.model_id,
                    input_tokens=response.usage.prompt_tokens if response.usage else 0,
                    output_tokens=response.usage.completion_tokens if response.usage else 0,
                    cost_usd=_safe_cost(response),
                    is_fallback=(label == "fallback"),
                    tool_calls=len(choice.message.tool_calls or []) if choice.message.tool_calls else 0,
                )
                if cost_tracker:
                    cost_tracker.add(record)

                # Normalize response
                result: dict[str, Any] = {
                    "content": choice.message.content or "",
                    "tool_calls": [],
                    "finish_reason": choice.finish_reason,
                    "model": spec.model_id,
                    "usage": record,
                }

                if choice.message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ]

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed ({label} {spec.model_id}, attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)  # exponential backoff

        logger.warning(f"All retries exhausted for {label} model {spec.model_id}")

    raise RuntimeError(f"All models failed. Last error: {last_error}")


def _safe_cost(response: Any) -> float:
    """Safely compute cost, returning 0 if cost calculation fails."""
    try:
        return completion_cost(completion_response=response)
    except Exception:
        return 0.0
