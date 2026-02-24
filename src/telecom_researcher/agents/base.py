"""Base agent class implementing the agentic loop with tool use."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from telecom_researcher.config import AgentModelConfig
from telecom_researcher.llm.client import CostTracker, call_llm
from telecom_researcher.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all pipeline agents.

    Implements the core agentic loop:
    1. Send messages to LLM (with tool schemas)
    2. If LLM returns tool calls, execute them and append results
    3. Repeat until LLM returns a final text response (finish_reason=stop)
    4. Return the final content

    Subclasses define their system prompt, tool set, and output parsing.
    """

    # Subclasses should override these
    name: str = "base_agent"
    max_iterations: int = 20  # max tool-use rounds before forced stop

    def __init__(
        self,
        model_config: AgentModelConfig,
        tool_registry: ToolRegistry | None = None,
        allowed_tools: list[str] | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        self.model_config = model_config
        self.tool_registry = tool_registry or ToolRegistry()
        self.allowed_tools = allowed_tools  # None = all registered tools
        self.cost_tracker = cost_tracker or CostTracker()

    def get_system_prompt(self, **kwargs: Any) -> str:
        """Return the system prompt for this agent. Override in subclasses."""
        return "You are a helpful research assistant."

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for the LLM call."""
        return self.tool_registry.get_schemas(self.allowed_tools)

    async def run(
        self,
        user_message: str,
        context: str = "",
        feedback: str = "",
        **prompt_kwargs: Any,
    ) -> AgentResult:
        """Execute the agentic loop.

        Args:
            user_message: The primary task/instruction for this agent.
            context: Additional context from previous pipeline stages.
            feedback: Reviewer feedback for revision cycles.
            **prompt_kwargs: Extra args passed to get_system_prompt().

        Returns:
            AgentResult with the final output and execution metadata.
        """
        system_prompt = self.get_system_prompt(**prompt_kwargs)

        # Build the user message with context and feedback
        parts = []
        if context:
            parts.append(f"## Context from Previous Stages\n\n{context}")
        if feedback:
            parts.append(f"## Reviewer Feedback (address these issues)\n\n{feedback}")
        parts.append(f"## Task\n\n{user_message}")
        parts.append(
            "## CRITICAL OUTPUT REQUIREMENT\n\n"
            "After you have finished using tools, your FINAL response must be ONLY a valid JSON object. "
            "Do NOT include any explanation, markdown, or text before or after the JSON. "
            "Start your final response with { and end with }."
        )
        full_user_message = "\n\n---\n\n".join(parts)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_message},
        ]

        tool_schemas = self.get_tool_schemas()
        tool_call_log: list[dict[str, Any]] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            response = await call_llm(
                model_config=self.model_config,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                cost_tracker=self.cost_tracker,
            )

            # Check if the model wants to use tools
            if response["tool_calls"]:
                # Build assistant message with tool calls in OpenAI format
                tool_calls_formatted = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in response["tool_calls"]
                ]
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response["content"] or None,
                    "tool_calls": tool_calls_formatted,
                }
                messages.append(assistant_msg)

                # Execute each tool call and append results
                for tc in response["tool_calls"]:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    logger.info(f"[{self.name}] Tool call: {func_name}({list(func_args.keys())})")
                    result = await self.tool_registry.execute(func_name, func_args)

                    tool_call_log.append({
                        "iteration": iteration,
                        "tool": func_name,
                        "args": func_args,
                        "success": result.success,
                        "result_preview": result.to_str()[:500],
                    })

                    # Append tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result.to_str(),
                    })
            else:
                # No tool calls - this is the final response
                logger.info(f"[{self.name}] Completed after {iteration} iterations")
                return AgentResult(
                    content=response["content"],
                    model_used=response["model"],
                    iterations=iteration,
                    tool_calls=tool_call_log,
                    cost_tracker=self.cost_tracker,
                )

        # Hit max iterations - return whatever we have
        logger.warning(f"[{self.name}] Hit max iterations ({self.max_iterations})")
        return AgentResult(
            content=response.get("content", "Max iterations reached without final response."),
            model_used=response.get("model", "unknown"),
            iterations=iteration,
            tool_calls=tool_call_log,
            cost_tracker=self.cost_tracker,
            hit_max_iterations=True,
        )


class AgentResult:
    """Result from an agent execution."""

    def __init__(
        self,
        content: str,
        model_used: str,
        iterations: int,
        tool_calls: list[dict[str, Any]],
        cost_tracker: CostTracker,
        hit_max_iterations: bool = False,
    ):
        self.content = content
        self.model_used = model_used
        self.iterations = iterations
        self.tool_calls = tool_calls
        self.cost_tracker = cost_tracker
        self.hit_max_iterations = hit_max_iterations

    @property
    def total_cost(self) -> float:
        return self.cost_tracker.total_cost

    @property
    def total_tokens(self) -> int:
        return self.cost_tracker.total_tokens

    def __repr__(self) -> str:
        return (
            f"AgentResult(model={self.model_used}, iterations={self.iterations}, "
            f"tool_calls={len(self.tool_calls)}, tokens={self.total_tokens}, "
            f"cost=${self.total_cost:.4f})"
        )
