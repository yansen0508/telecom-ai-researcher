"""Tool protocol, registry, and base implementations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    success: bool
    data: Any = None
    error: str | None = None

    def to_str(self) -> str:
        """Serialize for inclusion in LLM messages."""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.data, str):
            return self.data
        return json.dumps(self.data, ensure_ascii=False, default=str)


class Tool(ABC):
    """Base class for all tools available to agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters."""
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible tool schema for LLM calls."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry of available tools, manages tool lookup by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schemas(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for specified tools (or all)."""
        tools = self._tools.values()
        if tool_names:
            tools = [t for t in tools if t.name in tool_names]
        return [t.to_openai_schema() for t in tools]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return ToolResult(success=False, error=f"Tool execution failed: {e}")

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


# --- Concrete tools: read_file and write_file (always available) ---

class ReadFileTool(Tool):
    def __init__(self, working_dir: Any | None = None):
        self._working_dir = working_dir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path to read"},
            },
            "required": ["path"],
        }

    async def execute(self, *, path: str) -> ToolResult:
        try:
            from pathlib import Path
            p = Path(path)
            if not p.is_absolute() and self._working_dir:
                p = self._working_dir / p
            content = p.read_text(encoding="utf-8")
            return ToolResult(success=True, data=content)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class WriteFileTool(Tool):
    def __init__(self, working_dir: Any | None = None):
        self._working_dir = working_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, *, path: str, content: str) -> ToolResult:
        try:
            from pathlib import Path
            p = Path(path)
            if not p.is_absolute() and self._working_dir:
                p = self._working_dir / p
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(success=True, data=f"Written {len(content)} chars to {p}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
