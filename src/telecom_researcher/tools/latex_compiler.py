"""LaTeX compilation tool using tectonic."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from telecom_researcher.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class LatexCompilerTool(Tool):
    """Compile LaTeX documents to PDF using tectonic."""

    def __init__(self, working_dir: Path | None = None):
        self._working_dir = working_dir

    @property
    def name(self) -> str:
        return "latex_compile"

    @property
    def description(self) -> str:
        return (
            "Compile a LaTeX .tex file to PDF using tectonic. "
            "Tectonic automatically handles BibTeX and package downloads. "
            "Provide the path to the .tex file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tex_path": {
                    "type": "string",
                    "description": "Path to the .tex file to compile",
                },
            },
            "required": ["tex_path"],
        }

    async def execute(self, *, tex_path: str) -> ToolResult:
        tex_file = Path(tex_path)
        if not tex_file.exists():
            return ToolResult(success=False, error=f"File not found: {tex_path}")

        if not shutil.which("tectonic"):
            return ToolResult(
                success=False,
                error="tectonic not found. Install with: brew install tectonic",
            )

        working_dir = tex_file.parent

        try:
            process = await asyncio.create_subprocess_exec(
                "tectonic",
                str(tex_file.name),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(working_dir),
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=120
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            pdf_path = working_dir / tex_file.stem
            pdf_path = pdf_path.with_suffix(".pdf")

            if process.returncode != 0:
                return ToolResult(
                    success=False,
                    error=f"LaTeX compilation failed:\n{stderr}\n{stdout}",
                )

            if pdf_path.exists():
                logger.info(f"PDF compiled: {pdf_path}")
                return ToolResult(
                    success=True,
                    data={
                        "pdf_path": str(pdf_path),
                        "size_kb": pdf_path.stat().st_size // 1024,
                        "log": stdout[:2000],
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Compilation succeeded but PDF not found at {pdf_path}",
                )

        except asyncio.TimeoutError:
            return ToolResult(success=False, error="LaTeX compilation timed out (120s)")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
