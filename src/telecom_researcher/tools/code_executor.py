"""Sandboxed Python code execution tool."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from telecom_researcher.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class CodeExecutorTool(Tool):
    """Execute Python code in a sandboxed subprocess with timeout."""

    def __init__(self, working_dir: Path | None = None, timeout: int = 120):
        self._working_dir = working_dir or Path(tempfile.gettempdir())
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "code_execute"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a sandboxed subprocess. "
            "The code runs as a standalone script with access to numpy, scipy, matplotlib, pandas. "
            "Returns stdout and stderr. Use this to run simulations and generate results. "
            "Always save results to files (JSON/numpy) rather than printing large outputs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename to save the code as (default: script.py)",
                    "default": "script.py",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default: {120}, max: 600)",
                    "default": 120,
                },
            },
            "required": ["code"],
        }

    async def execute(
        self, *, code: str, filename: str = "script.py", timeout: int = 120
    ) -> ToolResult:
        timeout = min(timeout, 600)

        # Write code to file
        code_path = self._working_dir / filename
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code, encoding="utf-8")

        # Build restricted environment
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "LANG": "en_US.UTF-8",
            # Allow matplotlib to work headless
            "MPLBACKEND": "Agg",
        }

        try:
            process = await asyncio.create_subprocess_exec(
                "python3",
                str(code_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._working_dir),
                env=env,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")[:10000]
            stderr = stderr_bytes.decode("utf-8", errors="replace")[:5000]

            result_data = {
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "code_path": str(code_path),
            }

            if process.returncode != 0:
                logger.warning(f"Code execution failed (rc={process.returncode})")
                return ToolResult(
                    success=False,
                    data=result_data,
                    error=f"Exit code {process.returncode}:\n{stderr}",
                )

            logger.info(f"Code executed successfully: {filename}")
            return ToolResult(success=True, data=result_data)

        except asyncio.TimeoutError:
            process.kill()
            return ToolResult(
                success=False,
                error=f"Execution timed out after {timeout} seconds",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
