"""Figure generation tool with IEEE styling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from telecom_researcher.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class FigureGeneratorTool(Tool):
    """Generate publication-quality figures using matplotlib with IEEE style."""

    def __init__(self, output_dir: Path | None = None):
        self._output_dir = output_dir

    @property
    def name(self) -> str:
        return "figure_generate"

    @property
    def description(self) -> str:
        return (
            "Generate a publication-quality figure using matplotlib. "
            "Provide the matplotlib Python code that creates the figure. "
            "The code should use plt.savefig() to save to the specified filename. "
            "IEEE style (3.5-inch width, serif font, 8pt) is applied automatically."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Matplotlib Python code that generates the figure",
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (e.g., 'fig1_ber.pdf')",
                },
            },
            "required": ["code", "filename"],
        }

    async def execute(self, *, code: str, filename: str) -> ToolResult:
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt

            from telecom_researcher.templates.figure_style import IEEE_STYLE

            # Apply IEEE style
            matplotlib.rcParams.update(IEEE_STYLE)

            output_dir = self._output_dir or Path(".")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename

            # Inject the output path into the code namespace
            exec_globals = {
                "plt": plt,
                "np": __import__("numpy"),
                "output_path": str(output_path),
                "__builtins__": __builtins__,
            }

            # Execute the plotting code
            exec(code, exec_globals)

            # If the code didn't call savefig, save the current figure
            if not output_path.exists():
                plt.savefig(str(output_path), dpi=300, bbox_inches="tight")

            plt.close("all")

            if output_path.exists():
                logger.info(f"Figure saved: {output_path}")
                return ToolResult(
                    success=True,
                    data={
                        "path": str(output_path),
                        "size_kb": output_path.stat().st_size // 1024,
                    },
                )
            else:
                return ToolResult(success=False, error="Figure file not created")

        except Exception as e:
            import traceback
            return ToolResult(success=False, error=f"{e}\n{traceback.format_exc()}")
