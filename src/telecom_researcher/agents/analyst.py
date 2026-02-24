"""Analysis agent (Stage 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telecom_researcher.agents.base import BaseAgent


class AnalystAgent(BaseAgent):
    name = "analyst"
    max_iterations = 15

    def __init__(self, prompts_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._prompts_dir = prompts_dir

    def get_system_prompt(self, **kwargs: Any) -> str:
        if self._prompts_dir:
            prompt_file = self._prompts_dir / "analysis.md"
            if prompt_file.exists():
                return prompt_file.read_text(encoding="utf-8")
        return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are a data analysis and visualization specialist for telecom research.

Given experiment results and problem formulation, analyze and visualize:
1. Compare proposed method vs. baselines quantitatively
2. Generate IEEE-style matplotlib figures (3.5" width, serif font, 8pt)
3. Create LaTeX-formatted tables with booktabs style
4. Summarize 3-5 key findings
5. Write a detailed analysis narrative

Use the figure_generate tool to create figures and write_file for saving code.

Your output must be a valid JSON object with these keys:
- figures: list of {filename, caption, latex_include}
- tables: list of {latex_code, caption}
- key_findings: list of bullet-point strings
- analysis_narrative: detailed interpretation
- figure_generation_code: complete matplotlib code

All figures must be publication-quality. Use consistent colors and markers.
Return ONLY the JSON object, no markdown code fences."""
