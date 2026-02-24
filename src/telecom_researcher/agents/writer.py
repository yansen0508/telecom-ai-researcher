"""Manuscript Writing agent (Stage 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telecom_researcher.agents.base import BaseAgent


class WriterAgent(BaseAgent):
    name = "writer"
    max_iterations = 15

    def __init__(self, prompts_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._prompts_dir = prompts_dir

    def get_system_prompt(self, **kwargs: Any) -> str:
        if self._prompts_dir:
            prompt_file = self._prompts_dir / "writing.md"
            if prompt_file.exists():
                return prompt_file.read_text(encoding="utf-8")
        return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are an expert scientific paper writer specializing in IEEE conference papers.

Given all previous stage artifacts, write a complete IEEE-format paper:
1. Abstract (150-200 words)
2. Introduction (motivation, background, contributions)
3. Related Work (categorized literature)
4. System Model / Problem Formulation (equations from formulation artifact)
5. Proposed Method (detailed algorithm)
6. Simulation Results (from analysis artifact, include figures/tables)
7. Conclusion (summary, future work)

Use the write_file tool to save the .tex and .bib files.
Then use latex_compile to compile to PDF.

Your output must be a valid JSON object with these keys:
- latex_source: complete .tex file content
- bibtex_source: complete .bib file content
- section_drafts: dict mapping section names to their LaTeX content
- editor_comments: list of any issues found and fixed

Write in LaTeX using IEEEtran class. Total length: 5-6 pages.
Use \\cite{} for references, \\includegraphics for figures, numbered equations.
Be precise and concise. Quantify claims. Acknowledge limitations honestly.
Return ONLY the JSON object, no markdown code fences."""
