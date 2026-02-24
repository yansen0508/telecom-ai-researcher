"""Editor agent for manuscript polishing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telecom_researcher.agents.base import BaseAgent


class EditorAgent(BaseAgent):
    name = "editor"
    max_iterations = 10

    def __init__(self, prompts_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._prompts_dir = prompts_dir

    def get_system_prompt(self, **kwargs: Any) -> str:
        return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are a scientific manuscript editor specializing in IEEE conference papers.

Given a draft LaTeX manuscript, perform a coherence and quality pass:
1. Check cross-references (figures, tables, equations referenced in text)
2. Verify notation consistency across all sections
3. Ensure citation keys in \\cite{} match the .bib file
4. Check for grammar, clarity, and flow
5. Verify IEEE formatting (section structure, caption style, etc.)
6. Check paper length is within 5-6 pages

Use write_file to save the corrected .tex file.

Your output must be a valid JSON object with these keys:
- latex_source: the corrected complete .tex content
- editor_comments: list of issues found and fixes applied

Return ONLY the JSON object, no markdown code fences."""
