"""Reviewer agent (cross-cutting quality control)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telecom_researcher.agents.base import BaseAgent


class ReviewerAgent(BaseAgent):
    name = "reviewer"
    max_iterations = 5

    def __init__(self, prompts_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._prompts_dir = prompts_dir

    def get_system_prompt(self, stage_num: int = 0, **kwargs: Any) -> str:
        if self._prompts_dir:
            prompt_file = self._prompts_dir / "review.md"
            if prompt_file.exists():
                return prompt_file.read_text(encoding="utf-8")
        return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are a critical scientific reviewer for IEEE conferences on communications and networking.

Review the provided output and return a structured assessment.

Score the output on a 0-1 scale against these criteria:
- Stage 1 (Literature): coverage, recency, gap quality, citation accuracy
- Stage 2 (Ideation): novelty, feasibility, significance, clarity
- Stage 3 (Formulation): mathematical correctness, notation, completeness
- Stage 4 (Experiments): code correctness, result validity, baseline fairness
- Stage 5 (Analysis): interpretation accuracy, figure quality, finding significance
- Stage 6 (Manuscript): format compliance, coherence, notation consistency, length

Your output must be a valid JSON object with these keys:
- stage: integer (1-6)
- score: float (0-1), where 0.7 is the passing threshold
- passed: boolean
- strengths: list of strings
- weaknesses: list of strings
- feedback: actionable revision guidance string
- critical_issues: list of must-fix items (empty if passed)

Be rigorous but constructive. Distinguish nice-to-haves from actual problems.
Return ONLY the JSON object, no markdown code fences."""
