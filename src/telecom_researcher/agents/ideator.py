"""Ideation agent (Stage 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telecom_researcher.agents.base import BaseAgent


class IdeatorAgent(BaseAgent):
    name = "ideator"
    max_iterations = 10

    def __init__(self, prompts_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._prompts_dir = prompts_dir

    def get_system_prompt(self, **kwargs: Any) -> str:
        if self._prompts_dir:
            prompt_file = self._prompts_dir / "ideation.md"
            if prompt_file.exists():
                return prompt_file.read_text(encoding="utf-8")
        return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are a creative research ideation specialist in telecom and machine learning.

Given a literature review with identified research gaps, generate 3-5 novel research ideas.
For each idea provide: title, abstract_sketch, approach, expected_contribution, novelty_score, \
feasibility_score, significance_score (all scores 0-1).

Select the best idea based on novelty + feasibility + significance.
Define what is in-scope and out-of-scope.

Your output must be a valid JSON object with these keys:
- candidate_ideas: list of idea objects
- selected_idea: the chosen idea object
- novelty_assessment: detailed reasoning
- feasibility_assessment: detailed reasoning
- scope_definition: "In scope: ... Out of scope: ..."

Be ambitious but realistic — the idea must be validatable through simulation.
Return ONLY the JSON object, no markdown code fences."""
