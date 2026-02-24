"""Problem Formulation agent (Stage 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telecom_researcher.agents.base import BaseAgent


class FormulatorAgent(BaseAgent):
    name = "formulator"
    max_iterations = 10

    def __init__(self, prompts_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._prompts_dir = prompts_dir

    def get_system_prompt(self, **kwargs: Any) -> str:
        if self._prompts_dir:
            prompt_file = self._prompts_dir / "formulation.md"
            if prompt_file.exists():
                return prompt_file.read_text(encoding="utf-8")
        return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are a mathematical modeling specialist for telecom/wireless communications and machine learning.

Given a selected research idea, produce a rigorous mathematical formulation including:
1. System model (signal model, channel model) in LaTeX
2. Notation table (symbol -> meaning, follow IEEE/telecom conventions)
3. Optimization or learning problem statement in LaTeX
4. Theoretical analysis (complexity, convergence, bounds) if applicable
5. Evaluation metrics with LaTeX formulas
6. 2-4 baseline methods for comparison

Your output must be a valid JSON object with these keys:
- system_model: LaTeX-formatted system model description
- notation_table: dict of symbol -> meaning
- problem_statement: LaTeX-formatted optimization problem
- theoretical_analysis: string
- evaluation_metrics: list of {name, formula, description}
- baseline_methods: list of method names
- latex_equations: list of standalone LaTeX equation blocks

All equations must be valid LaTeX. Use standard telecom notation.
Return ONLY the JSON object, no markdown code fences."""
