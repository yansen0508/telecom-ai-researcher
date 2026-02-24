"""Experiment agent (Stage 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telecom_researcher.agents.base import BaseAgent


class ExperimenterAgent(BaseAgent):
    name = "experimenter"
    max_iterations = 25  # code-debug loops can be long

    def __init__(self, prompts_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._prompts_dir = prompts_dir

    def get_system_prompt(self, **kwargs: Any) -> str:
        if self._prompts_dir:
            prompt_file = self._prompts_dir / "experiments.md"
            if prompt_file.exists():
                return prompt_file.read_text(encoding="utf-8")
        return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are a simulation engineer specializing in telecom/wireless communication systems.

Given a problem formulation, design and implement simulation experiments:
1. Design experiment configurations (parameters, scenarios, baselines)
2. Write complete, runnable Python simulation code using numpy/scipy
3. Execute the code using the code_execute tool
4. Debug any errors (max 10 iterations)
5. Collect structured numerical results

Use the write_file tool to save your simulation code, then code_execute to run it.
Save results to JSON files for later analysis.

Your final output must be a valid JSON object with these keys:
- experiment_configs: list of {name, parameters, description}
- simulation_code: the complete Python code as a string
- code_path: path where code was saved
- raw_results: dict of experiment results
- result_tables: list of {header, rows} for comparison tables
- execution_logs: list of stdout/stderr strings
- debug_history: list of {error_message, fix_description, success}

Code must be self-contained, use vectorized numpy operations, and set random seeds.
Return ONLY the JSON object, no markdown code fences."""
