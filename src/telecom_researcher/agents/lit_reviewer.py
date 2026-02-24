"""Literature Review agent (Stage 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telecom_researcher.agents.base import BaseAgent


class LitReviewerAgent(BaseAgent):
    name = "lit_reviewer"
    max_iterations = 30  # may need many search calls

    def __init__(self, prompts_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._prompts_dir = prompts_dir

    def get_system_prompt(self, **kwargs: Any) -> str:
        if self._prompts_dir:
            prompt_file = self._prompts_dir / "literature_review.md"
            if prompt_file.exists():
                return prompt_file.read_text(encoding="utf-8")
        return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are a research literature review specialist focusing on telecom, wireless communications, \
networking, and machine learning.

Given a research topic, conduct a thorough literature review by searching arXiv and Semantic Scholar.

Your output must be a valid JSON object with these keys:
- topic: the research topic
- search_queries: list of search queries used
- papers: list of paper objects (title, authors, year, venue, arxiv_id, abstract, key_methods, key_results, limitations, relevance_score)
- gap_analysis: list of research gaps (description, supporting_evidence, opportunity_score)
- taxonomy: dict mapping category names to paper title lists
- bibtex_entries: list of BibTeX strings

Important: All citations must come from actual search results. Never hallucinate papers.
Focus on papers from the last 5 years but include seminal older works.
Aim for 15-20 reviewed papers with 3-5 identified research gaps.

Return ONLY the JSON object, no markdown code fences."""
