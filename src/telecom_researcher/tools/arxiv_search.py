"""arXiv paper search tool."""

from __future__ import annotations

import logging
from typing import Any

import arxiv

from telecom_researcher.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ArxivSearchTool(Tool):
    @property
    def name(self) -> str:
        return "arxiv_search"

    @property
    def description(self) -> str:
        return (
            "Search arXiv for academic papers. Returns paper metadata including "
            "title, authors, abstract, arxiv_id, and published date. "
            "Use specific technical queries for best results."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'beam management 6G machine learning')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10, max: 30)",
                    "default": 10,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort order (default: relevance)",
                    "default": "relevance",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self, *, query: str, max_results: int = 10, sort_by: str = "relevance"
    ) -> ToolResult:
        try:
            max_results = min(max_results, 30)

            sort_criterion = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
            }.get(sort_by, arxiv.SortCriterion.Relevance)

            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion,
            )

            papers = []
            for result in client.results(search):
                papers.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors[:5]],  # limit authors
                    "abstract": result.summary[:1000],  # truncate long abstracts
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "published": result.published.strftime("%Y-%m-%d") if result.published else "",
                    "updated": result.updated.strftime("%Y-%m-%d") if result.updated else "",
                    "categories": result.categories[:5],
                    "pdf_url": result.pdf_url,
                })

            logger.info(f"arXiv search '{query}': found {len(papers)} papers")
            return ToolResult(success=True, data=papers)

        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return ToolResult(success=False, error=str(e))
