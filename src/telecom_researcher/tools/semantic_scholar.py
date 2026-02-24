"""Semantic Scholar paper search tool."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from telecom_researcher.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarSearchTool(Tool):
    @property
    def name(self) -> str:
        return "semantic_scholar_search"

    @property
    def description(self) -> str:
        return (
            "Search Semantic Scholar for academic papers. Returns paper metadata "
            "including title, authors, abstract, citation count, and venue. "
            "Good for finding highly-cited and influential papers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results (default: 10, max: 20)",
                    "default": 10,
                },
                "year_range": {
                    "type": "string",
                    "description": "Year filter (e.g., '2020-2026', '2023-')",
                    "default": "",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self, *, query: str, max_results: int = 10, year_range: str = ""
    ) -> ToolResult:
        try:
            max_results = min(max_results, 20)
            fields = "title,authors,abstract,year,venue,citationCount,externalIds"

            params: dict[str, Any] = {
                "query": query,
                "limit": max_results,
                "fields": fields,
            }
            if year_range:
                params["year"] = year_range

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{S2_API_BASE}/paper/search",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()

            papers = []
            for paper in data.get("data", []):
                arxiv_id = ""
                external_ids = paper.get("externalIds") or {}
                if "ArXiv" in external_ids:
                    arxiv_id = external_ids["ArXiv"]

                papers.append({
                    "title": paper.get("title", ""),
                    "authors": [
                        a.get("name", "") for a in (paper.get("authors") or [])[:5]
                    ],
                    "abstract": (paper.get("abstract") or "")[:1000],
                    "year": paper.get("year", 0),
                    "venue": paper.get("venue", ""),
                    "citation_count": paper.get("citationCount", 0),
                    "arxiv_id": arxiv_id,
                })

            logger.info(f"Semantic Scholar search '{query}': found {len(papers)} papers")
            return ToolResult(success=True, data=papers)

        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return ToolResult(success=False, error=str(e))
