"""Tools exposed to the LLM reasoning layer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from spatial_db import database


def _get_dataset_name(dataset_name: str | None) -> str:
    """Resolve dataset from argument first, then from environment."""
    resolved = dataset_name or os.environ.get("DATASET_NAME")
    if not resolved:
        raise ValueError("dataset_name is required (arg or DATASET_NAME env var)")
    return resolved


def search_terms(
    query_terms: list[str],
    dataset_name: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Search component captions with BM25 and return the best matching components.

    Args:
            query_terms: List of term strings from the LLM (e.g. ["red chair", "desk"])
            dataset_name: Dataset to search; falls back to DATASET_NAME env var
            top_k: Max number of results to return

    Returns:
            Dict with ranked matches and metadata.
    """
    if not isinstance(query_terms, list):
        raise ValueError("query_terms must be a list of strings")

    resolved_dataset = _get_dataset_name(dataset_name)

    result = database.bm25_search(
        dataset_name=resolved_dataset,
        search_terms=query_terms,
        top_k=top_k,
        apply_elbow=False
    )
    
    return result


SEARCH_TERMS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_terms",
        "description": (
            "Search component captions using BM25 over the current dataset table and "
            "return the most relevant components."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_terms": {
                    "type": "array",
                    "description": "List of text terms to search for.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of ranked components to return.",
                    "default": 10,
                    "minimum": 1,
                },
            },
            "required": ["query_terms"],
            "additionalProperties": False,
        },
    },
}


TOOLS = [SEARCH_TERMS_TOOL]

TOOL_FUNCTIONS = {
    "search_terms": search_terms,
}
