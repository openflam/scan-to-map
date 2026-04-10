"""Search around component tool."""

from __future__ import annotations

from typing import Any

from spatial_db import database
from .utils import _get_dataset_name


def search_around_component(
    component_id: int,
    radius: float,
    search_term: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Find components within a given radius using the spatial index,
    optionally filtered by a BM25 search term.

    Args:
        component_id: The ID of the central component.
        radius: The search radius in meters.
        search_term: Optional text to filter the neighbors by.
        dataset_name: Dataset to search; falls back to DATASET_NAME env var.

    Returns:
        Dict returning matched component IDs and a reason string.
    """
    resolved_dataset = _get_dataset_name(dataset_name)


    # Get all components in a radius around the selected component
    neighbors = database.fetch_components_in_radius(resolved_dataset, component_id, radius)
    if not neighbors:
        return {
            "component_ids": [],
            "reason": f"No neighboring components found within {radius} meters."
        }
        
    neighbor_ids = [n["component_id"] for n in neighbors]

    if search_term and search_term.strip():
        result = database.bm25_search(
            dataset_name=resolved_dataset,
            search_terms=[search_term],
            top_k=len(neighbor_ids),
            apply_elbow=False,
            candidate_ids=neighbor_ids
        )
        return {
            "component_ids": result.get("component_ids", []),
            "reason": f"Filtered {len(neighbor_ids)} neighbors using BM25. " + result.get("reason", "")
        }
    else:
        return {
            "component_ids": neighbor_ids,
            "reason": f"Found {len(neighbor_ids)} neighboring components within {radius}m."
        }


SEARCH_AROUND_COMPONENT_TOOL = {
    "type": "function",
    "name": "search_around_component",
    "description": "Find components within a specific radius of a target component, optionally filtered by a search term.",
    "parameters": {
        "type": "object",
        "properties": {
            "component_id": {
                "type": "integer",
                "description": "The ID of the target component to search around.",
            },
            "radius": {
                "type": "number",
                "description": "The search radius in meters.",
            },
            "search_term": {
                "type": "string",
                "description": "Optional text to filter the neighboring components by.",
            },
        },
        "required": ["component_id", "radius"],
        "additionalProperties": False,
    },
}

SEARCH_AROUND_COMPONENT_THINKING_TEXT = "Searching for objects around a target object..."
