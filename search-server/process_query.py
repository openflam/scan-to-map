from typing import Dict, Any, List
import json
import sys
from pathlib import Path
from semantic_search import SemanticSearchProvider
import time

from spatial_db import database

def process_query(
    query: str,
    dataset_name: str,
    provider: SemanticSearchProvider,
) -> Dict[str, Any]:
    """
    Process a search query and return the bounding boxes of the most relevant components.

    Args:
        query: The search query string
        dataset_name: Name of the dataset
        provider: Semantic search provider to use for matching

    Returns:
        A dictionary with "bbox" (list of bounding boxes), "reason" (explanation for the choice),
        "search_time_ms" (time taken to match components in milliseconds), and optionally
        "answer_selection" when the provider returns ScanQA pool-based answers (OpenAI with pools).
    """
    # Get matched component IDs from the provider and measure time
    start_time = time.perf_counter()
    result = provider.match_components(query)
    end_time = time.perf_counter()
    search_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    component_ids = result["component_ids"]
    reason = result["reason"]
    answer_selection = result.get("answer_selection")

    # Fetch bounding boxes for matched components from the DB
    valid_bboxes = []
    valid_component_ids = []
    invalid_ids = []

    if component_ids:
        rows = database.fetch_components_by_ids(dataset_name, component_ids)
        bbox_map = {}
        for row in rows:
            try:
                bbox_map[row["component_id"]] = json.loads(row["bbox_json"]) if row["bbox_json"] else {}
            except json.JSONDecodeError:
                bbox_map[row["component_id"]] = {}

        for component_id in component_ids:
            if component_id in bbox_map:
                valid_bboxes.append(bbox_map[component_id])
                valid_component_ids.append(component_id)
            else:
                invalid_ids.append(component_id)

    # Handle cases where no valid bboxes were found
    if not valid_bboxes:
        print("Warning: No valid component IDs found. Using first component.")
        row = database.fetch_first_component(dataset_name)
        if row:
            try:
                valid_bboxes = [json.loads(row["bbox_json"]) if row["bbox_json"] else {}]
            except json.JSONDecodeError:
                valid_bboxes = [{}]
            valid_component_ids = [row["component_id"]]
    elif invalid_ids:
        print(f"Warning: Some invalid component IDs were ignored: {invalid_ids}")

    # Return the list of bounding boxes, component IDs, reason, and search time
    out = {
        "bbox": valid_bboxes,
        "component_ids": valid_component_ids,
        "reason": reason,
        "search_time_ms": search_time_ms,
    }
    if answer_selection is not None:
        out["answer_selection"] = answer_selection
    return out
