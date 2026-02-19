from typing import Dict, Any, List
import json
import sqlite3
from semantic_search import SemanticSearchProvider
import time


def process_query(
    query: str,
    db_path: str,
    provider: SemanticSearchProvider,
) -> Dict[str, Any]:
    """
    Process a search query and return the bounding boxes of the most relevant components.

    Args:
        query: The search query string
        db_path: Path to the SQLite components database
        provider: Semantic search provider to use for matching

    Returns:
        A dictionary with "bbox" (list of bounding boxes), "reason" (explanation for the choice),
        and "search_time_ms" (time taken to match components in milliseconds)
    """
    # Get matched component IDs from the provider and measure time
    start_time = time.perf_counter()
    result = provider.match_components(query)
    end_time = time.perf_counter()
    search_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    component_ids = result["component_ids"]
    reason = result["reason"]

    # Fetch bounding boxes for matched components from the DB
    valid_bboxes = []
    valid_component_ids = []
    invalid_ids = []

    if component_ids:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        placeholders = ",".join("?" * len(component_ids))
        cur.execute(
            f"SELECT component_id, bbox_json FROM components WHERE component_id IN ({placeholders})",
            component_ids,
        )
        bbox_map = {
            row["component_id"]: json.loads(row["bbox_json"]) for row in cur.fetchall()
        }
        con.close()

        for component_id in component_ids:
            if component_id in bbox_map:
                valid_bboxes.append(bbox_map[component_id])
                valid_component_ids.append(component_id)
            else:
                invalid_ids.append(component_id)

    # Handle cases where no valid bboxes were found
    if not valid_bboxes:
        print("Warning: No valid component IDs found. Using first component.")
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            "SELECT component_id, bbox_json FROM components ORDER BY component_id LIMIT 1"
        )
        row = cur.fetchone()
        con.close()
        if row:
            valid_bboxes = [json.loads(row["bbox_json"])]
            valid_component_ids = [row["component_id"]]
    elif invalid_ids:
        print(f"Warning: Some invalid component IDs were ignored: {invalid_ids}")

    # Return the list of bounding boxes, component IDs, reason, and search time
    return {
        "bbox": valid_bboxes,
        "component_ids": valid_component_ids,
        "reason": reason,
        "search_time_ms": search_time_ms,
    }
