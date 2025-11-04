from typing import Dict, Any, List
from semantic_search import SemanticSearchProvider
import time


def process_query(
    query: str,
    component_captions: Dict[int, Any],
    bbox_lookup: Dict[int, Any],
    provider: SemanticSearchProvider,
) -> Dict[str, Any]:
    """
    Process a search query and return the bounding boxes of the most relevant components.

    Args:
        query: The search query string
        component_captions: Dictionary keyed by component ID with captions
        bbox_lookup: Dictionary keyed by component ID with bbox data
        provider: Semantic search provider to use for matching

    Returns:
        A dictionary with "bbox" (list of bounding boxes), "reason" (explanation for the choice),
        and "search_time_ms" (time taken to match components in milliseconds)
    """
    # Use the provided provider
    search_provider = provider

    # Get matched component IDs from the provider and measure time
    start_time = time.perf_counter()
    result = search_provider.match_components(query)
    end_time = time.perf_counter()
    search_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    
    component_ids = result["component_ids"]
    reason = result["reason"]

    # Collect bounding boxes for the matched components
    valid_bboxes = []
    invalid_ids = []

    for component_id in component_ids:
        if component_id in component_captions and component_id in bbox_lookup:
            valid_bboxes.append(bbox_lookup[component_id])
        else:
            invalid_ids.append(component_id)

    # Handle cases where no valid bboxes were found
    if not valid_bboxes:
        print(f"Warning: No valid component IDs found. Using first component.")
        first_component_id = list(component_captions.keys())[0]
        if first_component_id in bbox_lookup:
            valid_bboxes = [bbox_lookup[first_component_id]]
        else:
            # Fallback to first bbox if component_id doesn't match
            valid_bboxes = [list(bbox_lookup.values())[0]]
        reason = "Fallback: Using first available component due to invalid IDs."
    elif invalid_ids:
        print(f"Warning: Some invalid component IDs were ignored: {invalid_ids}")

    # Return the list of bounding boxes, reason, and search time
    return {"bbox": valid_bboxes, "reason": reason, "search_time_ms": search_time_ms}
