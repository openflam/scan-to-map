from typing import Dict, Any, List, Optional
from semantic_search import SemanticSearchProvider, OpenAIProvider, BM25Provider


def set_provider(provider: SemanticSearchProvider) -> None:
    """
    Set a custom semantic search provider.

    Args:
        provider: An instance of a SemanticSearchProvider
    """
    global _default_provider
    _default_provider = provider


def process_query(
    query: str,
    component_captions: Dict[int, Any],
    bbox_lookup: Dict[int, Any],
    provider: Optional[SemanticSearchProvider] = None,
) -> Dict[str, Any]:
    """
    Process a search query and return the bounding boxes of the most relevant components.

    Args:
        query: The search query string
        component_captions: Dictionary keyed by component ID with captions
        bbox_lookup: Dictionary keyed by component ID with bbox data
        provider: Optional custom semantic search provider (uses default if not provided)

    Returns:
        A dictionary with "bbox" (list of bounding boxes) and "reason" (explanation for the choice)
    """
    # Use provided provider, or default provider, or create BM25 on-the-fly
    if provider is not None:
        search_provider = provider
    else:
        # Default: create a BM25 provider with the current component_captions
        search_provider = BM25Provider(component_captions)

    # Get matched component IDs from the provider
    result = search_provider.match_components(query, component_captions)
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

    # Return the list of bounding boxes and reason
    return {"bbox": valid_bboxes, "reason": reason}
