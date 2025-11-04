"""Prompt templates for semantic search."""


def build_component_matching_prompt(query: str, components_text: str) -> str:
    """
    Build a prompt for matching a search query to component descriptions.

    Args:
        query: The search query string
        components_text: Formatted text containing all component descriptions

    Returns:
        The complete prompt string
    """
    return f"""Given the following search query and a list of object descriptions, determine which component IDs best match the query. 

Only return those components that contain the objects almost exclusively without unrelated objects.

Search Query: "{query}"

Available Components:
{components_text}

Respond with ONLY a JSON object containing:
1. "component_ids": a comma-separated string of integer IDs of ALL components that match (e.g., "2,5,7" or "3" for a single match)
2. "reason": a brief one-sentence explanation of why these components match the query

Example response format:
{{"component_ids": "2,5,7", "reason": "These components contain printers which match the search query."}}

If multiple components match, include all of them. If no component matches well, choose the component that is closest to the query."""


SYSTEM_PROMPT = """You are a helpful assistant that matches search queries to object descriptions. Always respond with a valid JSON object containing component_ids (comma-separated string of integers) and reason (string)."""
