"""Tools module."""

from .bm25_search import search_terms, SEARCH_TERMS_TOOL
from .distance import get_distance, GET_DISTANCE_TOOL

TOOLS = [SEARCH_TERMS_TOOL, GET_DISTANCE_TOOL]

TOOL_FUNCTIONS = {
    "search_terms": search_terms,
    "get_distance": get_distance,
}
