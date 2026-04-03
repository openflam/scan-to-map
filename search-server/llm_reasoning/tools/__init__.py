"""Tools module."""

from .bm25_search import search_terms, SEARCH_TERMS_TOOL, SEARCH_THINKING_TEXT
from .distance import get_distance, GET_DISTANCE_TOOL, GET_DISTANCE_THINKING_TEXT
from .search_around import search_around_component, SEARCH_AROUND_COMPONENT_TOOL, SEARCH_AROUND_COMPONENT_THINKING_TEXT
from .image import get_images, GET_IMAGES_TOOL, GET_IMAGES_THINKING_TEXT

TOOLS = [SEARCH_TERMS_TOOL, GET_DISTANCE_TOOL, SEARCH_AROUND_COMPONENT_TOOL, GET_IMAGES_TOOL]

TOOL_FUNCTIONS = {
    "search_terms": search_terms,
    "get_distance": get_distance,
    "search_around_component": search_around_component,
    "get_images": get_images,
}

THINKING_TEXTS = {
    "search_terms": SEARCH_THINKING_TEXT,
    "get_distance": GET_DISTANCE_THINKING_TEXT,
    "search_around_component": SEARCH_AROUND_COMPONENT_THINKING_TEXT,
    "get_images": GET_IMAGES_THINKING_TEXT,
}

