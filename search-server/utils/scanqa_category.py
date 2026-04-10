"""
ScanQA simplified question category (same rules as ScanQA/scripts/scanqa_simplified_categories.py).
Used to pick answer vocabulary and optional prompt suffix for OpenAI search.
"""

from __future__ import annotations

import re

CATEGORY_ORDER = (
    "object_retrieval",
    "location",
    "color",
    "number",
    "others",
)


def bucket(question: str) -> str:
    """
    Assign exactly one of:
    object_retrieval | location | color | number | others
    """
    s = question.lower()

    if "what color" in s or "what is the color" in s or "what's the color" in s:
        return "color"

    if "how many" in s:
        return "number"

    if "what side" in s or re.search(r"\bwhere\b", s):
        return "location"

    object_phrases = (
        "what object",
        "what sits",
        "what can",
        "what are",
        "what is",
    )
    for phrase in object_phrases:
        if phrase in s:
            return "object_retrieval"

    return "others"
