"""Distance tool."""

from __future__ import annotations

import json
import math

from spatial_db import database
from .utils import _get_dataset_name


def get_distance(
    component_id_1: int,
    component_id_2: int,
    dataset_name: str | None = None,
) -> float:
    """
    Calculate the Euclidean distance between the centers of two components.

    Args:
            component_id_1: The ID of the first component.
            component_id_2: The ID of the second component.
            dataset_name: Dataset to search; falls back to DATASET_NAME env var.

    Returns:
            The distance in meters as a float.
    """
    resolved_dataset = _get_dataset_name(dataset_name)
    components = database.fetch_components_by_ids(resolved_dataset, [component_id_1, component_id_2])

    if len(components) != 2:
        raise ValueError("Could not find both components in the database.")

    comp1 = next((c for c in components if c["component_id"] == component_id_1), None)
    comp2 = next((c for c in components if c["component_id"] == component_id_2), None)

    if not comp1 or not comp2:
        raise ValueError("Could not find both components in the database.")

    try:
        bbox1 = json.loads(comp1["bbox_json"]) if comp1["bbox_json"] else {}
        bbox2 = json.loads(comp2["bbox_json"]) if comp2["bbox_json"] else {}
    except json.JSONDecodeError:
        raise ValueError("Invalid bbox JSON in the database.")

    if "min" not in bbox1 or "max" not in bbox1 or "min" not in bbox2 or "max" not in bbox2:
        raise ValueError("Missing min/max in component bounding box.")

    center1 = [
        (bbox1["min"][0] + bbox1["max"][0]) / 2.0,
        (bbox1["min"][1] + bbox1["max"][1]) / 2.0,
        (bbox1["min"][2] + bbox1["max"][2]) / 2.0,
    ]
    center2 = [
        (bbox2["min"][0] + bbox2["max"][0]) / 2.0,
        (bbox2["min"][1] + bbox2["max"][1]) / 2.0,
        (bbox2["min"][2] + bbox2["max"][2]) / 2.0,
    ]

    distance = math.sqrt(
        (center1[0] - center2[0]) ** 2 +
        (center1[1] - center2[1]) ** 2 +
        (center1[2] - center2[2]) ** 2
    )

    return distance


GET_DISTANCE_TOOL = {
    "type": "function",
    "name": "get_distance",
    "description": "Calculate the Euclidean distance between the centers of two components' bounding boxes in meters.",
    "parameters": {
        "type": "object",
        "properties": {
            "component_id_1": {
                "type": "integer",
                "description": "The ID of the first component.",
            },
            "component_id_2": {
                "type": "integer",
                "description": "The ID of the second component.",
            },
        },
        "required": ["component_id_1", "component_id_2"],
        "additionalProperties": False,
    },
}

GET_DISTANCE_THINKING_TEXT = "Calculating distance between objects..."

