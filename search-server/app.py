from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
from pathlib import Path
import numpy as np
from process_query import process_query
from semantic_search import (
    OpenAIProvider,
    BM25Provider,
    OpenAIRAGProvider,
    CLIPProvider,
)
from routing.path_calculation import calculate_route

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get dataset name from command line argument
if len(sys.argv) < 2:
    print("Usage: python app.py <dataset_name>")
    print("Example: python app.py ArenaLabSemanticNeg")
    sys.exit(1)

DATASET_NAME = sys.argv[1]
print(f"Loading data for dataset: {DATASET_NAME}")

# Load bbox_corners.json on startup
BBOX_DATA_PATH = (
    Path(__file__).parent / ".." / "outputs" / DATASET_NAME / "bbox_corners.json"
)

if not BBOX_DATA_PATH.exists():
    print(f"Error: bbox_corners.json not found at {BBOX_DATA_PATH}")
    sys.exit(1)

with open(BBOX_DATA_PATH, "r") as f:
    bbox_data = json.load(f)

# Load component_captions.json on startup
CAPTIONS_DATA_PATH = (
    Path(__file__).parent / ".." / "outputs" / DATASET_NAME / "component_captions.json"
)

if not CAPTIONS_DATA_PATH.exists():
    print(f"Error: component_captions.json not found at {CAPTIONS_DATA_PATH}")
    sys.exit(1)

with open(CAPTIONS_DATA_PATH, "r") as f:
    captions_list = json.load(f)

# Convert captions list to dictionary keyed by component_id
component_captions = {item["component_id"]: item for item in captions_list}

# Create bbox lookup dictionary keyed by component_id
bbox_lookup = {item["connected_comp_id"]: item["bbox"] for item in bbox_data}

# Initialize providers
print("Initializing search providers...")

FAISS_INDEX_PATH = (
    Path(__file__).parent / ".." / "outputs" / DATASET_NAME / "clip_embeddings.faiss"
)
EMBEDDINGS_NPZ_PATH = (
    Path(__file__).parent / ".." / "outputs" / DATASET_NAME / "clip_embeddings.npz"
)
CLIP_STATS_PATH = (
    Path(__file__).parent
    / ".."
    / "outputs"
    / DATASET_NAME
    / "clip_embedding_stats.json"
)

with open(CLIP_STATS_PATH, "r") as f:
    clip_stats = json.load(f)

clip_model_name = clip_stats["model_name"]
clip_pretrained = clip_stats["pretrained"]

print(f"CLIP model configuration: {clip_model_name} ({clip_pretrained})")

# Load occupancy grid and metadata
OCCUPANCY_GRID_PATH = (
    Path(__file__).parent / ".." / "outputs" / DATASET_NAME / "occupancy_grid.npy"
)
OCCUPANCY_METADATA_PATH = (
    Path(__file__).parent
    / ".."
    / "outputs"
    / DATASET_NAME
    / "occupancy_grid_metadata.json"
)

if OCCUPANCY_GRID_PATH.exists() and OCCUPANCY_METADATA_PATH.exists():
    occupancy_grid = np.load(OCCUPANCY_GRID_PATH)
    with open(OCCUPANCY_METADATA_PATH, "r") as f:
        occupancy_metadata = json.load(f)
    print(f"Loaded occupancy grid: {occupancy_grid.shape}")
else:
    occupancy_grid = None
    occupancy_metadata = None
    print("Warning: Occupancy grid not found. Routing will be disabled.")

openai_provider = OpenAIProvider(component_captions, model="gpt-4o-mini")
bm25_provider = BM25Provider(component_captions)
openai_rag_provider_gpt_5_mini = OpenAIRAGProvider(
    component_captions, model="gpt-5-mini", bm25_top_k=20
)
clip_provider = CLIPProvider(
    faiss_index_path=str(FAISS_INDEX_PATH),
    embeddings_npz_path=str(EMBEDDINGS_NPZ_PATH),
    component_captions=component_captions,
    model_name=clip_model_name,
    pretrained=clip_pretrained,
)

print("Providers initialized successfully")

# Map method names to providers
PROVIDERS = {
    "gpt-4o-mini [Full]": openai_provider,
    "BM25": bm25_provider,
    "gpt-5-mini [RAG]": openai_rag_provider_gpt_5_mini,
    "CLIP ViT-H-14": clip_provider,
}


def transform_coordinates(coords):
    """
    Transform coordinates from COLMAP coordinate system to Model3DViewer coordinate system.

    Args:
        coords: List or tuple of [x, y, z] coordinates

    Returns:
        List of transformed [x, y, z] coordinates
    """
    return [-coords[1], coords[2], coords[0]]


def transform_bbox(bbox):
    """
    Transform bounding box from COLMAP coordinate system to Model3DViewer coordinate system.

    Args:
        bbox: Bounding box dictionary with 'min' and 'max' keys

    Returns:
        Transformed bounding box dictionary with x_min, y_min, z_min, x_max, y_max, z_max
    """
    return {
        "x_min": -bbox["min"][1],
        "y_min": bbox["min"][2],
        "z_min": bbox["min"][0],
        "x_max": -bbox["max"][1],
        "y_max": bbox["max"][2],
        "z_max": bbox["max"][0],
    }


@app.route("/search", methods=["POST"])
def search():
    """
    Search endpoint that returns a bounding box.
    Uses the specified search provider to find the most relevant component based on the search query.

    The bounding box is transformed to match the format expected by Model3DViewer:
    - x_min = -bbox.min[1]
    - y_min = bbox.min[2]
    - z_min = bbox.min[0]
    - x_max = -bbox.max[1]
    - y_max = bbox.max[2]
    - z_max = bbox.max[0]
    """
    # Get search query and method
    search_query = request.json.get("query")
    method = request.json.get("method")

    if not search_query or len(search_query) == 0:
        return jsonify({"error": "No query provided"}), 400

    # Get the appropriate provider based on the method
    provider = PROVIDERS.get(method)
    if provider is None:
        return jsonify({"error": f"Invalid method: {method}"}), 400

    # Unpack the search query (assume only one entry for now)
    query_item = search_query[0]
    query_type = query_item.get("type")
    query_value = query_item.get("value")

    if not query_type or not query_value:
        return jsonify({"error": "Invalid query format"}), 400

    print(f"Processing query type '{query_type}' using method: '{method}'")

    # Validate query type compatibility with provider
    if query_type == "image" and method != "CLIP ViT-H-14":
        return (
            jsonify(
                {"error": f"Image queries are only supported with CLIP ViT-H-14 method"}
            ),
            400,
        )

    # For CLIP, pass the entire query_item; for others, extract text value
    if method == "CLIP ViT-H-14":
        query_input = query_item
    else:
        # Non-CLIP providers only support text
        if query_type != "text":
            return (
                jsonify({"error": f"Method {method} only supports text queries"}),
                400,
            )
        query_input = query_value

    # Find the most relevant component bounding boxes and reason using the selected provider
    result_data = process_query(query_input, component_captions, bbox_lookup, provider)
    bboxes = result_data["bbox"]  # This is now a list of bounding boxes
    reason = result_data["reason"]
    search_time_ms = result_data["search_time_ms"]

    # Transform each bounding box to match the coordinate system
    # used in Model3DViewer (as seen in App.tsx)
    transformed_bboxes = [transform_bbox(bbox) for bbox in bboxes]

    result = {
        "bbox": transformed_bboxes,
        "reason": reason,
        "search_time_ms": search_time_ms,
    }

    return jsonify(result)


@app.route("/get_route", methods=["POST"])
def get_route():
    """
    Route calculation endpoint.
    Takes source and destination search terms and returns a path.
    """
    # Check if occupancy grid is loaded
    if occupancy_grid is None or occupancy_metadata is None:
        return jsonify({"error": "Occupancy grid not available"}), 503

    # Get source, destination, and method from request
    data = request.json
    source = data.get("source")
    destination = data.get("destination")
    method = data.get("method")

    if not source or not destination:
        return jsonify({"error": "Both source and destination are required"}), 400

    if not method:
        return jsonify({"error": "Method is required"}), 400

    # Get the appropriate provider based on the method
    provider = PROVIDERS.get(method)
    if provider is None:
        return jsonify({"error": f"Invalid method: {method}"}), 400

    # Assume source and destination are single queries
    source = source[0] if isinstance(source, list) else source
    destination = destination[0] if isinstance(destination, list) else destination

    # For CLIP, pass the entire source/destination objects; for others, extract text value
    if method == "CLIP ViT-H-14":
        source_input = source
        destination_input = destination
    else:
        # Non-CLIP providers only support text
        source_input = source.get("value") if isinstance(source, dict) else source
        destination_input = (
            destination.get("value") if isinstance(destination, dict) else destination
        )

    print(
        f"Processing route from '{source}' to '{destination}' using method: '{method}'"
    )

    # Process source query
    source_result = process_query(
        source_input, component_captions, bbox_lookup, provider
    )
    source_bboxes = source_result["bbox"]

    # Process destination query
    destination_result = process_query(
        destination_input, component_captions, bbox_lookup, provider
    )
    destination_bboxes = destination_result["bbox"]

    # Use the first bounding box from each result
    if not source_bboxes or not destination_bboxes:
        return jsonify({"error": "Could not find source or destination location"}), 404

    source_bbox = source_bboxes[0]
    destination_bbox = destination_bboxes[0]

    # Calculate centers of bounding boxes
    source_center = tuple(source_bbox["center"])
    destination_center = tuple(destination_bbox["center"])

    # Calculate route
    path = calculate_route(
        source_center, destination_center, occupancy_grid, occupancy_metadata
    )

    # Transform path coordinates to match the coordinate system used in Model3DViewer
    transformed_path = [transform_coordinates(coord) for coord in path]

    # Transform source and destination bboxes to match the format expected by Model3DViewer
    transformed_source_bbox = transform_bbox(source_bbox)
    transformed_destination_bbox = transform_bbox(destination_bbox)

    return jsonify(
        {
            "path": transformed_path,
            "source_bbox": transformed_source_bbox,
            "destination_bbox": transformed_destination_bbox,
            "source_reason": source_result["reason"],
            "destination_reason": destination_result["reason"],
        }
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
