from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import base64
from process_query import process_query
from semantic_search import (
    OpenAIProvider,
    BM25Provider,
    OpenAIRAGProvider,
    CLIPProvider,
)
from routing.path_calculation import calculate_route

app = Flask(__name__)
CORS(app, methods=["GET", "POST", "DELETE", "OPTIONS"])  # Enable CORS for all routes

# Get dataset name from command line argument
if len(sys.argv) < 2:
    print("Usage: python app.py <dataset_name>")
    print("Example: python app.py ArenaLabSemanticNeg")
    sys.exit(1)

DATASET_NAME = sys.argv[1]
print(f"Loading data for dataset: {DATASET_NAME}")

# Load component data from SQLite database
DB_PATH = Path(__file__).parent / ".." / "outputs" / DATASET_NAME / "components.db"

if not DB_PATH.exists():
    print(f"Error: components.db not found at {DB_PATH}")
    print(f"Run: python create_database.py {DATASET_NAME}")
    sys.exit(1)

con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row
cur = con.cursor()
cur.execute("SELECT component_id, caption, bbox_json FROM components")
rows = cur.fetchall()
con.close()

# Build component_captions dict for the CLIP provider (which still uses an in-memory dict)
component_captions = {}
for row in rows:
    comp_id = row["component_id"]
    bbox = json.loads(row["bbox_json"])
    component_captions[comp_id] = {
        "component_id": comp_id,
        "caption": row["caption"],
        "bbox": bbox,
    }

print(f"Loaded {len(component_captions)} components from database")

# Load manifest.json on startup
MANIFEST_PATH = (
    Path(__file__).parent / ".." / "outputs" / DATASET_NAME / "crops" / "manifest.json"
)

if not MANIFEST_PATH.exists():
    print(f"Error: manifest.json not found at {MANIFEST_PATH}")
    sys.exit(1)

with open(MANIFEST_PATH, "r") as f:
    manifest_data = json.load(f)

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
FLOOR_HEIGHT_PATH = (
    Path(__file__).parent / ".." / "outputs" / DATASET_NAME / "floor_height.json"
)

if OCCUPANCY_GRID_PATH.exists() and OCCUPANCY_METADATA_PATH.exists():
    occupancy_grid = np.load(OCCUPANCY_GRID_PATH)
    with open(OCCUPANCY_METADATA_PATH, "r") as f:
        occupancy_metadata = json.load(f)
    print(f"Loaded occupancy grid: {occupancy_grid.shape}")

    # Load floor height if available
    floor_height_file = str(FLOOR_HEIGHT_PATH) if FLOOR_HEIGHT_PATH.exists() else None
    if floor_height_file:
        print(f"Loaded floor height file: {FLOOR_HEIGHT_PATH}")
    else:
        print("Warning: Floor height file not found. Using default z-coordinates.")
else:
    occupancy_grid = None
    occupancy_metadata = None
    floor_height_file = None
    print("Warning: Occupancy grid not found. Routing will be disabled.")

openai_provider = OpenAIProvider(str(DB_PATH), model="gpt-5-mini")
bm25_provider = BM25Provider(str(DB_PATH))
openai_rag_provider_gpt_5_mini = OpenAIRAGProvider(
    str(DB_PATH), model="gpt-5-mini", bm25_top_k=20
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
    "gpt-5-mini [Full]": openai_provider,
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
    return [coords[1], coords[2], coords[0]]


def transform_bbox(bbox):
    """
    Transform bounding box from COLMAP coordinate system to Model3DViewer coordinate system.

    Args:
        bbox: Bounding box dictionary with 'min' and 'max' keys

    Returns:
        Transformed bounding box dictionary with x_min, y_min, z_min, x_max, y_max, z_max
    """
    return {
        "x_min": bbox["min"][1],
        "y_min": bbox["min"][2],
        "z_min": bbox["min"][0],
        "x_max": bbox["max"][1],
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
    result_data = process_query(query_input, str(DB_PATH), provider)
    bboxes = result_data["bbox"]  # This is now a list of bounding boxes
    component_ids = result_data["component_ids"]  # List of component IDs
    reason = result_data["reason"]
    search_time_ms = result_data["search_time_ms"]

    # Fetch fresh captions from the database for the returned component IDs
    if component_ids:
        placeholders = ",".join("?" * len(component_ids))
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            f"SELECT component_id, caption FROM components WHERE component_id IN ({placeholders})",
            component_ids,
        )
        caption_rows = cur.fetchall()
        con.close()
        id_to_caption = {row["component_id"]: row["caption"] for row in caption_rows}
    else:
        id_to_caption = {}

    # Build components array with transformed bboxes and captions
    components = []
    for bbox, comp_id in zip(bboxes, component_ids):
        transformed_bbox = transform_bbox(bbox)
        caption = id_to_caption.get(comp_id, "No caption available")
        # Ensure component_id is returned as string for consistency with query parameters
        components.append(
            {"bbox": transformed_bbox, "caption": caption, "component_id": str(comp_id)}
        )

    result = {
        "reason": reason,
        "search_time_ms": search_time_ms,
        "components": components,
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
    source_result = process_query(source_input, str(DB_PATH), provider)
    source_bboxes = source_result["bbox"]

    # Process destination query
    destination_result = process_query(destination_input, str(DB_PATH), provider)
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
        source_center,
        destination_center,
        occupancy_grid,
        occupancy_metadata,
        floor_height_file,
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


@app.route("/update_component", methods=["POST"])
def update_component():
    """
    Update component endpoint.
    Updates the caption and/or bounding box for a given component ID in the database.
    At least one of 'caption' or 'bbox' must be provided.
    bbox must be in the format: {"min": [x, y, z], "max": [x, y, z]}
    """
    data = request.json
    component_id = data.get("component_id")
    new_caption = data.get("caption")
    new_bbox = data.get("bbox")

    if not component_id:
        return jsonify({"error": "No component_id provided"}), 400

    if new_caption is None and new_bbox is None:
        return (
            jsonify({"error": "At least one of 'caption' or 'bbox' must be provided"}),
            400,
        )

    if new_bbox is not None and ("min" not in new_bbox or "max" not in new_bbox):
        return jsonify({"error": "bbox must have 'min' and 'max' keys"}), 400

    try:
        comp_id_int = int(component_id)
    except (ValueError, TypeError):
        return jsonify({"error": f"Invalid component_id: {component_id}"}), 400

    fields, params = [], []
    if new_caption is not None:
        fields.append("caption = ?")
        params.append(new_caption)
    if new_bbox is not None:
        fields.append("bbox_json = ?")
        params.append(json.dumps(new_bbox))
    params.append(comp_id_int)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        f"UPDATE components SET {', '.join(fields)} WHERE component_id = ?",
        params,
    )
    con.commit()
    updated = cur.rowcount
    con.close()

    if updated == 0:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    result = {"component_id": component_id}
    if new_caption is not None:
        result["caption"] = new_caption
    if new_bbox is not None:
        result["bbox"] = new_bbox
    return jsonify(result)


@app.route("/delete_component", methods=["DELETE"])
def delete_component():
    """
    Delete component endpoint.
    Deletes the component with the given component ID from the database.
    """
    data = request.json
    component_id = data.get("component_id")

    if not component_id:
        return jsonify({"error": "No component_id provided"}), 400

    try:
        comp_id_int = int(component_id)
    except (ValueError, TypeError):
        return jsonify({"error": f"Invalid component_id: {component_id}"}), 400

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM components WHERE component_id = ?", (comp_id_int,))
    con.commit()
    deleted = cur.rowcount
    con.close()

    if deleted == 0:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    return jsonify({"component_id": component_id, "deleted": True})


@app.route("/get_component_info", methods=["GET"])
def get_component_info():
    """
    Get component information endpoint.
    Returns the caption and corresponding crop image with highest fraction_visible for a given component ID.
    """
    component_id = request.args.get("component_id")

    if not component_id:
        return jsonify({"error": "No component_id provided"}), 400

    # Query the database for caption and best_crop
    try:
        comp_id_int = int(component_id)
    except ValueError:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "SELECT caption, best_crop FROM components WHERE component_id = ?",
        (comp_id_int,),
    )
    row = cur.fetchone()
    con.close()

    if row is None:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    caption = row["caption"] or "No caption available"
    crop_filename = row["best_crop"]

    if not crop_filename:
        return jsonify(
            {
                "component_id": component_id,
                "caption": caption,
                "image": None,
                "message": "No crops available for this component",
            }
        )

    # Look up the crop metadata in the manifest using the stored filename
    crop_meta = {}
    component_crops = manifest_data.get(component_id, {}).get("crops", [])
    for crop in component_crops:
        if crop.get("crop_filename") == crop_filename:
            crop_meta = crop
            break

    # Read the crop image file and encode as base64
    image_path = (
        Path(__file__).parent
        / ".."
        / "outputs"
        / DATASET_NAME
        / "crops"
        / f"component_{component_id}"
        / crop_filename
    )

    image_base64 = None
    if image_path.exists():
        try:
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            print(f"Error reading crop image {image_path}: {e}")
    else:
        print(f"Warning: Crop image not found at {image_path}")

    return jsonify(
        {
            "component_id": component_id,
            "caption": caption,
            "crop_filename": crop_filename,
            "image_base64": image_base64,
            "fraction_visible": crop_meta.get("fraction_visible"),
            "source_image": crop_meta.get("source_image"),
            "crop_coordinates": crop_meta.get("crop_coordinates"),
        }
    )


@app.route("/download_all_components", methods=["GET"])
def download_all_components():
    """
    Download all components endpoint.
    Returns a JSON array in the bbox_corners.json format expected by App.tsx:
    [
      {
        "connected_comp_id": <int>,
        "bbox": {"min": [x, y, z], "max": [x, y, z]}
      },
      ...
    ]
    """
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT component_id, bbox_json FROM components")
    rows = cur.fetchall()
    con.close()

    result = []
    for row in rows:
        bbox = json.loads(row["bbox_json"])
        result.append(
            {
                "connected_comp_id": row["component_id"],
                "bbox": bbox,
            }
        )

    response = jsonify(result)
    response.headers["Content-Disposition"] = "attachment; filename=bbox_corners.json"
    return response


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
