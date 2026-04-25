from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    send_from_directory,
    render_template,
)
from flask_cors import CORS
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import base64

from spatial_db import database
from process_query import process_query
from semantic_search import (
    OpenAIProvider,
    BM25Provider,
    OpenAIRAGProvider,
)
from utils.load_clip import load_clip_provider
from routing.path_calculation import calculate_route
import queue
import threading
from llm_reasoning.llm_agent import LLMAgent, call_tool

STATIC_DIR = Path(__file__).parent / "front-end-build"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app, methods=["GET", "POST", "DELETE", "OPTIONS"])  # Enable CORS for all routes

# Parse command line arguments
parser = argparse.ArgumentParser(description="Search server for scan-to-map")
parser.add_argument(
    "dataset_name",
    nargs="?",
    default=None,
    help=(
        "Optional dataset name. If provided, CLIP will be pre-initialized for this dataset. "
        "If omitted, CLIP is disabled."
    ),
)
args = parser.parse_args()

# CLIP state: pre-initialized at startup for a single dataset (if provided).
# If no dataset name is given, CLIP is disabled.
CLIP_DATASET_NAME = args.dataset_name  # None means CLIP is disabled
clip_provider = None

if CLIP_DATASET_NAME is not None:
    print(f"Pre-initializing CLIP for dataset: {CLIP_DATASET_NAME}")
    OUTPUTS_ROOT = Path(__file__).parent / ".." / "outputs"
    clip_provider = load_clip_provider(CLIP_DATASET_NAME, OUTPUTS_ROOT)
    print("CLIP initialized successfully")
else:
    print(
        "No dataset provided at startup — CLIP disabled. "
        "Pass a dataset name as a positional argument to enable CLIP."
    )


@app.route("/load_mesh", methods=["GET"])
def load_mesh():
    """
    Serves the raw.glb mesh file for the requested dataset.
    """
    dataset_name = request.args.get("dataset_name")
    if not dataset_name:
        return jsonify({"error": "dataset_name query parameter is required"}), 400

    base_dir = Path(__file__).parent / ".."

    # First try: outputs directory
    output_mesh_path = base_dir / "outputs" / dataset_name / "raw.glb"
    if output_mesh_path.exists():
        return send_file(output_mesh_path, mimetype="model/gltf-binary")

    # Second try: old polycam_data directory
    mesh_path = base_dir / "data" / dataset_name / "polycam_data" / "raw.glb"
    if not mesh_path.exists():
        return jsonify({"error": f"Mesh file not found for dataset {dataset_name}"}), 404

    return send_file(mesh_path, mimetype="model/gltf-binary")


@app.route("/get_providers_list", methods=["GET"])
def get_providers_list():
    """
    Returns the list of available search provider names.
    CLIP ViT-H-14 is only listed when it was pre-initialized at startup.
    """
    providers = ["gpt-5-mini [Full]", "BM25", "gpt-5-mini [RAG]", "gpt-5.4-tools"]
    if clip_provider is not None:
        providers.append("CLIP ViT-H-14")
    return jsonify({"providers": providers})


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
        bbox: Bounding box dictionary with 'corners' key

    Returns:
        Transformed bounding box dictionary with 'corners'
    """
    return {
        "corners": [
            [c[1], c[2], c[0]] for c in bbox.get("corners", [])
        ]
    }


def initialize_provider(method, dataset_name):
    """
    Initialize and return the appropriate search provider for the given method.

    Non-CLIP providers (OpenAI, BM25) are cheap to initialize and are created
    fresh per request from the given dataset_name.

    CLIP is expensive to initialize, so it is pre-loaded at startup for a single
    dataset.  It is only returned here when the requested dataset_name matches
    CLIP_DATASET_NAME and the provider was successfully initialized at startup.
    """
    if method == "gpt-5-mini [Full]":
        return OpenAIProvider(dataset_name, model="gpt-5-mini")
    elif method == "BM25":
        return BM25Provider(dataset_name)
    elif method == "gpt-5-mini [RAG]":
        return OpenAIRAGProvider(dataset_name, model="gpt-5-mini", bm25_top_k=20)
    elif method == "CLIP ViT-H-14":
        if clip_provider is None:
            return None  # CLIP was not initialized at startup
        if dataset_name != CLIP_DATASET_NAME:
            return None  # CLIP is only available for the startup dataset
        return clip_provider
    elif method == "gpt-5.4-tools":
        return "streaming"  # Return a dummy string so it's not None
    return None


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
    # Get search query, method, and dataset_name
    dataset_name = request.json.get("dataset_name")
    search_query = request.json.get("query")
    method = request.json.get("method")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400

    if not search_query or len(search_query) == 0:
        return jsonify({"error": "No query provided"}), 400

    if not database.check_dataset_exists(dataset_name):
        return (
            jsonify({"error": f"Dataset '{dataset_name}' not found in database."}),
            404,
        )

    # Initialize the appropriate provider
    provider = initialize_provider(method, dataset_name)
    if provider is None:
        if method == "CLIP ViT-H-14":
            return (
                jsonify(
                    {
                        "error": "CLIP is not available for this dataset. Start the server with the dataset name to enable CLIP."
                    }
                ),
                400,
            )
        return jsonify({"error": f"Invalid method: {method}"}), 400

    # Unpack the search query (assume only one entry for now)
    query_item = search_query[0]
    query_type = query_item.get("type")
    query_value = query_item.get("value")

    if not query_type or not query_value:
        return jsonify({"error": "Invalid query format"}), 400

    print(f"Processing query type '{query_type}' using method: '{method}'")

    # Validate query type compatibility with provider
    if query_type == "image" and clip_provider is None:
        return (
            jsonify(
                {
                    "error": "Image queries are not available: CLIP was not initialized at startup"
                }
            ),
            400,
        )

    if query_type == "image" and method != "CLIP ViT-H-14":
        return (
            jsonify(
                {"error": "Image queries are only supported with CLIP ViT-H-14 method"}
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

    if method == "gpt-5.4-tools":
        return jsonify({"error": "Use /search_stream for gpt-5.4-tools"}), 400

    # Find the most relevant component bounding boxes and reason using the selected provider
    result_data = process_query(query_input, dataset_name, provider)
    bboxes = result_data["bbox"]  # This is now a list of bounding boxes
    component_ids = result_data["component_ids"]  # List of component IDs
    reason = result_data["reason"]
    search_time_ms = result_data["search_time_ms"]

    # Fetch fresh captions from the database for the returned component IDs
    if component_ids:
        caption_rows = database.fetch_components_by_ids(dataset_name, component_ids)
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


@app.route("/search_stream", methods=["POST"])
def search_stream():
    """
    Search endpoint that streams reasoning and search results using Server-Sent Events (SSE).
    Currently only supports the gpt-5.4-tools method.
    """
    # Get search query, method, and dataset_name
    dataset_name = request.json.get("dataset_name")
    search_query = request.json.get("query")
    method = request.json.get("method")
    tools = request.json.get("tools")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400

    if not search_query or len(search_query) == 0:
        return jsonify({"error": "No query provided"}), 400

    if not database.check_dataset_exists(dataset_name):
        return (
            jsonify({"error": f"Dataset '{dataset_name}' not found in database."}),
            404,
        )

    # Unpack the search query (assume only one entry for now)
    query_item = search_query[0]
    query_type = query_item.get("type")

    if query_type != "text":
        return jsonify({"error": "Streaming only supports text queries"}), 400

    if method != "gpt-5.4-tools":
        return (
            jsonify({"error": "Streaming currently only supports gpt-5.4-tools"}),
            400,
        )

    query_input = query_item.get("value")
    if not query_input:
        return jsonify({"error": "No query string provided for gpt-5.4-tools"}), 400

    q = queue.Queue()

    def on_stream_event(event):
        q.put({"type": "event", "data": event})

    def run_agent():
        try:
            agent = LLMAgent(model="gpt-5.4", allowed_tools=tools)
            result = agent.answer_query_stream(
                query=query_input,
                dataset_name=dataset_name,
                on_stream_event=on_stream_event,
            )
            q.put({"type": "result", "data": result})
        except Exception as e:
            q.put({"type": "error", "error": str(e)})

    t = threading.Thread(target=run_agent)
    t.start()

    def generate():
        import time

        start_time = time.perf_counter()
        while True:
            item = q.get()
            if item["type"] == "event":
                yield f"data: {json.dumps(item['data'])}\n\n"
            elif item["type"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'error': item['error']})}\n\n"
                break
            elif item["type"] == "result":
                result = item["data"]
                end_time = time.perf_counter()
                search_time_ms = (end_time - start_time) * 1000

                component_ids = result.get("component_ids", [])
                reason = result.get("reason", "")

                valid_bboxes = []
                valid_component_ids = []
                invalid_ids = []

                if component_ids:
                    rows = database.fetch_components_by_ids(dataset_name, component_ids)

                    bbox_map = {}
                    for row in rows:
                        try:
                            bbox = (
                                json.loads(row["bbox_json"]) if row["bbox_json"] else {}
                            )
                        except json.JSONDecodeError:
                            bbox = {}
                        bbox_map[row["component_id"]] = {
                            "bbox": bbox,
                            "caption": row["caption"],
                        }

                    for comp_id in component_ids:
                        if comp_id in bbox_map:
                            valid_bboxes.append(bbox_map[comp_id]["bbox"])
                            valid_component_ids.append(comp_id)
                        else:
                            invalid_ids.append(comp_id)

                if not valid_bboxes:
                    print(
                        "Warning: No valid component IDs found for gpt-5.4-tools. Using first component."
                    )
                    row = database.fetch_first_component(dataset_name)
                    if row:
                        try:
                            bbox = (
                                json.loads(row["bbox_json"]) if row["bbox_json"] else {}
                            )
                        except json.JSONDecodeError:
                            bbox = {}
                        valid_bboxes = [bbox]
                        valid_component_ids = [row["component_id"]]
                        bbox_map = {row["component_id"]: {"caption": row["caption"]}}

                # Build components array with transformed bboxes and captions
                components = []
                for bbox, comp_id in zip(valid_bboxes, valid_component_ids):
                    transformed_bbox = transform_bbox(bbox)
                    caption = (
                        bbox_map.get(comp_id, {}).get("caption")
                        or "No caption available"
                    )
                    components.append(
                        {
                            "bbox": transformed_bbox,
                            "caption": caption,
                            "component_id": str(comp_id),
                        }
                    )

                final_result = {
                    "reason": reason,
                    "search_time_ms": search_time_ms,
                    "components": components,
                }

                yield f"data: {json.dumps({'type': 'result', 'data': final_result})}\n\n"
                break

    return app.response_class(generate(), mimetype="text/event-stream")


@app.route("/get_route", methods=["POST"])
def get_route():
    """
    Route calculation endpoint.
    Takes source and destination search terms and returns a path.
    """
    data = request.json
    dataset_name = data.get("dataset_name")
    source = data.get("source")
    destination = data.get("destination")
    method = data.get("method")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400

    if not source or not destination:
        return jsonify({"error": "Both source and destination are required"}), 400

    if not method:
        return jsonify({"error": "Method is required"}), 400

    if not database.check_dataset_exists(dataset_name):
        return (
            jsonify({"error": f"Dataset '{dataset_name}' not found in database."}),
            404,
        )

    # Load occupancy grid and metadata on demand
    dataset_dir = Path(__file__).parent / ".." / "outputs" / dataset_name
    occupancy_grid_path = dataset_dir / "occupancy_grid.npy"
    occupancy_metadata_path = dataset_dir / "occupancy_grid_metadata.json"
    floor_height_path = dataset_dir / "floor_height.json"

    if not occupancy_grid_path.exists() or not occupancy_metadata_path.exists():
        return jsonify({"error": "Occupancy grid not available"}), 503

    occupancy_grid = np.load(occupancy_grid_path)
    with open(occupancy_metadata_path, "r") as f:
        occupancy_metadata = json.load(f)
    floor_height_file = str(floor_height_path) if floor_height_path.exists() else None

    # Initialize the appropriate provider
    provider = initialize_provider(method, dataset_name)
    if provider is None:
        if method == "CLIP ViT-H-14":
            return (
                jsonify(
                    {
                        "error": "CLIP is not available for this dataset. Start the server with the dataset name to enable CLIP."
                    }
                ),
                400,
            )
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
    source_result = process_query(source_input, dataset_name, provider)
    source_bboxes = source_result["bbox"]

    # Process destination query
    destination_result = process_query(destination_input, dataset_name, provider)
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
    bbox must be in the format: {"corners": [[x, y, z], ...]}
    """
    data = request.json
    dataset_name = data.get("dataset_name")
    component_id = data.get("component_id")
    new_caption = data.get("caption")
    new_bbox = data.get("bbox")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400

    if not database.check_dataset_exists(dataset_name):
        return (
            jsonify({"error": f"Dataset '{dataset_name}' not found in database."}),
            404,
        )

    if not component_id:
        return jsonify({"error": "No component_id provided"}), 400

    if new_caption is None and new_bbox is None:
        return (
            jsonify({"error": "At least one of 'caption' or 'bbox' must be provided"}),
            400,
        )

    if new_bbox is not None and "corners" not in new_bbox:
        return jsonify({"error": "bbox must have 'corners' key"}), 400

    try:
        comp_id_int = int(component_id)
    except (ValueError, TypeError):
        return jsonify({"error": f"Invalid component_id: {component_id}"}), 400

    updated = database.update_component(
        dataset_name, comp_id_int, new_caption, new_bbox
    )

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
    dataset_name = data.get("dataset_name")
    component_id = data.get("component_id")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400

    if not component_id:
        return jsonify({"error": "No component_id provided"}), 400

    if not database.check_dataset_exists(dataset_name):
        return (
            jsonify({"error": f"Dataset '{dataset_name}' not found in database."}),
            404,
        )

    try:
        comp_id_int = int(component_id)
    except (ValueError, TypeError):
        return jsonify({"error": f"Invalid component_id: {component_id}"}), 400

    deleted = database.delete_component(dataset_name, comp_id_int)

    if deleted == 0:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    return jsonify({"component_id": component_id, "deleted": True})


@app.route("/get_component_info", methods=["GET"])
def get_component_info():
    """
    Get component information endpoint.
    Returns the caption and corresponding crop image with highest fraction_visible for a given component ID.
    """
    dataset_name = request.args.get("dataset_name")
    component_id = request.args.get("component_id")

    if not dataset_name:
        return jsonify({"error": "dataset_name query parameter is required"}), 400

    if not component_id:
        return jsonify({"error": "No component_id provided"}), 400

    if not database.check_dataset_exists(dataset_name):
        return (
            jsonify({"error": f"Dataset '{dataset_name}' not found in database."}),
            404,
        )

    # Query the database for caption and best_crop
    try:
        comp_id_int = int(component_id)
    except ValueError:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    row = database.fetch_component_info(dataset_name, comp_id_int)

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

    # Read the crop image file and encode as base64
    image_path = (
        Path(__file__).parent
        / ".."
        / "outputs"
        / dataset_name
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
        "bbox": {"corners": [[x, y, z], ...]}
      },
      ...
    ]
    """
    dataset_name = request.args.get("dataset_name")
    if not dataset_name:
        return jsonify({"error": "dataset_name query parameter is required"}), 400

    if not database.check_dataset_exists(dataset_name):
        return (
            jsonify({"error": f"Dataset '{dataset_name}' not found in database."}),
            404,
        )

    rows = database.fetch_all_components(dataset_name)

    result = []
    for row in rows:
        if row.get("bbox_json"):
            try:
                bbox = json.loads(row["bbox_json"])
            except json.JSONDecodeError:
                bbox = {}
        else:
            bbox = {}

        result.append(
            {
                "connected_comp_id": row["component_id"],
                "bbox": bbox,
            }
        )

    response = jsonify(result)
    response.headers["Content-Disposition"] = "attachment; filename=bbox_corners.json"
    return response


@app.route("/call_tool", methods=["POST"])
def call_tool_route():
    """
    Endpoint to explicitly execute a specific reasoning tool.
    Expects a JSON payload with 'tool_name' and optionally 'arguments' and 'dataset_name'.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    tool_name = data.get("tool_name")
    arguments = data.get("arguments", {})
    dataset_name = data.get("dataset_name")

    if not tool_name:
        return jsonify({"error": "tool_name is required"}), 400

    result = call_tool(tool_name, arguments, dataset_name=dataset_name)
    return jsonify({"result": result})


@app.route("/")
def index():
    """
    Root route.
    If a dataset_name query parameter is supplied, serve the Vite-built React app.
    Otherwise, render the dataset picker page using the datasets currently loaded
    into the PostGIS database.
    """
    dataset_name = request.args.get("dataset_name")

    if dataset_name:
        # A dataset was chosen — hand off to the React SPA
        if STATIC_DIR.exists():
            return send_from_directory(str(STATIC_DIR), "index.html")
        return (
            jsonify(
                {
                    "error": "Frontend not built. Run: npm run build in semantic-3d-search-demo/"
                }
            ),
            404,
        )

    # No dataset selected — show the picker
    datasets = []
    outputs_root = Path(__file__).parent / ".." / "outputs"
    if outputs_root.exists():
        db_tables = set(database.list_dataset_tables())
        for directory in outputs_root.iterdir():
            if directory.is_dir() and database.get_table_name(directory.name) in db_tables:
                datasets.append(directory.name)
    datasets.sort()

    return render_template("dataset_picker.html", datasets=datasets)


@app.route("/benchmark_collection")
def benchmark_collection():
    """
    Benchmark route.
    If a dataset_name query parameter is supplied, serve the benchmark.html file.
    Otherwise, render the dataset picker page using the datasets currently loaded
    into the PostGIS database.
    """
    dataset_name = request.args.get("dataset_name")

    if dataset_name:
        if STATIC_DIR.exists():
            return send_from_directory(str(STATIC_DIR), "benchmark.html")
        return (
            jsonify(
                {
                    "error": "Frontend not built. Run: npm run build in semantic-3d-search-demo/"
                }
            ),
            404,
        )

    datasets = []
    outputs_root = Path(__file__).parent / ".." / "outputs"
    if outputs_root.exists():
        db_tables = set(database.list_dataset_tables())
        for directory in outputs_root.iterdir():
            if directory.is_dir() and database.get_table_name(directory.name) in db_tables:
                datasets.append(directory.name)
    datasets.sort()

    return render_template("dataset_picker.html", datasets=datasets, target_url="/benchmark_collection")


@app.route("/save_benchmark", methods=["POST"])
def save_benchmark():
    import time
    import uuid
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    benchmark_dir = Path(__file__).parent / ".." / "benchmark" / "data"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    benchmark_name = data.get("benchmark_name")
    if benchmark_name:
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', benchmark_name)
        if not safe_name.endswith('.json'):
            safe_name += '.json'
        filename = safe_name
    else:
        filename = f"benchmark_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.json"

    filepath = benchmark_dir / filename

    try:
        import os
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        os.chmod(filepath, 0o666)
        return jsonify({"success": True, "file": str(filename)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_benchmark_names", methods=["GET"])
def get_benchmark_names():
    benchmark_dir = Path(__file__).parent / ".." / "benchmark" / "data"
    if not benchmark_dir.exists():
        return jsonify({"names": []})

    names = []
    for file in benchmark_dir.glob("*.json"):
        names.append(file.name[:-5])
        
    return jsonify({"names": sorted(names)})

@app.route("/get_benchmark", methods=["GET"])
def get_benchmark():
    name = request.args.get("name")
    if not name:
        return jsonify({"error": "name parameter is required"}), 400

    benchmark_dir = Path(__file__).parent / ".." / "benchmark" / "data"
    
    # We apply same filtering logic as saving
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    filepath = benchmark_dir / f"{safe_name}.json"
    
    if not filepath.exists():
        return jsonify({"error": "Benchmark not found"}), 404
        
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/<path:path>")
def serve_frontend(path: str):
    """
    Catch-all that serves the Vite-built React app for all non-API paths.
    """
    if STATIC_DIR.exists():
        return send_from_directory(str(STATIC_DIR), "index.html")
    return (
        jsonify(
            {
                "error": "Frontend not built. Run: npm run build in semantic-3d-search-demo/"
            }
        ),
        404,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
