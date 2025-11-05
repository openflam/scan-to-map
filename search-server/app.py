from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
from pathlib import Path
from process_query import process_query
from semantic_search import (
    OpenAIProvider,
    BM25Provider,
    OpenAIRAGProvider,
    CLIPProvider,
)

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

clip_model_name = clip_stats.get("model_name", "ViT-B-32")
clip_pretrained = clip_stats.get("pretrained", "laion2b_s34b_b79k")

print(f"CLIP model configuration: {clip_model_name} ({clip_pretrained})")

openai_provider = OpenAIProvider(component_captions, model="gpt-4o-mini")
bm25_provider = BM25Provider(component_captions)
openai_rag_provider = OpenAIRAGProvider(
    component_captions, model="gpt-4o-mini", bm25_top_k=20
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
    "gpt-4o-mini [RAG]": openai_rag_provider,
    "CLIP ViT-H-14": clip_provider,
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
    transformed_bboxes = []
    for bbox in bboxes:
        transformed_bbox = {
            "x_min": -bbox["min"][1],
            "y_min": bbox["min"][2],
            "z_min": bbox["min"][0],
            "x_max": -bbox["max"][1],
            "y_max": bbox["max"][2],
            "z_max": bbox["max"][0],
        }
        transformed_bboxes.append(transformed_bbox)

    result = {
        "bbox": transformed_bboxes,
        "reason": reason,
        "search_time_ms": search_time_ms,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
