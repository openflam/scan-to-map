from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
from pathlib import Path
from process_query import process_query

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


@app.route("/search", methods=["GET", "POST"])
def search():
    """
    Search endpoint that returns a bounding box.
    Uses OpenAI API to find the most relevant component based on the search query.

    The bounding box is transformed to match the format expected by Model3DViewer:
    - x_min = -bbox.min[1]
    - y_min = bbox.min[2]
    - z_min = bbox.min[0]
    - x_max = -bbox.max[1]
    - y_max = bbox.max[2]
    - z_max = bbox.max[0]
    """
    # Get search query
    if request.method == "POST":
        query = request.json.get("query", "") if request.json else ""
    else:
        query = request.args.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Find the most relevant component bounding boxes and reason using OpenAI
    result_data = process_query(query, component_captions, bbox_lookup)
    bboxes = result_data["bbox"]  # This is now a list of bounding boxes
    reason = result_data["reason"]

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
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
