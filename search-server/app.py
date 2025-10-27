from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from pathlib import Path
from process_query import process_query

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load bbox_corners.json on startup
BBOX_DATA_PATH = (
    Path(__file__).parent / ".." / "outputs" / "PrintersNoNeg" / "bbox_corners.json"
)

with open(BBOX_DATA_PATH, "r") as f:
    bbox_data = json.load(f)

# Load component_captions.json on startup
CAPTIONS_DATA_PATH = (
    Path(__file__).parent
    / ".."
    / "outputs"
    / "PrintersNoNeg"
    / "component_captions.json"
)

with open(CAPTIONS_DATA_PATH, "r") as f:
    captions_list = json.load(f)

# Convert captions list to dictionary keyed by component_id
component_captions = {item["component_id"]: item for item in captions_list}


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

    # Find the most relevant component bounding box using OpenAI
    bbox = process_query(query, component_captions)

    # Transform the bounding box to match the coordinate system
    # used in Model3DViewer (as seen in App.tsx)
    result = {
        "x_min": -bbox["min"][1],
        "y_min": bbox["min"][2],
        "z_min": bbox["min"][0],
        "x_max": -bbox["max"][1],
        "y_max": bbox["max"][2],
        "z_max": bbox["max"][0],
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
