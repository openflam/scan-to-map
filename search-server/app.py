from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load bbox_corners.json on startup
BBOX_DATA_PATH = (
    Path(__file__).parent / ".." / "outputs" / "PrintersNoNeg" / "bbox_corners.json"
)

with open(BBOX_DATA_PATH, "r") as f:
    bbox_data = json.load(f)


@app.route("/search", methods=["GET", "POST"])
def search():
    """
    Search endpoint that returns a bounding box.
    For testing, returns the 3rd component (index 2) from bbox_corners.json.

    The bounding box is transformed to match the format expected by Model3DViewer:
    - x_min = -bbox.min[1]
    - y_min = bbox.min[2]
    - z_min = bbox.min[0]
    - x_max = -bbox.max[1]
    - y_max = bbox.max[2]
    - z_max = bbox.max[0]
    """
    # Get search query (not used for testing, but included for future use)
    if request.method == "POST":
        query = request.json.get("query", "")
    else:
        query = request.args.get("query", "")

    # For testing: return the 3rd component (index 2)
    component = bbox_data[2]
    bbox = component["bbox"]

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
