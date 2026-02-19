"""
create_database.py

Reads component_captions.json and bbox_corners.json from an
outputs/<dataset_name> directory and creates a SQLite database
(components.db) in that same directory.

Schema
------
components
    component_id   INTEGER  PRIMARY KEY
    caption        TEXT
    bbox_json      TEXT     -- full bbox object stored as JSON
    best_crop      TEXT     -- crop_filename with highest fraction_visible from manifest.json

Usage
-----
    python create_database.py <dataset_name>
    python create_database.py ArenaLabSemanticNeg
"""

import json
import sqlite3
import sys
from pathlib import Path


def create_database(dataset_name: str) -> Path:
    outputs_dir = Path(__file__).parent / ".." / "outputs" / dataset_name

    captions_path = outputs_dir / "component_captions.json"
    if not captions_path.exists():
        print(f"Error: {captions_path} not found")
        sys.exit(1)

    bbox_path = outputs_dir / "bbox_corners.json"
    if not bbox_path.exists():
        print(f"Error: {bbox_path} not found")
        sys.exit(1)

    with open(captions_path, "r") as f:
        captions_list = json.load(f)

    with open(bbox_path, "r") as f:
        bbox_data = json.load(f)

    manifest_path = outputs_dir / "crops" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
    else:
        print(
            f"Warning: manifest.json not found at {manifest_path}. best_crop will be NULL."
        )
        manifest_data = {}

    # Build lookup: connected_comp_id -> bbox
    bbox_lookup = {item["connected_comp_id"]: item["bbox"] for item in bbox_data}

    # Build lookup: component_id (str) -> best crop filename
    def get_best_crop(comp_id: int) -> str:
        entry = manifest_data.get(str(comp_id)) or manifest_data.get(comp_id)
        if not entry:
            return None
        crops = entry.get("crops", [])
        if not crops:
            return None
        best = max(crops, key=lambda c: c.get("fraction_visible", 0))
        return best.get("crop_filename")

    db_path = outputs_dir / "components.db"

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("DROP TABLE IF EXISTS components")
    cur.execute(
        """
        CREATE TABLE components (
            component_id  INTEGER PRIMARY KEY,
            caption       TEXT,
            bbox_json     TEXT,
            best_crop     TEXT
        )
        """
    )

    rows = []
    missing_bbox = []
    for item in captions_list:
        component_id = item["component_id"]
        caption = item.get("caption", "")
        bbox = bbox_lookup.get(component_id)

        if bbox is None:
            missing_bbox.append(component_id)
            bbox = {}

        rows.append(
            (component_id, caption, json.dumps(bbox), get_best_crop(component_id))
        )

    cur.executemany(
        "INSERT INTO components (component_id, caption, bbox_json, best_crop) VALUES (?, ?, ?, ?)",
        rows,
    )

    con.commit()
    con.close()

    print(f"Created database at {db_path} with {len(rows)} components")
    if missing_bbox:
        print(
            f"Warning: {len(missing_bbox)} component(s) had no matching bbox in "
            f"bbox_corners.json: {missing_bbox}"
        )
    return db_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_database.py <dataset_name>")
        print("Example: python create_database.py ArenaLabSemanticNeg")
        sys.exit(1)

    create_database(sys.argv[1])
