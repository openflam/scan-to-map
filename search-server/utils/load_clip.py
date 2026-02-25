import json
from pathlib import Path

from semantic_search import CLIPProvider


def load_clip_provider(dataset_name: str, outputs_root: Path) -> CLIPProvider:
    """
    Load and return a CLIPProvider for the given dataset.

    Args:
        dataset_name: Name of the dataset (e.g. "ArenaLabSemanticNeg").
        outputs_root: Absolute path to the outputs/ directory.

    Returns:
        An initialised CLIPProvider instance.
    """
    dataset_dir = outputs_root / dataset_name

    faiss_index_path = dataset_dir / "clip_embeddings.faiss"
    embeddings_npz_path = dataset_dir / "clip_embeddings.npz"
    clip_stats_path = dataset_dir / "clip_embedding_stats.json"

    with open(clip_stats_path, "r") as f:
        clip_stats = json.load(f)

    clip_model_name = clip_stats["model_name"]
    clip_pretrained = clip_stats["pretrained"]
    print(f"CLIP model configuration: {clip_model_name} ({clip_pretrained})")

    # Build component_captions dict (required by CLIPProvider for in-memory lookup)
    import sqlite3

    db_path = dataset_dir / "components.db"
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT component_id, caption, bbox_json FROM components")
    rows = cur.fetchall()
    con.close()

    component_captions = {}
    for row in rows:
        comp_id = row["component_id"]
        bbox = json.loads(row["bbox_json"])
        component_captions[comp_id] = {
            "component_id": comp_id,
            "caption": row["caption"],
            "bbox": bbox,
        }
    print(f"Loaded {len(component_captions)} components for CLIP provider")

    return CLIPProvider(
        faiss_index_path=str(faiss_index_path),
        embeddings_npz_path=str(embeddings_npz_path),
        component_captions=component_captions,
        model_name=clip_model_name,
        pretrained=clip_pretrained,
    )
