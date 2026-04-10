"""
ScanQA evaluation bridge for scan-to-map.

For each ScanQA question this script:
  1. Looks up the scan-to-map outputs for the corresponding ScanNet scene.
  2. Encodes the question text with the same OpenCLIP model that was used to
     generate the visual component embeddings.
  3. Searches the per-scene FAISS index for the top-10 most similar components.
  4. Returns those components' VLM captions as the predicted answer candidates.
  5. Saves predictions in the exact pickle format expected by ScanQA/scripts/score.py
     and then calls score.py to report the metrics.

Pipeline overview
-----------------
ScanQA val questions
        │
        ▼  (group by scene_id)
per-scene FAISS + captions  ←  scan-to-map outputs/<dataset>/
        │
        ▼  (CLIP text encode → nearest-neighbour search)
pred answers (component captions)
        │
        ▼  score.py  →  Top-1 EM / Top-10 EM / F-value / BLEU / …

Usage
-----
    python eval/scanqa_eval.py \\
        --scanqa-root   /path/to/ScanQA/data/qa/ \\
        --outputs-root  /path/to/scan-to-map/outputs/ \\
        [--split val] \\
        [--top-k 10] \\
        [--clip-model ViT-H-14] \\
        [--clip-pretrained laion2B-s32B-b79K] \\
        [--device 0] \\
        [--output-dir eval_results/] \\
        [--run-score]

The --run-score flag will call ScanQA's score.py automatically.
Point --scanqa-score-script at the script if it is not auto-detected.

Notes
-----
* The CLIP model/pretrained arguments must match what was used during
  `segment3d/main.py --clip-model ... --clip-pretrained ...` so that the
  text query embedding lives in the same space as the image embeddings.

* ScanNet scene_id → scan-to-map dataset_name mapping:
      scene0000_00  →  scannet_scene0000_00

* Only scenes that have an outputs/<dataset>/ directory with the required
  artifacts are evaluated; scenes without outputs are reported and skipped.

Required artifacts per scene (under outputs/<dataset>/):
    clip_embeddings.faiss        – FAISS HNSW index
    clip_embeddings.npz          – numpy array + component_ids array
    clip_embedding_stats.json    – records which CLIP model was used
    component_captions.json      – component_id → caption string
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------

def load_scene_artifacts(
    scene_outputs_dir: Path,
) -> Tuple[object, np.ndarray, List[str], Dict[str, str]] | None:
    """
    Load all scan-to-map artifacts for one scene.

    Returns (faiss_index, embeddings, component_ids, captions_dict) or None if
    any required file is missing.

    Parameters
    ----------
    scene_outputs_dir : Path
        outputs/<dataset_name>/
    """
    required = [
        "clip_embeddings.faiss",
        "clip_embeddings.npz",
        "clip_embedding_stats.json",
        "component_captions.json",
    ]
    missing = [r for r in required if not (scene_outputs_dir / r).exists()]
    if missing:
        print(f"  [SKIP] {scene_outputs_dir.name}: missing {missing}")
        return None

    try:
        import faiss
    except ImportError:
        raise ImportError("faiss-cpu is required.  pip install faiss-cpu")

    index = faiss.read_index(str(scene_outputs_dir / "clip_embeddings.faiss"))

    npz = np.load(str(scene_outputs_dir / "clip_embeddings.npz"), allow_pickle=True)
    embeddings   = npz["embeddings"].astype(np.float32)     # (N, D)
    component_ids = npz["component_ids"].tolist()            # list[str]

    with open(scene_outputs_dir / "component_captions.json", "r") as f:
        raw_captions = json.load(f)

    # Normalise caption dict: support both {id: str} and {id: {caption: str}}
    captions: Dict[str, str] = {}
    for k, v in raw_captions.items():
        if isinstance(v, str):
            captions[str(k)] = v
        elif isinstance(v, dict):
            captions[str(k)] = v.get("caption", "")
        else:
            captions[str(k)] = str(v)

    return index, embeddings, [str(c) for c in component_ids], captions


def load_clip_model(
    model_name: str,
    pretrained: str,
    device_str: str,
):
    """Load an OpenCLIP model for text encoding (returns model + tokenizer)."""
    try:
        import open_clip
    except ImportError:
        raise ImportError("open-clip-torch is required.  pip install open-clip-torch")

    import torch

    print(f"Loading CLIP model {model_name} ({pretrained}) on {device_str} …")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device_str).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Per-question CLIP text search
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts(questions: List[str], model, tokenizer, device_str: str) -> np.ndarray:
    """Encode a list of question strings → L2-normalised float32 (N, D)."""
    import torch

    tokens = tokenizer(questions).to(device_str)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def search_top_k(
    query_vec: np.ndarray,           # (1, D) float32
    faiss_index,
    component_ids: List[str],
    captions: Dict[str, str],
    top_k: int = 10,
) -> List[str]:
    """
    FAISS nearest-neighbour search → list of captions (top-k answers).

    The FAISS HNSW index stores image embeddings in the same order as
    component_ids.  We return the captions for the top-k matching components.
    """
    k = min(top_k, faiss_index.ntotal)
    _, indices = faiss_index.search(query_vec, k)          # (1, k)

    answers = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(component_ids):
            continue
        cid = component_ids[idx]
        caption = captions.get(str(cid), "")
        if caption:
            answers.append(caption)

    # Pad to top_k with the last caption if needed (keeps score.py happy)
    while answers and len(answers) < top_k:
        answers.append(answers[-1])

    return answers[:top_k]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    scanqa_root: str,
    outputs_root: str,
    split: str = "val",
    top_k: int = 10,
    clip_model: str = "ViT-H-14",
    clip_pretrained: str = "laion2B-s32B-b79K",
    device: Optional[int] = None,
    output_dir: str = "eval_results",
    run_score: bool = False,
    scanqa_score_script: Optional[str] = None,
) -> Path:
    """
    Run CLIP-based ScanQA evaluation.

    Returns the path to the saved predictions pickle.
    """
    import torch

    # ---- setup ----------------------------------------------------------------
    device_str = f"cuda:{device}" if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device_str}")

    scanqa_path   = Path(scanqa_root)
    outputs_path  = Path(outputs_root)
    out_dir       = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ScanQA questions ------------------------------------------------
    qa_file = scanqa_path / f"ScanQA_v1.0_{split}.json"
    if not qa_file.exists():
        raise FileNotFoundError(f"ScanQA data not found: {qa_file}")

    with open(qa_file, "r") as f:
        qa_data = json.load(f)                 # list of question dicts

    print(f"Loaded {len(qa_data)} questions from {qa_file}")

    # Group by scene_id
    by_scene: Dict[str, list] = {}
    for item in qa_data:
        sid = item["scene_id"]
        by_scene.setdefault(sid, []).append(item)

    print(f"Questions span {len(by_scene)} scenes")

    # ---- map scene_id → dataset_name → outputs dir ---------------------------
    # ScanNet scene_id (e.g. "scene0000_00") → dataset_name "scannet_scene0000_00"
    def scene_to_dataset(scene_id: str) -> str:
        return f"scannet_{scene_id}"

    # Find which scenes have scan-to-map outputs
    available_datasets = {d.name for d in outputs_path.iterdir() if d.is_dir()} \
        if outputs_path.exists() else set()

    scenes_to_eval = {
        sid for sid in by_scene
        if scene_to_dataset(sid) in available_datasets
    }
    scenes_missing = set(by_scene.keys()) - scenes_to_eval

    if scenes_missing:
        print(f"\nWarning: {len(scenes_missing)} scenes have no scan-to-map outputs "
              f"and will be skipped.")
        print(f"  Missing datasets: "
              + ", ".join(scene_to_dataset(s) for s in sorted(scenes_missing)[:5])
              + ("…" if len(scenes_missing) > 5 else ""))

    print(f"\nEvaluating {len(scenes_to_eval)} scenes "
          f"({sum(len(by_scene[s]) for s in scenes_to_eval)} questions)\n")

    if not scenes_to_eval:
        print("ERROR: No scenes with outputs found.  "
              "Run the full scan-to-map pipeline first:\n"
              "  cd segment3d/\n"
              "  python main.py --dataset scannet_<scene_id>")
        sys.exit(1)

    # ---- load CLIP model once ------------------------------------------------
    model, tokenizer = load_clip_model(clip_model, clip_pretrained, device_str)

    # ---- per-scene CLIP search -----------------------------------------------
    # predictions format expected by score.py:
    #   { scene_id: { question_id: { "pred_answers_at10": [str, …] } } }
    predictions: Dict[str, Dict] = {}
    total_q = 0

    for scene_id in sorted(scenes_to_eval):
        dataset_name = scene_to_dataset(scene_id)
        scene_dir    = outputs_path / dataset_name
        questions    = by_scene[scene_id]

        print(f"  Scene {scene_id}  ({len(questions)} questions)")

        artifacts = load_scene_artifacts(scene_dir)
        if artifacts is None:
            continue

        faiss_index, embeddings, component_ids, captions = artifacts

        # Batch-encode all questions for this scene
        q_texts = [q["question"] for q in questions]
        q_vecs  = encode_texts(q_texts, model, tokenizer, device_str)  # (Q, D)

        scene_preds: Dict = {}
        for q_item, q_vec in zip(questions, q_vecs):
            qid = str(q_item["question_id"])
            answers = search_top_k(
                q_vec[np.newaxis, :].astype(np.float32),
                faiss_index,
                component_ids,
                captions,
                top_k=top_k,
            )
            scene_preds[qid] = {"pred_answers_at10": answers}

        predictions[scene_id] = scene_preds
        total_q += len(questions)

    print(f"\nGenerated predictions for {total_q} questions across "
          f"{len(predictions)} scenes")

    # ---- save predictions ----------------------------------------------------
    pred_path = out_dir / f"pred.{split}.pkl"
    with open(pred_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Predictions saved → {pred_path}")

    # ---- (optional) run score.py --------------------------------------------
    if run_score:
        _run_score_script(
            pred_path=pred_path,
            scanqa_score_script=scanqa_score_script,
            pred_folder=out_dir,
        )

    return pred_path


# ---------------------------------------------------------------------------
# Score.py runner
# ---------------------------------------------------------------------------

def _run_score_script(
    pred_path: Path,
    scanqa_score_script: Optional[str],
    pred_folder: Path,
) -> None:
    """
    Call ScanQA/scripts/score.py with the --folder argument pointing at
    the directory that contains pred.val.pkl.
    """
    # Auto-detect score.py relative to this file's location
    if scanqa_score_script is None:
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent.parent / "ScanQA" / "scripts" / "score.py",
            here.parent / "ScanQA" / "scripts" / "score.py",
        ]
        for c in candidates:
            if c.exists():
                scanqa_score_script = str(c)
                break

    if scanqa_score_script is None or not Path(scanqa_score_script).exists():
        print("\n[score] Cannot find ScanQA/scripts/score.py automatically.")
        print("  Run it manually:")
        print(f"    cd <ScanQA_root>")
        print(f"    python scripts/score.py --folder <abs_path_containing_pred.val.pkl>")
        return

    # score.py expects --folder to point at a directory that lives under
    # ScanQA's outputs/ root AND must contain pred.val.pkl.
    # We copy pred.pkl into a temp subfolder of ScanQA's outputs/ to satisfy
    # that constraint, or we can just call score.py with a patched CONF.
    #
    # Simplest approach: call score.py pointing at the folder directly.
    # score.py uses CONF.PATH.OUTPUT / folder_name, so we create a symlink
    # or just override with --folder as an absolute path by monkey-patching.
    #
    # Actually the cleanest path: just call score.py after setting
    # CONF.PATH.OUTPUT to the *parent* of pred_folder.

    score_py = Path(scanqa_score_script).resolve()
    score_root = score_py.parent.parent          # ScanQA root

    env = os.environ.copy()
    env["PYTHONPATH"] = str(score_root) + os.pathsep + env.get("PYTHONPATH", "")

    # Pass folder as the name relative to pred_folder.parent
    folder_name = pred_folder.name
    parent_dir  = str(pred_folder.parent)

    # Patch CONF.PATH.OUTPUT at call time via env var (score.py uses easydict CONF)
    # This is simpler than modifying score.py; we do it by injecting a tiny
    # wrapper around the CONF before running the script.
    wrapper = (
        f"import sys; sys.path.insert(0, {repr(str(score_root))});\n"
        f"from lib.config import CONF; CONF.PATH.OUTPUT = {repr(parent_dir)};\n"
        f"import runpy; runpy.run_path({repr(str(score_py))}, run_name='__main__')"
    )

    cmd = [sys.executable, "-c", wrapper, "--folder", folder_name]
    print(f"\n[score] Running: {' '.join(cmd[:3])} --folder {folder_name}")
    subprocess.run(cmd, env=env, cwd=str(score_root))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Lazy import so the module is importable without torch installed
    global torch
    import torch  # noqa: F811  (re-bind for the @torch.no_grad decorator above)

    parser = argparse.ArgumentParser(
        description="Evaluate scan-to-map with ScanQA metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scanqa-root",
        required=True,
        help="Path to ScanQA data directory containing ScanQA_v1.0_val.json",
    )
    parser.add_argument(
        "--outputs-root",
        required=True,
        help="Path to scan-to-map outputs/ directory",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test_w_obj", "test_wo_obj"],
        help="Dataset split to evaluate (default: val)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top components to return as answer candidates (default: 10)",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-H-14",
        help="OpenCLIP model name — must match what segment3d used (default: ViT-H-14)",
    )
    parser.add_argument(
        "--clip-pretrained",
        default="laion2B-s32B-b79K",
        help="OpenCLIP pretrained weights (default: laion2B-s32B-b79K)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="CUDA device index (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory to save predictions and results (default: eval_results/)",
    )
    parser.add_argument(
        "--run-score",
        action="store_true",
        help="Automatically call ScanQA/scripts/score.py after saving predictions",
    )
    parser.add_argument(
        "--scanqa-score-script",
        default=None,
        help="Explicit path to ScanQA/scripts/score.py (auto-detected if omitted)",
    )

    args = parser.parse_args()

    run_evaluation(
        scanqa_root=args.scanqa_root,
        outputs_root=args.outputs_root,
        split=args.split,
        top_k=args.top_k,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        output_dir=args.output_dir,
        run_score=args.run_score,
        scanqa_score_script=args.scanqa_score_script,
    )


if __name__ == "__main__":
    main()
