"""Debug a connected component by visualizing all contributing masks.

Reads connected_components.json and the masks in the masks directory to
render every mask that contributed to the specified component.

Usage:
    python debug_component.py --dataset_name ProjectLabStudio_NoNeg --component_id 5
    python debug_component.py --dataset_name ProjectLabStudio_NoNeg --component_id 5 --display
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports – fail early with a helpful message if missing
# ---------------------------------------------------------------------------
try:
    from pycocotools import mask as mask_utils
except ImportError:
    sys.exit("pycocotools is required.  Install with: pip install pycocotools")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow is required.  Install with: pip install Pillow")

# Add the segment3d package to the path so we can import io_paths / config
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from src.io_paths import (
    get_colmap_model_dir,
    get_images_dir,
    get_masks_dir,
    get_outputs_dir,
    load_config,
)  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OVERLAY_COLOR = (255, 80, 0)  # orange-red for the mask fill
OVERLAY_ALPHA = 0.55  # blend amount


def parse_mask_id(mask_id: str):
    """Split 'frame_00001_mask_5' → ('frame_00001', 5)."""
    frame_stem, _, idx_str = mask_id.rpartition("_mask_")
    if not frame_stem:
        raise ValueError(f"Cannot parse mask_id: {mask_id!r}")
    return frame_stem, int(idx_str)


def load_masks_json(masks_dir: Path, frame_stem: str) -> Optional[List[dict]]:
    """Load the masks list for a given frame stem, or return None if missing."""
    path = masks_dir / f"{frame_stem}_masks.json"
    if not path.exists():
        print(f"  [warn] Mask file not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    raise ValueError(f"Unexpected format in {path}")


def decode_mask(rle_entry: dict) -> np.ndarray:
    """Decode a pycocotools RLE/polygon entry to a boolean numpy array."""
    seg = rle_entry["segmentation"]
    # If it's already an RLE dict with size/counts use decode directly;
    # otherwise encode it first (polygon).
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    # Polygon format – need height/width from the outer dict if available
    raise ValueError("Polygon segmentation not supported; expected RLE dict.")


def render_mask_image(
    mask_bool: np.ndarray,
    mask_id: str,
    orig_image: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Create an RGB image showing the mask as a semi-transparent coloured overlay.
    If orig_image is provided it is used as the background; otherwise falls back
    to a plain white background.
    """
    H, W = mask_bool.shape

    if orig_image is not None:
        # Resize original to match mask dimensions if needed
        if (orig_image.height, orig_image.width) != (H, W):
            orig_image = orig_image.resize((W, H), Image.LANCZOS)
        bg = np.array(orig_image.convert("RGB"), dtype=np.uint8)
    else:
        bg = np.full((H, W, 3), 255, dtype=np.uint8)

    # Apply overlay colour where mask is True
    overlay = bg.copy()
    overlay[mask_bool] = OVERLAY_COLOR

    # Blend only the masked region; leave the rest as-is
    alpha = np.where(mask_bool[:, :, None], OVERLAY_ALPHA, 0.0)
    blended = (overlay * alpha + bg * (1 - alpha)).astype(np.uint8)

    # Draw a red border along the mask boundary (dilate XOR erode)
    from scipy.ndimage import binary_dilation, binary_erosion

    border = binary_dilation(mask_bool, iterations=2) & ~binary_erosion(
        mask_bool, iterations=2
    )
    blended[border] = (220, 0, 0)  # red border

    img = Image.fromarray(blended, mode="RGB")

    # Draw label
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()

    label = mask_id
    # Semi-transparent label background
    bbox = draw.textbbox((4, 4), label, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255), font=font)

    return img


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------


def write_ply(path: Path, xyzs: np.ndarray, rgbs: np.ndarray) -> None:
    """Write a binary little-endian PLY file with float32 XYZ and uint8 RGB."""
    n = len(xyzs)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # Build a structured array: 3x float32 + 3x uint8 = 15 bytes per point
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    data = np.empty(n, dtype=dtype)
    data["x"] = xyzs[:, 0].astype(np.float32)
    data["y"] = xyzs[:, 1].astype(np.float32)
    data["z"] = xyzs[:, 2].astype(np.float32)
    data["red"] = rgbs[:, 0]
    data["green"] = rgbs[:, 1]
    data["blue"] = rgbs[:, 2]

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def load_original_image(images_dir: Path, frame_stem: str) -> Optional[Image.Image]:
    """Try common extensions and return a PIL image, or None if not found."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = images_dir / f"{frame_stem}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    print(f"  [warn] Original image not found for {frame_stem} in {images_dir}")
    return None


def debug_component(
    dataset_name: str, component_id: int, display: bool = False
) -> None:
    config = load_config(dataset_name)
    masks_dir = get_masks_dir(config)
    outputs_dir = get_outputs_dir(config)
    try:
        images_dir = get_images_dir(config)
    except Exception:
        images_dir = None
        print(
            "[warn] Could not resolve images_dir – overlays will use white background."
        )
    try:
        colmap_model_dir = get_colmap_model_dir(config)
    except Exception:
        colmap_model_dir = None
        print(
            "[warn] Could not resolve colmap_model_dir – skipping point cloud export."
        )

    components_path = outputs_dir / "connected_components.json"
    if not components_path.exists():
        sys.exit(f"connected_components.json not found at {components_path}")

    with components_path.open("r", encoding="utf-8") as f:
        components = json.load(f)

    # Find the requested component
    component = None
    for c in components:
        if c.get("connected_comp_id") == component_id:
            component = c
            break

    if component is None:
        ids = [c.get("connected_comp_id") for c in components]
        sys.exit(
            f"Component {component_id} not found.  "
            f"Available IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}"
        )

    mask_id_set: List[str] = component.get("mask_id_set", [])

    if not mask_id_set:
        sys.exit(
            f"Component {component_id} has no 'mask_id_set'.  "
            "Re-run the mask_graph pipeline to regenerate connected_components.json "
            "with provenance tracking."
        )

    print(f"Component {component_id}: {len(mask_id_set)} contributing mask(s)")

    # Group mask_ids by frame so we load each file only once
    by_frame: Dict[str, List[tuple]] = defaultdict(list)
    for mask_id in sorted(mask_id_set):
        frame_stem, mask_idx = parse_mask_id(mask_id)
        by_frame[frame_stem].append((mask_idx, mask_id))

    # Output directory
    out_dir = outputs_dir / "debug_components" / f"component_{component_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: List[Image.Image] = []

    for frame_stem in sorted(by_frame.keys()):
        masks_list = load_masks_json(masks_dir, frame_stem)
        if masks_list is None:
            continue

        orig_image = load_original_image(images_dir, frame_stem) if images_dir else None

        for mask_idx, mask_id in sorted(by_frame[frame_stem]):
            if mask_idx >= len(masks_list):
                print(
                    f"  [warn] mask index {mask_idx} out of range for {frame_stem} "
                    f"(file has {len(masks_list)} masks)"
                )
                continue

            try:
                mask_bool = decode_mask(masks_list[mask_idx])
            except Exception as exc:
                print(f"  [warn] Failed to decode {mask_id}: {exc}")
                continue

            img = render_mask_image(mask_bool, mask_id, orig_image=orig_image)
            rendered.append(img)

            out_path = out_dir / f"{mask_id}.png"
            img.save(out_path)
            print(f"  Saved {out_path}")

    if not rendered:
        print("No masks could be rendered.")
        return

    print(f"\nAll images saved to: {out_dir}")

    # -----------------------------------------------------------------------
    # Point cloud export
    # -----------------------------------------------------------------------
    if colmap_model_dir is not None:
        print("\nBuilding point cloud...")
        from src.colmap_io import load_colmap_model

        _, _, points3D = load_colmap_model(str(colmap_model_dir))

        component_ids = set(component["set_of_point3DIds"])

        all_ids = list(points3D.keys())
        xyzs = np.array([points3D[pid].xyz for pid in all_ids], dtype=np.float32)

        # COLMAP is Z-up; convert to Y-up for standard 3D viewers:
        #   new X =  old X,  new Y =  old Z,  new Z = -old Y
        xyzs = np.stack([xyzs[:, 0], xyzs[:, 2], -xyzs[:, 1]], axis=1)
        rgbs = np.zeros((len(all_ids), 3), dtype=np.uint8)

        for i, pid in enumerate(all_ids):
            if pid in component_ids:
                rgbs[i] = (255, 0, 0)  # red – component points
            else:
                rgbs[i] = (255, 255, 255)  # white – everything else

        ply_path = out_dir / f"component_{component_id}.ply"
        write_ply(ply_path, xyzs, rgbs)
        print(f"Point cloud saved to: {ply_path}")

    if display:
        try:
            import matplotlib.pyplot as plt

            fig, axes = (
                plt.subplots(1, len(rendered), figsize=(4 * len(rendered), 4))
                if len(rendered) > 1
                else plt.subplots(1, 1, figsize=(6, 6))
            )

            axes_flat = [axes] if len(rendered) == 1 else axes.flat
            for ax, img, mask_id in zip(axes_flat, rendered, sorted(mask_id_set)):
                ax.imshow(img)
                ax.set_title(mask_id, fontsize=7)
                ax.axis("off")

            fig.suptitle(
                f"Component {component_id} – {len(rendered)} masks", fontsize=12
            )
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available – skipping display.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize all masks that contributed to a connected component"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g. ProjectLabStudio_NoNeg)",
    )
    parser.add_argument(
        "--component_id",
        type=int,
        required=True,
        help="The connected_comp_id to inspect",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show images interactively using matplotlib",
    )
    args = parser.parse_args()
    debug_component(
        dataset_name=args.dataset_name,
        component_id=args.component_id,
        display=args.display,
    )


if __name__ == "__main__":
    main()
