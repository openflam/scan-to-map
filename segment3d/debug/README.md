# Debug Scripts

Standalone inspection tools for the `segment3d` pipeline outputs.
Run them from the **repo root** with the `segment3d-env` conda environment active.

---

## `component.py` — Visualize a single connected component

Renders every mask that contributed to a given connected component and, optionally, exports its 3D point cloud.

```bash
python debug/component.py --dataset_name <name> --component_id <int> [--display] [--only-mask]
```

| Flag             | Default      | Description                                                |
| ---------------- | ------------ | ---------------------------------------------------------- |
| `--dataset_name` | _(required)_ | Dataset folder name under `data/` and `outputs/`           |
| `--component_id` | _(required)_ | `connected_comp_id` to inspect                             |
| `--display`      | off          | Show rendered images interactively via matplotlib          |
| `--only-mask`    | off          | Crop to masked pixels only (white background outside mask) |

**Output** — written to `outputs/<dataset_name>/debug_components/component_<id>/`:

- One PNG per contributing mask showing the mask as a semi-transparent orange overlay on the source frame, with the mask ID labelled in the corner.
- `component_<id>.ply` — binary PLY point cloud of the full scene, with the component's 3D points highlighted in orange and all other points in grey. Open in MeshLab, CloudCompare, etc.

Supports both connected-component formats:

- `mask_id_set` (e.g. `frame_00001_mask_5`) — reads SAM masks from the masks directory.
- `instance_ids` (e.g. `Boxes_seq_0_0`) — reads per-frame JSONs from `object_level_masks/masks/`.

---

## `inter_components.py` — Compare instances across two connected components

Inspects cross-component instance overlap between two connected components.
By default it is fast: it reads stored values directly from `unmerged_edges.json`.
Pass `--fresh-calculate` to recompute everything from scratch (loads COLMAP + CLIP).

```bash
# Fast path – read from unmerged_edges.json only
python debug/inter_components.py --dataset_name <name> --comp_id_a <int> --comp_id_b <int>

# Full recalculation – loads COLMAP model and CLIP model
python debug/inter_components.py --dataset_name <name> --comp_id_a <int> --comp_id_b <int> --fresh-calculate
```

| Flag                | Default                   | Description                                                          |
| ------------------- | ------------------------- | -------------------------------------------------------------------- |
| `--dataset_name`    | _(required)_              | Dataset folder name under `data/` and `outputs/`                     |
| `--comp_id_a`       | _(required)_              | First connected-component ID                                         |
| `--comp_id_b`       | _(required)_              | Second connected-component ID                                        |
| `--voxel_size_cm`   | from `DEFAULT_PARAMETERS` | Voxel side length used for the Jaccard grid                          |
| `--fresh-calculate` | off                       | Recompute voxel Jaccard, containment, and CLIP distance from scratch |

**Output** — printed to stdout in two sections:

1. **Calculated values** _(only with `--fresh-calculate`)_ — a table over every cross-component instance pair with columns:
   - `Jaccard` — voxel-grid intersection-over-union of the two instances' 3D point sets.
   - `Contain` — intersection / min(|A|, |B|), i.e. how much the smaller instance is contained in the larger.
   - `CLIPdist` — cosine distance (1 − dot product) between the two instances' CLIP image embeddings.

2. **Unmerged edges from file** — the matching rows from `unmerged_edges.json` for these component pairs, showing the stored Jaccard, CLIP distance, and the rejection reason that prevented the merge.
