# Coordinate Transformations Reference

This document details every coordinate transform in the scan-to-map pipeline, covering both the **Polycam** and **ScanNet++** data paths.

---

## Coordinate Systems

| Name                | Convention      | Up Axis                  | Source                                       |
| ------------------- | --------------- | ------------------------ | -------------------------------------------- |
| **COLMAP**          | Right-handed    | Varies (depends on scan) | SfM reconstruction                           |
| **ScanNet++ mesh**  | Right-handed    | Z-up                     | `mesh_aligned_0.05.ply`                      |
| **glTF / Three.js** | Right-handed    | Y-up                     | Industry standard                            |
| **Viewer**          | Y-up (Three.js) | Y                        | `Model3DViewer.tsx` via `@react-three/fiber` |

---

## Pipeline A: Polycam Data

### A1. Mesh — Polycam GLB (`data/<dataset>/polycam_data/raw.glb`)

Polycam exports meshes in **glTF Y-up** convention. The viewer loads the GLB directly with no additional rotation.

- **File**: `Model.tsx` (semantic-3d-search-demo/src/viewer/Model.tsx) — `useGLTF(url)` → `<primitive object={scene} />`
- **Transform**: None. Mesh is native glTF Y-up.

### A2. Point Cloud Alignment (`data-processor/polycam.py`)

The Polycam mesh is sampled and aligned to the COLMAP sparse cloud.

#### A2a. GLB → Z-up Sampling Space

Samples are taken from the GLB mesh and rotated **-90° around X** to go from Y-up to Z-up for alignment with the COLMAP cloud.

```
(x, y, z)_glTF  →  (x, -z, y)_Z-up
```

- **File**: [polycam.py L144–153](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/data-processor/polycam.py#L144-L153)
- **Reason**: The COLMAP sparse point cloud is in an arbitrary Z-up-like frame; mapping GLB samples into the same convention lets ICP/RANSAC align them.

#### A2b. RANSAC + ICP Registration

Computes `T_global` (4×4 rigid) that maps the Z-up sampled mesh onto the COLMAP frame.

- **File**: [polycam.py L268–423](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/data-processor/polycam.py#L268-L423)
- **Outputs**: `T_sampled_to_colmap` and its inverse `T_colmap_to_sampled`

#### A2c. GLB Transform for Export

To export a transformed GLB that will look correct when re-imported (Y-up tools), the script sandwiches the COLMAP alignment between the Y-up↔Z-up basis change:

```
T_blender_ready = T_fix_inv @ T_global @ T_fix
```

- **File**: [polycam.py L173–212](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/data-processor/polycam.py#L173-L212)
- **Reason**: The raw GLB is in Y-up; `T_fix` brings it to Z-up, `T_global` aligns to COLMAP, `T_fix_inv` brings back to Y-up for glTF export.

#### A2d. COLMAP Model Transform

The COLMAP model is transformed into the Z-up sampled frame so it aligns with the raw GLB when loaded in Blender. An additional +90° rotation about Z is applied.

- **File**: [polycam.py L594–612](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/data-processor/polycam.py#L594-L612)

### A3. Bounding Box Computation (Polycam)

Bounding boxes are computed from **COLMAP 3D point** coordinates (the transformed COLMAP model).

```
bbox.min = [X_colmap, Y_colmap, Z_colmap]
bbox.max = [X_colmap, Y_colmap, Z_colmap]
```

- **File**: [segment3d/src/bbox_corners.py](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/segment3d/src/bbox_corners.py#L13-L97)
- **Coordinate system**: Transformed COLMAP frame (post-alignment)

---

## Pipeline B: ScanNet++ Data

### B1. Mesh — PLY to GLB (`data-processor/scannet_pp/create_gltf.py`)

The source mesh `mesh_aligned_0.05.ply` is in **Z-up**. Two rotations are baked into the exported GLB:

| Step         | Transform          | Effect on `(X, Y, Z)`                     |
| ------------ | ------------------ | ----------------------------------------- |
| R1           | -90° around X-axis | `(X, Y, Z)` → `(X, -Z, Y)` — Z-up to Y-up |
| R2           | -90° around Y-axis | `(X', Y', Z')` → `(Z', Y', -X')`          |
| **Combined** | R2 ∘ R1            | `(X, Y, Z)` → **`(Y, -Z, -X)`**           |

- **File**: [create_gltf.py L46–56](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/data-processor/scannet_pp/create_gltf.py#L46-L56)
- **Reason for R1**: Convert ScanNet++ Z-up to glTF Y-up standard.
- **Reason for R2**: Compensate for an additional rotation that happens in the viewer (comment in source: _"for some transformations that happen in the viewer"_).

### B2. Bounding Box Computation (ScanNet++)

Bounding boxes are computed from the **raw** `mesh_aligned_0.05.ply` vertex positions — **no rotation is applied**.

```
bbox.min = [X_raw, Y_raw, Z_raw]
bbox.max = [X_raw, Y_raw, Z_raw]
```

- **File**: [create_bboxes.py L58–84](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/data-processor/scannet_pp/create_bboxes.py#L58-L84) (called with [mesh_xyz from line 102](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/data-processor/scannet_pp/create_bboxes.py#L102))
- **Coordinate system**: Raw ScanNet++ Z-up frame

---

## Runtime Transforms (Shared by Both Pipelines)

### R1. Server-Side Axis Swap (`search-server/app.py`)

When returning bboxes from `/search` or `/search_stream`, the server applies a **cyclic axis permutation**:

```python
def transform_bbox(bbox):
    return {
        "x_min": bbox["min"][1],   # viewer X = stored Y
        "y_min": bbox["min"][2],   # viewer Y = stored Z
        "z_min": bbox["min"][0],   # viewer Z = stored X
        "x_max": bbox["max"][1],
        "y_max": bbox["max"][2],
        "z_max": bbox["max"][0],
    }
```

- **File**: [app.py L101–131](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/search-server/app.py#L101-L131)
- **Also used for**: Route path coordinates (`transform_coordinates` at [L101–111](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/search-server/app.py#L101-L111))

### R2. `/download_all_components` — No Server-Side Transform

The endpoint returns bboxes **as-is** from the database (no `transform_bbox` call).

- **File**: [app.py L721–765](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/search-server/app.py#L721-L765)

### R3. Client-Side Axis Swap (Frontend)

The same `[y, z, x]` axis swap is applied client-side when loading auto-tag annotations and the occupancy grid:

```typescript
// SearchBar.tsx — "Download Annotations" button
const bboxes = data.map((item) => ({
  x_min: item.bbox.min[1],
  y_min: item.bbox.min[2],
  z_min: item.bbox.min[0],
  x_max: item.bbox.max[1],
  y_max: item.bbox.max[2],
  z_max: item.bbox.max[0],
}));
```

- **Files**:
  - [SearchBar.tsx L73–80](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/semantic-3d-search-demo/src/SearchBar.tsx#L73-L80)
  - [App.tsx L55–62](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/semantic-3d-search-demo/src/App.tsx#L55-L62) (occupancy grid)

### R4. Viewer — No Mesh Transform

The GLB is rendered as-is by Three.js, which natively uses Y-up. No additional rotation is applied to the mesh in the viewer.

- **File**: [Model.tsx L5–44](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/semantic-3d-search-demo/src/viewer/Model.tsx#L5-L44)

### R5. Inverse Transform on Save

When saving an edited bbox back to the server, the client reverses the axis swap:

```typescript
// Model3DViewer.tsx — handleSave
updates.bbox = {
  min: [bboxToSave.z_min, bboxToSave.x_min, bboxToSave.y_min],
  max: [bboxToSave.z_max, bboxToSave.x_max, bboxToSave.y_max],
};
```

- **File**: [Model3DViewer.tsx L190–196](file:///home/sagar/Repos/openFLAME-repos/scan-to-map/semantic-3d-search-demo/src/Model3DViewer.tsx#L190-L196)

---

## Summary: Stored Coords → Viewer Coords

| Pipeline      | Stored `bbox.min`            | Mesh in Viewer              | BBox in Viewer (after axis swap) |
| ------------- | ---------------------------- | --------------------------- | -------------------------------- |
| **Polycam**   | `(X_c, Y_c, Z_c)` (COLMAP)   | Aligned to COLMAP           | `(Y_c, Z_c, X_c)`                |
| **ScanNet++** | `(X_s, Y_s, Z_s)` (raw mesh) | `(Y_s, -Z_s, -X_s)` (R2∘R1) | `(Y_s, Z_s, X_s)`                |

> [!WARNING]
> **Known issue (ScanNet++ only)**: The bbox axis swap produces `(Y_s, Z_s, X_s)` but the mesh is at `(Y_s, -Z_s, -X_s)`. The Y and Z components have sign mismatches, causing annotations to appear misaligned with the mesh. This does not affect Polycam data.
