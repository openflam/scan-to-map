# Split/merge component sensitivity experiment

This workflow measures how downstream benchmark performance changes when the
scene-memory segmentation contains component splits or spatially plausible
component merges. It follows the isolated artifact layout and staged execution
of `../missing_components`.

## Design

For a scene with `N` searchable components, a rate `r` performs
`round(r * N)` component-boundary errors:

- a split bisects a component's axis-aligned box along its longest dimension,
  keeps the original ID for one child, and assigns the other a new ID;
- a merge greedily pairs disjoint components with their nearest available box,
  retains one deterministic canonical ID, and unions their boxes.

The 5%, 10%, 20%, and 30% conditions are nested within each scene and seed.
Captions and crops are oracle-preserving: split children reuse the source
caption and crops, while merged components concatenate source captions and
combine source crops. This isolates component topology from caption noise.
This remains the default behavior so existing experiment plans are unchanged.

Pass `--recreate-crops-captions` to measure the coupled segmentation and visual
memory effect instead. For every affected split child or merged component, the
preparer projects its new 3-D box through the source COLMAP cameras using the
same quaternion, camera-model distortion, visibility threshold, and clamping
logic as `segment3d/src/project_bbox.py`. It then writes fresh JPEG crops from
the original source images and captions those crops through
`segment3d/src/captioning`. Unaffected components continue to reference their
clean artifacts.

When `outputs/<scene>/connected_components.json` is available, split point
memberships are partitioned by the split plane and merge memberships are
unioned, so image eligibility uses exact COLMAP point visibility. If membership
data is absent, the source crop manifest supplies the candidate source views;
the new box is still reprojected and newly cropped. The selected protocol is
recorded in each alias's `scene_memory_perturbation.json`.
For ScanNet++, pass `--scannetpp-data-root` to use
`<root>/<scene-id>/dslr/colmap` with the corresponding
`dslr/resized_images`. Because those DSLR image IDs differ from the original
`frame_*.jpg` crop manifests, component point memberships are approximated from
the DSLR sparse points inside each source 3-D box, then split or unioned for the
perturbation. DSLR `OPENCV_FISHEYE` distortion is applied during projection.

Every scene records `component_lineage`, mapping a perturbed component ID to
the clean component IDs it represents. Summarization reports the existing
strict-ID metrics as well as source-space precision, recall, F1, a topology
ceiling, split redundancy, and merge contamination.

The default study has one clean baseline and split/merge conditions at four
rates with two seeds: 17 conditions. With the current 69-question benchmark and
one tool configuration, this is 1,173 top-level benchmark queries.

## Dry run

Validate all source artifacts and print exact query/API-call counts without
creating files, loading PostGIS, or calling a model:

```bash
docker compose exec search-server python \
  /app/benchmark-eval/sensitivity_analysis/split_merge_components/experiment.py \
  --dry-run
```

To show deterministic metrics only in the query plan, add
`--disable-ai-judge`.

To include crop/caption regeneration in the plan without loading the caption
model or writing files:

```bash
docker compose exec search-server python \
  /app/benchmark-eval/sensitivity_analysis/split_merge_components/experiment.py \
  --recreate-crops-captions \
  --scannetpp-data-root /home/sagar/Repos/open-datasets/ScanNetPP/data/data \
  --dry-run
```

The dry run also prints the number of affected component captions and reports
how many source scenes are locally ready. A full recreation run requires each
scene's original `data/<scene>/ns_data/images` directory and transformed or
default HLOC COLMAP reconstruction. The current default caption settings match
the segment3d pipeline; they can be changed with `--caption-model`,
`--caption-device`, `--caption-n-images`, and `--caption-batch-size`.

## Full workflow

```bash
docker compose exec search-server python \
  /app/benchmark-eval/sensitivity_analysis/split_merge_components/experiment.py
```

The command prepares aliases, loads only their PostGIS tables, generates
answers, computes metrics, and writes aggregate CSV, JSON, and PDF artifacts to
`benchmark/split_merge_components/analysis`.

Crop/caption recreation needs the local GPU captioning dependencies used by
segment3d. Prepare those artifacts in that environment, then resume the
benchmark stages in the search-server container:

```bash
conda run -n segment3d-env python \
  benchmark-eval/sensitivity_analysis/split_merge_components/prepare.py \
  --recreate-crops-captions \
  --scannetpp-data-root /home/sagar/Repos/open-datasets/ScanNetPP/data/data

docker compose exec search-server python \
  /app/benchmark-eval/sensitivity_analysis/split_merge_components/experiment.py \
  --recreate-crops-captions --resume
```

## Individual stages

```bash
python benchmark-eval/sensitivity_analysis/split_merge_components/prepare.py --dry-run
python benchmark-eval/sensitivity_analysis/split_merge_components/prepare.py
python benchmark-eval/sensitivity_analysis/split_merge_components/run.py
python benchmark-eval/sensitivity_analysis/split_merge_components/run.py --stage load-db
python benchmark-eval/sensitivity_analysis/split_merge_components/run.py --stage answers
python benchmark-eval/sensitivity_analysis/split_merge_components/run.py --stage metrics
python benchmark-eval/sensitivity_analysis/split_merge_components/summarize.py
```

The runner defaults to printing commands. Use `--stage all` to execute all
runtime stages. A matching interrupted run can be continued with `--resume`
and, after verifying their provenance, `--reuse-existing-db-tables`.

Use `summarize_partial.py` only for progress inspection. Its output includes a
condition completeness table and must not be treated as final results.

## Artifacts and safety

- Clean `outputs/<dataset>` and `benchmark/data` files remain unchanged.
- Perturbed aliases are created under `outputs/split_merge_...`.
- With recreation enabled, affected component crop directories contain new JPEG
  files (not links), and aliases contain their new projection coordinates.
- Questions, results, metrics, and analysis remain below
  `benchmark/split_merge_components`.
- Existing aliases, experiment directories, results, metrics, and analysis
  directories are never silently overwritten.
- `connected_components.json` is not materialized in perturbed aliases because
  it is not consumed by the benchmark search loader and some ScanNet++ scene
  memories do not contain point-membership data. The searchable captions,
  bounding boxes, crop manifest, and crop-image access remain ID-consistent.
