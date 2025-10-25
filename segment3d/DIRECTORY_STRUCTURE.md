# Expected Directory Structure

The pipeline expects a consistent directory structure for all datasets.

## Input Structure

```
scan-to-map/
├── data/
│   ├── Area2300/                    # Dataset 1
│   │   ├── ns_data/
│   │   │   └── images/              # Images for Area2300
│   │   │       ├── frame_00001.jpg
│   │   │       ├── frame_00002.jpg
│   │   │       └── ...
│   │   └── colmap_known_poses/
│   │       └── sparse/
│   │           └── 1/               # COLMAP reconstruction
│   │               ├── cameras.txt
│   │               ├── images.txt
│   │               └── points3D.txt
│   │
│   ├── Area2400/                    # Dataset 2 (example)
│   │   ├── ns_data/
│   │   │   └── images/
│   │   └── colmap_known_poses/
│   │       └── sparse/
│   │           └── 1/
│   │
│   └── MyCustomDataset/             # Dataset 3 (example)
│       ├── ns_data/
│       │   └── images/
│       └── colmap_known_poses/
│           └── sparse/
│               └── 1/
│
├── checkpoints/
│   └── sam_vit_h_4b8939.pth        # SAM model (shared across all datasets)
│
└── segment3d/
    ├── main.py
    ├── config.py
    └── ...
```

## Output Structure

After running the pipeline, outputs are organized by dataset:

```
scan-to-map/
└── outputs/
    ├── Area2300/                    # Outputs for Area2300
    │   ├── masks/
    │   │   ├── frame_00001_masks.json
    │   │   └── ...
    │   ├── associations/
    │   │   ├── frame_00001_associations.json
    │   │   └── ...
    │   ├── mask_graph.gpickle
    │   ├── connected_components.json
    │   ├── bbox_corners.json
    │   ├── image_crop_coordinates.json
    │   ├── crop_stats.json
    │   └── crops/
    │       ├── component_0/
    │       │   ├── frame_00001_crop000.jpg
    │       │   └── ...
    │       ├── component_1/
    │       └── manifest.json
    │
    ├── Area2400/                    # Outputs for Area2400
    │   └── ...
    │
    └── MyCustomDataset/             # Outputs for MyCustomDataset
        └── ...
```

## Path Generation Rules

The `config.py` file automatically generates paths based on the dataset name:

| Config Key | Generated Path |
|------------|----------------|
| `images_dir` | `data/{dataset_name}/ns_data/images` |
| `colmap_model_dir` | `data/{dataset_name}/colmap_known_poses/sparse/1` |
| `masks_dir` | `outputs/{dataset_name}/masks` |
| `associations_dir` | `outputs/{dataset_name}/associations` |
| `outputs_dir` | `outputs/{dataset_name}` |
| `sam_ckpt` | `checkpoints/sam_vit_h_4b8939.pth` (shared) |
| `sam_model_type` | `vit_h` (constant) |
| `device` | `cuda` (constant) |

## Adding a New Dataset

To add a new dataset called "MyNewDataset":

1. Create the directory structure:
   ```bash
   mkdir -p data/MyNewDataset/ns_data/images
   mkdir -p data/MyNewDataset/colmap_known_poses/sparse/1
   ```

2. Place your data:
   - Copy images to `data/MyNewDataset/ns_data/images/`
   - Copy COLMAP files to `data/MyNewDataset/colmap_known_poses/sparse/1/`

3. Run the pipeline:
   ```bash
   python main.py --dataset MyNewDataset
   ```

That's it! No code changes needed.

## Dataset Discovery

The `list_datasets()` function automatically discovers datasets by listing subdirectories in `data/`:

```bash
python -c "from config import list_datasets; print(list_datasets())"
# Output: ['Area2300', 'Area2400', 'MyCustomDataset']
```

You can also see available datasets when running:
```bash
python main.py --help
```
