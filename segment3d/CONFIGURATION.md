# Configuration System

The pipeline now supports multiple datasets through a Python-based configuration system.

## Quick Start

Run the pipeline for a dataset:

```bash
python main.py --dataset Area2300
```

## Configuration Files

### `config.py`

Automatically generates dataset-specific configurations based on a consistent directory structure. 

**Expected directory structure for each dataset:**
```
data/
├── {dataset_name}/
│   ├── ns_data/
│   │   └── images/          # Input images
│   └── colmap_known_poses/
│       └── sparse/
│           └── 1/            # COLMAP reconstruction
```

**Generated paths for each dataset:**
- `images_dir` → `data/{dataset_name}/ns_data/images`
- `colmap_model_dir` → `data/{dataset_name}/colmap_known_poses/sparse/1`
- `masks_dir` → `outputs/{dataset_name}/masks`
- `associations_dir` → `outputs/{dataset_name}/associations`
- `outputs_dir` → `outputs/{dataset_name}`

**Global settings (same for all datasets):**
- `sam_ckpt` → `checkpoints/sam_vit_h_4b8939.pth`
- `sam_model_type` → `vit_h`
- `device` → `cuda`

### Adding a New Dataset

Simply create a new directory under `data/` with the expected structure:

```bash
mkdir -p data/Area2400/ns_data/images
mkdir -p data/Area2400/colmap_known_poses/sparse/1
# Copy your data into these directories
```

The dataset will be automatically discovered! No code changes needed.

### List Available Datasets

The `list_datasets()` function automatically scans the `data/` directory and returns all subdirectories as available datasets.

## Usage Examples

### Run complete pipeline:
```bash
python main.py --dataset Area2300
```

### With custom parameters:
```bash
python main.py --dataset Area2300 --K 10 --tau 0.3 --percentile 90
```

### Skip already-completed steps:
```bash
python main.py --dataset Area2300 --skip-sam --skip-association
```

### Get help:
```bash
python main.py --help
```

## Available Arguments

- `--dataset` - Dataset to process (default: Area2300)
- `--skip-sam` - Skip SAM segmentation step
- `--skip-association` - Skip 2D-3D association step
- `--K` - Number of nearest neighbors for mask graph (default: 5)
- `--tau` - Jaccard similarity threshold (default: 0.2)
- `--percentile` - Percentile for bbox outlier removal (default: 95.0)
- `--min-fraction` - Minimum fraction of visible points (default: 0.3)

## Output Structure

Each dataset has its own output directory:

```
outputs/
├── Area2300/
│   ├── masks/
│   ├── associations/
│   ├── mask_graph.gpickle
│   ├── connected_components.json
│   ├── bbox_corners.json
│   ├── image_crop_coordinates.json
│   ├── crop_stats.json
│   └── crops/
│       ├── component_0/
│       ├── component_1/
│       └── manifest.json
└── Area2400/
    └── ...
```

## Implementation Details

The dataset name is passed through the pipeline in two ways:

1. **Direct parameter**: `load_config(dataset_name="Area2300")`
2. **Environment variable**: Set `SCAN_TO_MAP_DATASET` environment variable

The `main.py` script sets the environment variable so all downstream modules can access the dataset configuration without explicitly passing it through every function call.

**Note:** `dataset_name` is always required. If not provided as an argument, the function will check the `SCAN_TO_MAP_DATASET` environment variable. If neither is available, a `ValueError` will be raised.
