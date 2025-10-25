# Scripts

This directory contains the main pipeline script.

## Main Pipeline Script

**`main.py`** - Runs the complete scan-to-map pipeline sequentially.

### Usage

```bash
cd segment3d
python scripts/main.py
```

### Options

```
  --skip-sam            Skip SAM segmentation step (use existing masks)
  --skip-association    Skip 2D-3D association step (use existing associations)
  --K K                 Number of nearest neighbors for mask graph (default: 5)
  --tau TAU             Jaccard similarity threshold for mask graph (default: 0.2)
  --percentile PERCENTILE
                        Percentile threshold for bbox outlier removal (default: 95.0)
  --min-fraction MIN_FRACTION
                        Minimum fraction of visible points for projection (default: 0.3)
```

### Examples

Run the complete pipeline with default parameters:
```bash
python scripts/main.py
```

Run with custom parameters:
```bash
python scripts/main.py --K 10 --tau 0.3 --percentile 90 --min-fraction 0.5
```

Skip SAM if masks already exist:
```bash
python scripts/main.py --skip-sam
```

Skip SAM and associations:
```bash
python scripts/main.py --skip-sam --skip-association
```

### Pipeline Steps

The script runs these steps in order:

1. **SAM Segmentation** - Generate masks for all images
2. **2D-3D Association** - Map masks to COLMAP 3D points  
3. **Mask Graph** - Build connectivity graph and extract components
4. **Bounding Boxes** - Compute 3D bboxes for each component
5. **Projection** - Project 3D bboxes to 2D image coordinates
6. **Cropping** - Extract cropped image regions

### Requirements

- All configuration must be set in `segment3d/config.yaml`
- SAM checkpoint must be downloaded and configured
- COLMAP reconstruction must be available

See `CLI_REFERENCE.md` for more details on individual steps.
