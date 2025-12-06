#!/bin/bash

# Check if dataset name is provided
if [ -z "$1" ]; then
    echo "Error: Dataset name is required"
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME="$1"

echo "Running floor detection pipeline for dataset: $DATASET_NAME"
echo "============================================"
echo ""

# Step 1: Run SAM3 for floor detection (requires sam3 environment)
echo "Step 1: Running SAM3 floor detection..."
conda run -n sam3 --no-capture-output python -m src.floordetection.run_sam3 --dataset "$DATASET_NAME"
if [ $? -ne 0 ]; then
    echo "Error: SAM3 floor detection failed"
    exit 1
fi
echo ""

# Step 2: Filter floor points (requires segment3d-env environment)
echo "Step 2: Filtering floor points..."
conda run -n segment3d-env --no-capture-output python -m src.floordetection.filter_points --dataset "$DATASET_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Floor points filtering failed"
    exit 1
fi
echo ""

# Step 3: Visualize floor point cloud (requires segment3d-env environment)
echo "Step 3: Visualizing floor point cloud..."
conda run -n segment3d-env --no-capture-output python -m src.floordetection.visualize_floor_pcd --dataset "$DATASET_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Floor point cloud visualization failed"
    exit 1
fi
echo ""

# Step 4: Generate occupancy grid (requires segment3d-env environment)
echo "Step 4: Generating occupancy grid..."
conda run -n segment3d-env --no-capture-output python -m src.floordetection.occupancy_grid --dataset "$DATASET_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Occupancy grid generation failed"
    exit 1
fi
echo ""

echo "============================================"
echo "Floor detection pipeline completed successfully for dataset: $DATASET_NAME"
