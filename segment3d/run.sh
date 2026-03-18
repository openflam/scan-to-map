#!/bin/bash

# Check if dataset name is provided
if [ -z "$1" ]; then
    echo "Error: Dataset name is required"
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME="$1"

echo "Running pipeline for dataset: $DATASET_NAME"
echo "============================================"
echo ""

set -e

# Step 0: Create objects inventory
python -m src.objects_inventory.orchestrator --dataset "$DATASET_NAME" --batch-size 8 --identifier-type openai --model gpt-4o-mini

# Step 1: Normalize labels
python -m src.objects_inventory.normalize_labels --dataset "$DATASET_NAME" --distance-threshold 0.05 --skip-lemmatize

# Step 2: Run SAM3
python src/per_object_sam3/sam3_runner.py --dataset "$DATASET_NAME"

# Step 3: Post-SAM3 pipeline
python -m src.per_object_sam3.postsam3_pipeline --dataset "$DATASET_NAME" --skip-caption

# Step 4: Run captioning
python -m src.captioning.orchestrator --dataset "$DATASET_NAME"

# Step 5: Run floor detection
bash run_floordetection.sh "$DATASET_NAME"

echo "============================================"
echo "All steps completed successfully for dataset: $DATASET_NAME"
