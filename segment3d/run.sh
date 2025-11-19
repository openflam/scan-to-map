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

# Step 1: Run main pipeline without captioning
python main.py --dataset "$DATASET_NAME" --skip-caption


# Step 2: Run captioning
python -m src.captioning.orchestrator --dataset "$DATASET_NAME"

echo "============================================"
echo "All steps completed successfully for dataset: $DATASET_NAME"
