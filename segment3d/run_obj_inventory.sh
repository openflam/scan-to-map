#!/bin/bash
# Run the per-object inventory pipeline.
#
# Usage:
#   bash run_obj_inventory.sh <dataset_name> [object1] [object2] ...
#
# Examples:
#   bash run_obj_inventory.sh ProjectLabStudio_inv_method
#   bash run_obj_inventory.sh ProjectLabStudio_inv_method "drill press" "table" "chair"
#
# Step 1 (sam3 env):          run SAM3 video predictor on per-object frame sequences
# Step 2+ (segment3d-env):    associate → mask graph → bbox → segment crops → caption → clip

set -e

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
if [ -z "$1" ]; then
    echo "Error: Dataset name is required"
    echo "Usage: $0 <dataset_name> [object1] [object2] ..."
    exit 1
fi

DATASET="$1"
shift  # remaining args are object names

echo "=================================================="
echo "  OBJECT INVENTORY PIPELINE"
echo "  Dataset: $DATASET"
if [ "$#" -gt 0 ]; then
    echo "  Objects: $*"
else
    echo "  Objects: all"
fi
echo "=================================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1: Run SAM3  (conda env: sam3)
# ---------------------------------------------------------------------------
echo ">>> STEP 1: SAM3 segmentation  [env: sam3]"

SAM3_ARGS="--dataset $DATASET"
if [ "$#" -gt 0 ]; then
    SAM3_ARGS="$SAM3_ARGS --objects"
    for obj in "$@"; do
        SAM3_ARGS="$SAM3_ARGS \"$obj\""
    done
fi

eval conda run -n sam3 --no-capture-output \
    python -m src.per_object_sam3.sam3_runner $SAM3_ARGS

echo ""
echo ">>> Step 1 complete."
echo ""

# ---------------------------------------------------------------------------
# Step 2-5: Post-SAM3 pipeline without captioning  (conda env: segment3d-env)
# ---------------------------------------------------------------------------
echo ">>> STEPS 2-5: Post-SAM3 pipeline (no caption)  [env: segment3d-env]"

conda run -n segment3d-env --no-capture-output \
    python -m src.per_object_sam3.postsam3_pipeline --dataset "$DATASET" --skip-caption

echo ""
echo ">>> Steps 2-5 complete."
echo ""

# ---------------------------------------------------------------------------
# Step 6: Captioning  (conda env: segment3d-env)
# ---------------------------------------------------------------------------
echo ">>> STEP 6: Captioning  [env: segment3d-env]"

conda run -n segment3d-env --no-capture-output \
    python -m src.captioning.orchestrator --dataset "$DATASET"

echo ""
echo "=================================================="
echo "  ALL STEPS COMPLETE  –  $DATASET"
echo "=================================================="
