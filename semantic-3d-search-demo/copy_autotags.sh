#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 DATASET" >&2
    exit 2
fi

DATASET=$1

# Resolve script directory so relative paths are consistent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC1="$SCRIPT_DIR/../outputs/$DATASET/bbox_corners.json"
SRC2="$SCRIPT_DIR/../outputs/$DATASET/occupancy_bbox.json"
ALIGNED_GLB="$SCRIPT_DIR/../data/$DATASET/alignment/transformed_rotated.glb"
if [[ -f "$ALIGNED_GLB" ]]; then
    SRC3="$ALIGNED_GLB"
    echo "Copying aligned GLB: $ALIGNED_GLB"
else
    SRC3="$SCRIPT_DIR/../data/$DATASET/polycam_data/raw.glb"
    echo "Warning: copying raw GLB: $SRC3"
fi
DEST_DIR="$SCRIPT_DIR/public/data"

mkdir -p "$DEST_DIR"

if [[ ! -f "$SRC1" ]]; then
    echo "Source file not found: $SRC1" >&2
    exit 3
fi

if [[ ! -f "$SRC2" ]]; then
    echo "Source file not found: $SRC2" >&2
    exit 4
fi

if [[ ! -f "$SRC3" ]]; then
    echo "Source file not found: $SRC3" >&2
    exit 4
fi

cp -v -- "$SRC1" "$DEST_DIR/"
cp -v -- "$SRC2" "$DEST_DIR/"
cp -v -- "$SRC3" "$DEST_DIR/raw.glb"

echo "Copied to $DEST_DIR"