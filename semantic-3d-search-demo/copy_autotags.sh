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
SRC2="$SCRIPT_DIR/../data/$DATASET/polycam_data/raw.glb"
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

cp -v -- "$SRC1" "$DEST_DIR/"
cp -v -- "$SRC2" "$DEST_DIR/"

echo "Copied to $DEST_DIR"