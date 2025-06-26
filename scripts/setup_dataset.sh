#!/bin/bash
set -e

echo "[1/3] Downloading dataset via kagglehub..."
python3 -c "
import kagglehub
path = kagglehub.dataset_download('moltean/fruits')
print(path)
" > dataset_path.txt

DATASET_PATH=$(cat dataset_path.txt)
TARGET_DIR=../data/fruits

echo "[2/3] Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "[3/3] Copying dataset from $DATASET_PATH to $TARGET_DIR"
cp -r "$DATASET_PATH"/* "$TARGET_DIR"

rm dataset_path.txt
echo "✅ Dataset ready at $TARGET_DIR"
