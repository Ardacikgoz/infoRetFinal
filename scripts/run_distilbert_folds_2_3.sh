#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train_distilbert.py"

echo "Running DistilBERT for fold 2..."
"$PYTHON_BIN" "$TRAIN_SCRIPT" --fold 2

echo "Running DistilBERT for fold 3..."
"$PYTHON_BIN" "$TRAIN_SCRIPT" --fold 3

echo "Done. Folds 2 and 3 completed."
