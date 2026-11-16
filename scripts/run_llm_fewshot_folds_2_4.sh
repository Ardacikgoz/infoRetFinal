#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/eval_local_llm.py"
OUTPUT_DIR="$PROJECT_ROOT/newOutputs/llm_zero"

# Shared settings (same style you used)
MODE="few-shot"
MODEL_NAME="phi3:mini"
FEW_SHOT_K=3
FEW_SHOT_SOURCE_FOLD=4
TIMEOUT_SECONDS=180
NUM_PREDICT=60
REPAIR_NUM_PREDICT=40
REPAIR_TIMEOUT_SECONDS=180
REPAIR_MAX_RETRIES=2
MAX_REPAIR_PARSE_ERRORS=100
SAMPLE_CHUNK_SIZE=500

for FOLD in 2 3 4; do
  echo "=== Running fold ${FOLD} ==="
  "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --fold "$FOLD" \
    --mode "$MODE" \
    --few-shot-k "$FEW_SHOT_K" \
    --few-shot-source-fold "$FEW_SHOT_SOURCE_FOLD" \
    --model-name "$MODEL_NAME" \
    --use-http \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --num-predict "$NUM_PREDICT" \
    --repair-parse-errors \
    --repair-num-predict "$REPAIR_NUM_PREDICT" \
    --repair-timeout-seconds "$REPAIR_TIMEOUT_SECONDS" \
    --repair-max-retries "$REPAIR_MAX_RETRIES" \
    --max-repair-parse-errors "$MAX_REPAIR_PARSE_ERRORS" \
    --sample-chunk-size "$SAMPLE_CHUNK_SIZE" \
    --output-dir "$OUTPUT_DIR"
done

echo "Done: folds 2, 3, 4 finished."
