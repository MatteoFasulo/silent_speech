#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR=${1:-"./models/transduction_model"}
OUTPUT_FILE=${2:-"wer_results.csv"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"python evaluate.py --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory evaluation_output --testset_file testset_origdev.json"}

echo "model,wer" >"$OUTPUT_FILE"

shopt -s nullglob
for ckpt in "$CKPT_DIR"/*.pt; do
  model_name="$(basename "$ckpt")"
  echo "Processing model: $model_name"

  if out="$($EVAL_SCRIPT --models "$ckpt" 2>&1)"; then
    wer=$(printf '%s\n' "$out" | tr -d '\r' | \
         grep -Eo 'WER:[[:space:]]*[0-9]+(\.[0-9]+)?' | \
         grep -Eo '[0-9]+(\.[0-9]+)?')

    if [ -n "$wer" ]; then
      printf '%s,%s\n' "$model_name" "$wer" >>"$OUTPUT_FILE"
      echo "WER: $wer"
    else
      echo "Warning: WER not found in output for $model_name" >&2
      printf '%s,ERROR\n' "$model_name" >>"$OUTPUT_FILE"
    fi
  else
    echo "Error: Evaluation failed for $model_name" >&2
    printf '%s,FAILED\n' "$model_name" >>"$OUTPUT_FILE"
  fi

  echo ""
done
shopt -u nullglob

echo "Results saved to $OUTPUT_FILE"
