#!/bin/bash
set -e

TASK=$1   # expects: emg2audio or emg2text
EXTRA_ARGS="${@:2}"

# --start_training_from /usr/scratch2/sassauna2/msc25f18/checkpoints/pretraining/20@rope@gelu/20@rope@gelu-epoch=49-val_loss=0.0091.ckpt

if [[ -z "$TASK" ]]; then
  echo "Usage: $0 <task>"
  echo "  task = emg2audio | emg2text"
  exit 1
fi

SEEDS=(1379 8642 5601 1234 42)

for seed in "${SEEDS[@]}"; do
  echo "Running task=$TASK with seed=$seed"

  if [[ "$TASK" == "emg2audio" ]]; then
    python transduction_model.py \
      --hifigan_checkpoint hifigan_finetuned/checkpoint \
      --output_directory "./models/transduction_model/" \
      --ckpt_directory "$SCRATCH/checkpoints/finetuning/$TASK/" \
      --seed "$seed" \
      $EXTRA_ARGS

  elif [[ "$TASK" == "emg2text" ]]; then
    python recognition_model.py \
      --output_directory "./models/recognition_model/" \
      --ckpt_directory "$SCRATCH/checkpoints/finetuning/$TASK/" \
      --lm_directory "$SCRATCH/datasets/KenLM/" \
      --seed "$seed" \
      $EXTRA_ARGS

  else
    echo "Unknown task: $TASK"
    exit 1
  fi

done
