#!/bin/bash
set -e

SEEDS=(1379 8642 5601 1234 42)

for seed in "${SEEDS[@]}"; do
  echo "Running training with seed=$seed"
  python transduction_model.py --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory "./models/transduction_model/" --start_training_from /capstor/scratch/cscs/mfasulo/checkpoints/pretraining/20@rope@gelu/20@rope@gelu-epoch=49-val_loss=0.0091.ckpt --seed "$seed"
done
