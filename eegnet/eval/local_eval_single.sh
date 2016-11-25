#!/bin/bash

# Quit early if any command fails.
set -ex

gcloud beta ml local train \
  --package-path=trainer \
  --module-name=trainer.task \
  -- \
  --dataset_dir="/content/dataset/eval/*.tfr" \
  --checkpoint_dir="/content/logs/eval_test" \
  --log_dir="/content/logs/eval_run" \
  --batch_size=1 \
  --num_splits=1 \
  --num_iters=20
  