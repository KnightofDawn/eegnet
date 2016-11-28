#!/bin/bash

# Quit early if any command fails.
set -ex

python trainer/task.py \
  --dataset_dir="/content/dataset/eval/*.tfr" \
  --checkpoint_dir="/content/checkpoints/gcloud_4th_eegnetv2_split1_batch16" \
  --log_dir="/content/logs" \
  --batch_size=1 \
  --num_splits=1
  