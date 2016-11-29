#!/bin/bash

# Quit early if any command fails.
set -ex

python trainer/task.py \
  --dataset_dir="/shared/dataset/train/*.tfr" \
  --checkpoint_dir="/shared/checkpoints/gcloud_3rd_pool2400_split1_batch7" \
  --log_dir="/shared/logs" \
  --batch_size=1 \
  --num_splits=1
  