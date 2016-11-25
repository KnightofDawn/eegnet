#!/bin/bash

# Quit early if any command fails.
set -ex

python trainer/task.py \
  --dataset_dir="/shared/dataset/test/*.tfr" \
  --checkpoint_dir="/shared/checkpoints/gcloud_2nd_pool2400_split1_batch3" \
  --log_dir="/shared/logs" \
  --batch_size=1 \
  --num_splits=1
  