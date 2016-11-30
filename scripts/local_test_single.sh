#!/bin/bash

# Quit early if any command fails.
set -ex

cd /content && \
gcloud beta ml local train \
  --package-path=src \
  --module-name=src.test \
  -- \
  --dataset_dir="/content/dataset/test/*.tfr" \
  --checkpoint_dir="/content/checkpoints/gcloud_3rd_pool2400_split1_batch7" \
  --log_dir="/content/logs" \
  --batch_size=1 \
  --num_splits=1

