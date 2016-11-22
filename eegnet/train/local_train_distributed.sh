#!/bin/bash

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sets up the development environment for Cloud ML on the Cloud Datalab Docker
# container.

# Quit early if any command fails.
set -ex

JOB_NAME=eegnet_local_distributed_${USER}_$(date +%Y%m%d_%H%M%S)
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_BUCKET=gs://${PROJECT_ID}-ml
TRAIN_PATH=${TRAIN_BUCKET}/${JOB_NAME}

gcloud beta ml local train \
  --package-path=trainer \
  --module-name=trainer.task \
  --distributed \
  -- \
  --dataset_dir="content/dataset/train/*.tfr" \
  --log_dir="content/logs"