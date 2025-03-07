#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# The model weights can be found in https://console.cloud.google.com/storage/browser/tips_data
ALL_CHECKPOINTS=(
  "tips_oss_s14_highres_distilled"
  "tips_oss_b14_highres_distilled"
  "tips_oss_l14_highres_distilled"
  "tips_oss_so400m14_highres_largetext_distilled"
  "tips_oss_g14_lowres"
  "tips_oss_g14_highres"
)

echo "Downloading the tokenizer."
wget https://storage.googleapis.com/tips_data/v1_0/checkpoints/tokenizer.model

for CHECKPOINT in "${ALL_CHECKPOINTS[@]}"; do
  echo "Downloading ${CHECKPOINT} (vision encoder weights)"
  wget https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/${CHECKPOINT}_vision.npz
  echo "Downloading ${CHECKPOINT} (text encoder weights)"
  wget https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/${CHECKPOINT}_text.npz
done
