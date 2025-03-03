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

"""TIPS model config."""

import ml_collections

_MEAN_RGB = [0., 0., 0.]
_STDDEV_RGB = [1., 1., 1.]

# The 'g' variant refers to the DINO-v2 'giant2', which differs from ViT-g.
# The differences are highlighted in https://arxiv.org/pdf/2304.07193 Section 5.
_VARIANT_DICT = {
    'tips_oss_g14_highres': 'g/14',
    'tips_oss_g14_lowres': 'g/14',
    'tips_oss_so400m14_highres_largetext_distilled': 'So400m/14',
    'tips_oss_l14_highres_distilled': 'L/14',
    'tips_oss_b14_highres_distilled': 'B/14',
    'tips_oss_s14_highres_distilled': 'S/14',
}


def get_config(variant: str):
  """Returns the TIPS model config."""
  config = ml_collections.ConfigDict()
  if variant not in _VARIANT_DICT:
    raise ValueError(
        f'Unknown TIPS variant: {variant}. Please choose one of: '
        f'{list(_VARIANT_DICT.keys())}')

  config.variant = _VARIANT_DICT[variant]
  config.rgb_mean = _MEAN_RGB
  config.rgb_std = _STDDEV_RGB

  config.pooling = 'tok'
  config.pos_interpolation_method = 'bilinear'

  # TIPS defaults to 2 CLS tokens.
  config.num_cls_tokens = 2

  return config
