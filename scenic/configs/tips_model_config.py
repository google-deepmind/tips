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

# TIPS-v1 models.
_VARIANT_DICT_V1 = {
    'tips_oss_g14_highres': 'g/14',
    'tips_oss_g14_lowres': 'g/14',
    'tips_oss_so400m14_highres_largetext_distilled': 'So400m/14',
    'tips_oss_l14_highres_distilled': 'L/14',
    'tips_oss_b14_highres_distilled': 'B/14',
    'tips_oss_s14_highres_distilled': 'S/14',
}

# TIPS-v2 models.
_VARIANT_DICT_V2 = {
    'tips_v2_g14': 'g/14',
    'tips_v2_so14': 'So400m/14',
    'tips_v2_l14': 'L/14',
    'tips_v2_b14': 'B/14',
}


def get_config(variant: str):
  """Returns the TIPS model config."""
  config = ml_collections.ConfigDict()

  if variant in _VARIANT_DICT_V1:
    config.variant = _VARIANT_DICT_V1[variant]
    pos_embed_shape = (16, 16)
  elif variant in _VARIANT_DICT_V2:
    config.variant = _VARIANT_DICT_V2[variant]
    pos_embed_shape = (32, 32)
  else:
    all_variants = list(_VARIANT_DICT_V1.keys()) + list(_VARIANT_DICT_V2.keys())
    raise ValueError(
        f'Unknown TIPS variant: {variant}. Please choose one of: '
        f'{all_variants}')

  config.rgb_mean = _MEAN_RGB
  config.rgb_std = _STDDEV_RGB

  config.pooling = 'tok'
  config.pos_interpolation_method = 'bilinear'

  config.positional_embedding = ml_collections.ConfigDict()
  config.positional_embedding.shape = pos_embed_shape

  # TIPS defaults to 2 CLS tokens.
  config.num_cls_tokens = 2

  # TIPS text encoder config.
  config.text_max_seq_length = 64
  config.text_vocab_size = 32000

  return config