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

"""Checkpoint helpers functions."""

import logging
import typing as t
import flax
import numpy as np


def load_checkpoint(
    checkpoint_path: str,
    params_to_load: t.Dict[str, np.ndarray],
    strict: bool = True,
) -> t.Dict[str, np.ndarray]:
  """Loads a TIPS checkpoint and checks that the parameters are compatible."""
  params_to_load_flat = flax.traverse_util.flatten_dict(params_to_load, sep='/')
  params_loaded_flat = dict(np.load(checkpoint_path, allow_pickle=True))

  # Check that params to load are in the checkpoint, and have identical shapes.
  for k in params_to_load_flat:
    if k not in params_loaded_flat:
      raise ValueError(f'Param {k} not found in checkpoint.')
    if params_loaded_flat[k].shape != params_to_load_flat[k].shape:
      raise ValueError(f'Param {k} has wrong shape in checkpoint.')

  # Check that the checkpoint does not contain extra parameter groups.
  for k in params_loaded_flat:
    if k not in params_to_load_flat:
      if strict:
        raise ValueError(f'Param {k} not found in params_to_load.')
      else:
        logging.warning('Param %s not found in params_to_load.', k)

  return flax.traverse_util.unflatten_dict(params_loaded_flat, sep='/')
