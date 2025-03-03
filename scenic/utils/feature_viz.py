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

"""Visualization helpers for features."""

import typing as t
import jax.numpy as jnp
import numpy as np
from sklearn import decomposition


_ArrayLike = t.Union[np.ndarray, jnp.ndarray]


def normalize(x, order: int = 2):
  return x / np.linalg.norm(
      x, ord=order, axis=-1, keepdims=True).clip(min=1e-3)


class PCAVisualizer:
  """PCA visualizer."""

  def __init__(
      self,
      features: _ArrayLike,
      n_samples: int = 100000,
      n_components: int = 3) -> None:
    """Creates a PCA object for visualizing features of shape [..., F]."""
    features = np.array(features)
    pca_object = decomposition.PCA(n_components=n_components)
    features = features.reshape([-1, features.shape[-1]])
    features = features[np.random.randint(0, features.shape[0], n_samples), :]
    pca_object.fit(features)
    self.pca_object = pca_object
    self.n_components = n_components

  def __call__(self, features: _ArrayLike) -> np.ndarray:
    """Apply PCA to features of shape [..., F]."""
    features = np.array(features)
    features_pca = self.pca_object.transform(
        features.reshape([-1, features.shape[-1]])
        ).reshape(features.shape[:-1] + (self.n_components,))
    return normalize(features_pca) * 0.5 + 0.5
