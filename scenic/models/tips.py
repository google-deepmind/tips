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

"""The TIPS model definition."""

import typing as t

import flax.linen as nn
import jax.numpy as jnp

from tips.scenic.models import text
from tips.scenic.models import vit


class VisionEncoder(nn.Module):
  """TIPS vision encoder based on ViT."""

  variant: str
  pooling: str = 'tok'
  num_cls_tokens: int = 2  # TIPS defaults to 2 CLS tokens.
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  dtype: t.Any = jnp.float32

  def setup(self):
    super().setup()

    self.encoder = vit.ViT(
        variant=self.variant,
        num_cls_tokens=self.num_cls_tokens,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        )
    self.patches = self.encoder.patches

  def pool_features(self, x: jnp.ndarray)-> t.Tuple[jnp.ndarray, jnp.ndarray]:
    """Extracts the spatial and vector features from the backhone.

    Currently supports only 'tok' pooling (CLS tokens). The CLS tokens are
    always prepended to the spatial (patch) tokens.

    Args:
      x: The input features.

    Returns:
      x_patch: The spatial features.
      x_vec: The vector embedding(s).
    """
    if self.pooling == 'tok':
      x_vec = x[:, :self.num_cls_tokens, :]
      x_patch = x[:, self.num_cls_tokens:, :]
    else:
      raise ValueError(f'Invalid pooling: {self.pooling}')
    return x_patch, x_vec

  def reshape_spatial_features(
      self, x: jnp.ndarray, h: int, w: int) -> jnp.ndarray:
    """Re-shapes the spatial features according to the initial dimensions."""
    fh = h // self.patches[0]
    fw = w // self.patches[1]
    bs, l, f = x.shape
    if l != fh * fw:
      raise ValueError(f'Invalid shape: {x.shape}')
    return x.reshape(bs, fh, fw, f)

  @nn.compact
  def __call__(
      self, x: jnp.ndarray, *, train: bool, debug: bool = False
  ) -> t.Tuple[jnp.ndarray, jnp.ndarray]:
    del debug
    x = vit.maybe_center_pad(
        x, patch_h=self.patches[0], patch_w=self.patches[1])
    h, w = x.shape[1:3]  # w, h of images after padding.
    x = self.encoder(x, train=train)
    x_patch, x_vec = self.pool_features(x)
    x_patch = self.reshape_spatial_features(x_patch, h, w)

    return x_patch, x_vec


class TextEncoder(nn.Module):
  """TIPS Text encoder."""

  variant: str
  vocab_size: int = 32_000
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: t.Any = jnp.float32
  scale_sqrt_depth: bool = True  # Default param in PAX experiments.

  def setup(self):
    super().setup()
    text_config = vit.get_vit_config(self.variant)
    text_config['num_layers'] = 12
    # The text tower layers are fixed independent of vision tower size.
    # Exception: The So400m/14 text tower is a symmetric copy of the vision
    # tower.
    self.num_layers = 12
    if self.variant != 'So400m/14':
      self.num_layers = text_config['num_layers']
    self.embedding_dim = text_config['hidden_size']
    self.mlp_dim = text_config['mlp_dim']
    self.num_heads = text_config['num_heads']
    self.embedder = text.Embedding(
        name='token_emb',
        num_classes=self.vocab_size,
        embedding_dim=self.embedding_dim,
        scale_sqrt_depth=self.scale_sqrt_depth)
    self.pos_embedder = text.PositionalEmbedding(
        embedding_dim=self.embedding_dim)
    self.transformer = text.StackedTransformer(
        name='transformer',
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
    )
    self.pooling = text.GlobalAvgPooling(pooling_dims=[1])
    self.norm = nn.LayerNorm(dtype=self.dtype, name='text_encoder_norm')

  def __call__(
      self,
      ids: jnp.ndarray,
      paddings: jnp.ndarray,
      train: bool,
  ) -> jnp.ndarray:
    """Applies TextEncoder module."""
    _, seq_length = ids.shape
    mask = (paddings == 0).astype(jnp.int32)
    x = self.embedder(ids)
    x = x + self.pos_embedder(seq_length=seq_length)
    x = self.transformer(x, mask, deterministic=not train)
    x = self.norm(x)
    x = self.pooling(x, compatible_paddings=paddings[:, :, jnp.newaxis])
    return x

