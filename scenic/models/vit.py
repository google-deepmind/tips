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

"""Standard ViT model definition."""

import logging
import math
import typing as t

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers

Initializer = t.Callable[[jnp.ndarray, t.Sequence[int], jnp.dtype], jnp.ndarray]


def get_vit_config(variant: str) -> t.Dict[str, t.Any]:
  v, patch = variant.split('/')
  return {
      # pylint:disable=line-too-long
      'hidden_size': {'S': 384, 'B': 768, 'L': 1024, 'So400m': 1152, 'g': 1536}[v],
      'num_layers': {'S': 12, 'B': 12, 'L': 24, 'So400m': 27, 'g': 40}[v],
      'mlp_dim': {'S': 1536, 'B': 3072, 'L': 4096, 'So400m': 4304, 'g': 6144}[v],
      'num_heads': {'S': 6, 'B': 12, 'L': 16, 'So400m': 16, 'g': 24}[v],
      'patch_size': (int(patch), int(patch)),
      'ffn_layer': {'S': 'mlp', 'B': 'mlp', 'L': 'mlp', 'So400m': 'mlp', 'g': 'swiglu'}[v],
      # pylint:enable=line-too-long
  }


def maybe_center_pad(x: jnp.ndarray, patch_h: int, patch_w: int):
  """Pads the input to the next multiple of the patch size."""
  h_old, w_old = x.shape[1:3]
  pad_h = math.ceil(h_old / patch_h) * patch_h - h_old
  pad_w = math.ceil(w_old / patch_w) * patch_w - w_old
  if pad_w > 0 or pad_h > 0:
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    logging.info(
        'Applying center padding (%d, %d), (%d, %d)',
        pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
    x = jnp.pad(
        x, ((0, 0),
            (pad_h_top, pad_h_bottom),
            (pad_w_left, pad_w_right),
            (0, 0)))
  return x


class ToTokenSequence(nn.Module):
  """Transform a batch of views into a sequence of tokens."""

  patches: ml_collections.ConfigDict
  hidden_size: int
  num_cls_tokens: int = 0
  posembs: t.Tuple[int, int] = (16, 16)
  pos_interpolation_method: str = 'bilinear'

  def add_positional_encodings(self, x: jnp.ndarray) -> jnp.ndarray:
    """Support a few variants for sinsuoidal 2D position embeddings."""
    n, h, w, c = x.shape
    posemb = self.param(
        'posembed_input',
        nn.initializers.normal(stddev=1/np.sqrt(c)),
        (1, self.posembs[0], self.posembs[1], c), x.dtype)
    # Interpolate the positional encodings.
    if (h, w) != self.posembs:
      posemb = jax.image.resize(
          posemb, (1, h, w, c), self.pos_interpolation_method)
    x = x + posemb
    x = jnp.reshape(x, [n, h * w, c])

    assert x.ndim == 3  # Shape is `[batch, len, emb]`.
    return x

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, seqlen: int = -1):

    fh, fw = self.patches
    # Extracting patches and then embedding is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(x)

    # Add positional encodings.
    x = self.add_positional_encodings(x)

    # Add extra "cls" tokens.
    if self.num_cls_tokens > 0:
      n, _, c = x.shape
      cls_tok = self.param(
          'cls',
          nn.initializers.zeros,
          (1, self.num_cls_tokens, c),
          x.dtype)
      cls_tok = jnp.tile(cls_tok, [n, 1, 1])
      x = jnp.concatenate([cls_tok, x], axis=1)
    return x


class FFNSwiGluFused(nn.Module):
  """SwiGlu variant of the feed-forward block.

  https://arxiv.org/abs/2002.05202v1
  """

  mlp_dim: int
  out_dim: t.Optional[int] = None
  dropout_rate: float = 0.0
  use_bias: bool = False
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.zeros
  precision: t.Optional[jax.lax.Precision] = None
  dtype: jnp.ndarray = jnp.float32

  def _hidden_layer(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py#L57  # pylint: disable=line-too-long
    mlp_dim = (int(self.mlp_dim * 2 / 3) + 7) // 8 * 8
    xw = nn.Dense(
        mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
    )(inputs)
    xv = nn.Dense(
        mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
    )(inputs)
    xw = nn.swish(xw)
    x = xw * xv
    return x

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, *, deterministic: bool
  ) -> jnp.ndarray:
    """Applies FFN module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = self._hidden_layer(inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(x)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic)
    return output


class VisionEncoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows
      from 0 to the provided value.
    ffn_layer: type of the feed-forward layer. Options are 'mlp', 'swiglufused'.

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: t.Any = jnp.float32
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  ffn_layer: str = 'mlp'

  def setup(self):
    super().setup()

    if self.ffn_layer == 'mlp':
      ffn_layer = attention_layers.MlpBlock(
          mlp_dim=self.mlp_dim,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          activation_fn=nn.gelu,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='MlpBlock_0')
    elif self.ffn_layer == 'swiglu':
      ffn_layer = FFNSwiGluFused(
          mlp_dim=self.mlp_dim,
          dtype=self.dtype,
          use_bias=True,
          dropout_rate=self.dropout_rate,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='FFNSwiGluFused_0')
    else:
      raise ValueError(f'Unsupported ffn_layer: {self.ffn_layer}')
    self.ffn = ffn_layer
    self.ln_0 = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_0')
    self.ln_1 = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_1')
    self.attention = nn.MultiHeadDotProductAttention(
        name='MultiHeadDotProductAttention_0',
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate)

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = self.ln_0(inputs)
    x = self.attention(x, x, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    # MLP block.
    y = self.ln_1(x)
    y = self.ffn(y, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


class StackedTransformer(nn.Module):
  """Stacked transformer."""

  mlp_dim: int
  num_layers: int
  num_heads: int
  ffn_layer: str = 'mlp'
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  dtype: t.Any = jnp.float32

  def setup(self):
    encoder_blocks = []
    for lyr in range(self.num_layers):
      encoder_blocks.append(
          VisionEncoder1DBlock(
              mlp_dim=self.mlp_dim,
              num_heads=self.num_heads,
              dropout_rate=self.dropout_rate,
              attention_dropout_rate=self.attention_dropout_rate,
              stochastic_depth=(lyr / max(self.num_layers - 1, 1))
              * self.stochastic_depth,
              name=f'encoderblock_{lyr}',
              ffn_layer=self.ffn_layer,
              dtype=self.dtype))
    self.encoder_blocks = encoder_blocks

  def __call__(
      self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies StackedTransformer module."""
    for block in self.encoder_blocks:
      x = block(x, deterministic=deterministic)
    return x


class ViT(nn.Module):
  """Dense Features backbone based on ViT."""

  variant: str
  freeze_backbone: bool = False
  num_cls_tokens: int = 1
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: t.Any = jnp.float32

  def setup(self):
    super().setup()
    vit_config = get_vit_config(self.variant)
    self.patches = vit_config['patch_size']
    self.hidden_size = vit_config['hidden_size']
    self.num_layers = vit_config['num_layers']
    self.mlp_dim = vit_config['mlp_dim']
    self.num_heads = vit_config['num_heads']
    self.ffn_layer = vit_config['ffn_layer']

    # Setup for layers.
    self.token_fn = ToTokenSequence(
        name='ToTokenSequence_0',
        patches=self.patches,
        hidden_size=self.hidden_size,
        num_cls_tokens=self.num_cls_tokens,
        posembs=(16, 16),
        )
    self.norm = nn.LayerNorm(name='encoder_norm')
    self.transformer = StackedTransformer(
        name='transformer',
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        ffn_layer=self.ffn_layer,
    )

  @nn.compact
  def __call__(
      self, x: jnp.ndarray, *, train: bool, debug: bool = False) -> jnp.ndarray:
    del debug
    logging.info('train=%s shape before padding=%s', train, x.shape)
    x = maybe_center_pad(x, patch_h=self.patches[0], patch_w=self.patches[1])
    logging.info('train=%s shape after padding=%s', train, x.shape)

    x = self.token_fn(x)
    x = self.transformer(x, deterministic=not train)
    x = self.norm(x)

    return x
