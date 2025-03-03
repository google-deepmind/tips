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

"""Text-encoder related modules."""

import math
import typing as t

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.layers import nn_layers
import tensorflow as tf
import tensorflow_text


Initializer = t.Callable[[jnp.ndarray, t.Sequence[int], jnp.dtype], jnp.ndarray]


class Tokenizer(object):
  """A simple tokenizer."""

  def __init__(self, tokenizer_path: str):
    """Initializes the tokenizer."""
    with open(tokenizer_path, 'rb') as f:
      model = f.read()
    self.tokenizer = tensorflow_text.SentencepieceTokenizer(
        model=model, add_eos=False, add_bos=False)

  def tokenize(self, input_text, max_len=64):
    tokens = self.tokenizer.tokenize(tf.strings.lower(input_text)).to_tensor()
    curr_len = tokens.shape[1]
    is_padding = tf.zeros((tokens.shape[0], max_len))
    if curr_len > max_len:
      tokens = tokens[:, :max_len]
    else:
      padding_len = max_len - curr_len
      tokens = tf.pad(tokens, [[0, 0], [0, padding_len]], constant_values=0)
      is_padding = tf.cast(tokens == 0, tf.int32)
    return tokens.numpy(), is_padding.numpy()


class Embedding(nn.Module):
  """A simple embedding layer that performs embedding lookups from ids.

  Simple version of
  https://github.com/google/praxis/blob/main/praxis/layers/embedding_softmax.py#L97

  Attributes:
    num_classes: Number of tokens in the vocabulary.
    embedding_dim: Depth of the embedding output.
    scale_sqrt_depth: If set to True, activations are scaled with
      sqrt(embedding_dim) in emb_lookup.
  """

  num_classes: int = 0
  embedding_dim: int = 0
  scale_sqrt_depth: bool = True

  def setup(self) -> None:
    assert self.num_classes > 0
    assert self.embedding_dim > 0

    self.emb_var = self.param(
        'emb_var',
        nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0),
        (self.num_classes, self.embedding_dim),
        jnp.float32)

  def emb_lookup(self, ids: jnp.ndarray) -> jnp.ndarray:
    embs = self.emb_var[ids]

    if self.scale_sqrt_depth:
      embs *= self.embedding_dim**0.5

    return embs

  def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
    return self.emb_lookup(ids)


class PositionalEmbedding(nn.Module):
  """Generates fixed position embedding for a given 1-d sequence.

  Simplified version of
  https://github.com/google/praxis/blob/main/praxis/layers/embedding_softmax.py#L1011

  Attributes:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dim: Dimension of the embedding to be generated.
  """

  min_timescale: int = 1
  max_timescale: int = 10_000
  embedding_dim: int = 0

  def __call__(
      self, seq_length: int | None = None, position: jnp.ndarray | None = None
  ) -> jnp.ndarray:
    """Generates a jnp.ndarray of sinusoids with different frequencies.

    Args:
      seq_length: an optional Python int definiing the output sequence length.
        if the `position` argument is specified.
      position:   [B, seq_length], optional position for each token in the
        sequence, only required when the sequence is packed.

    Returns:
      [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
    """
    if position is None:
      assert seq_length is not None
      # [1, seqlen]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    else:
      assert position.ndim == 2, position.shape

    num_timescales = self.embedding_dim // 2
    log_timescale_increment = math.log(
        float(self.max_timescale) / float(self.min_timescale)
    ) / jnp.maximum(jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1)
    inv_timescales = self.min_timescale * jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = (
        position[:, :, jnp.newaxis]
        * inv_timescales[jnp.newaxis, jnp.newaxis, :]
    )
    signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2)
    # Force usage of `np` rather than `jnp` to compute static values at trace
    # time.
    signal = jnp.pad(
        signal, [[0, 0], [0, 0], [0, np.mod(self.embedding_dim, 2)]]
    )
    return signal


class GlobalAvgPooling(nn.Module):
  """Performs a simple global pooling over the input with optional paddings.

  Attributes:
    pooling_dims: A list of dims to perform pooling over.
    keepdims: If True, keep dimension of inputs after pooling.
  """
  pooling_dims: t.Sequence[int] | None = None
  epsilon: float = 1e-8

  def setup(self) -> None:
    if self.pooling_dims is None:
      raise ValueError('pooling_dims must be set as a list.')
    else:
      if not all([p_dims >= 0 for p_dims in self.pooling_dims]):
        raise ValueError('pooling_dims must be non-negative integers.')

  def __call__(
      self,
      inputs: jnp.ndarray,
      compatible_paddings: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies global average spatial pooling to inputs.

    Args:
      inputs: An input tensor.
      compatible_paddings: paddings of inputs with shapes compatible
        with inputs, e.g. compatible_paddings with shape [B, 1] for inputs with
        shape [B, D].

    Returns:
      Output tensor with global pooling applied.
    """
    padded_value = jnp.zeros(shape=(), dtype=inputs.dtype)
    padded_value = jnp.ones_like(inputs) * padded_value
    inputs = jnp.where(compatible_paddings > 0, padded_value, inputs)
    valid_inputs = (
        jnp.sum(
            1.0 - compatible_paddings,
            self.pooling_dims,
            keepdims=True,
            dtype=inputs.dtype)
        + self.epsilon)
    inputs_sum = jnp.sum(inputs, self.pooling_dims, keepdims=True)
    outputs = jnp.divide(inputs_sum, valid_inputs).astype(inputs.dtype)
    outputs = jnp.squeeze(outputs, axis=self.pooling_dims)
    return outputs


class MlpBlockWithMask(nn.Module):
  """Transformer MLP / feed-forward block that supports masking."""

  mlp_dim: int
  out_dim: t.Optional[int] = None
  dropout_rate: float = 0.1
  use_bias: bool = True
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  activation_fn: t.Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  precision: t.Optional[jax.lax.Precision] = None
  dtype: jnp.ndarray = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, mask, deterministic: bool):
    """Applies Transformer MlpBlock with mask module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            inputs)
    x = nn_layers.IdentityLayer(name='mlp1')(self.activation_fn(x))
    x = x * mask[..., None]  # First masking.
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(x)
    output = output * mask[..., None]  # Second masking.
    output = nn_layers.IdentityLayer(name='mlp2')(output)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic)
    return output


class TextEncoder1DBlock(nn.Module):
  """Transformer text encoder layer.

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
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, mask: jnp.ndarray, deterministic: bool
  ) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      mask: Input mask.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(name='LayerNorm_0', dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate)(
            x, x, mask=mask[:, jnp.newaxis, jnp.newaxis, :])
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_1')(x)
    mlp0 = MlpBlockWithMask(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.relu,  # ReLU is the choice for the PAX experiments.
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name='MlpBlock_0'
    )
    y = mlp0(y, mask=mask, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return x + y


class StackedTransformer(nn.Module):
  """Stacked transformer."""

  mlp_dim: int
  num_layers: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: t.Any = jnp.float32

  def setup(self):
    encoder_blocks = []
    for lyr in range(self.num_layers):
      encoder_blocks.append(
          TextEncoder1DBlock(
              mlp_dim=self.mlp_dim,
              num_heads=self.num_heads,
              dropout_rate=self.dropout_rate,
              attention_dropout_rate=self.attention_dropout_rate,
              stochastic_depth=(
                  (lyr / max(self.num_layers - 1, 1)) * self.stochastic_depth),
              name=f'encoderblock_{lyr}',
              ))
      self.encoder_blocks = encoder_blocks

  def __call__(
      self, x: jnp.ndarray, mask: jnp.ndarray, deterministic: bool
  ) -> jnp.ndarray:
    """Applies StackedTransformer module."""
    for block in self.encoder_blocks:
      x = block(x, mask, deterministic=deterministic)
    return x

