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

"""Text encoder implementation in PyTorch."""

import typing as t

import tensorflow as tf
import tensorflow_text
import torch
from torch import nn
import torch.nn.functional as F


class Tokenizer(object):
  """A simple tokenizer."""

  def __init__(self, tokenizer_path: str):
    """Initializes the tokenizer."""
    with open(tokenizer_path, 'rb') as f:
      model = f.read()
    self.tokenizer = tensorflow_text.SentencepieceTokenizer(
        model=model, add_eos=False, add_bos=False
    )

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


class PositionalEmbedding(nn.Module):
  """Generates position embedding for a given 1-d sequence.

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

  def __init__(self, embedding_dim: int):
    super().__init__()
    self.embedding_dim = embedding_dim

  def __call__(self, seq_length: int = None, position: torch.tensor = None):
    """Generates a torch.tensor of sinusoids with different frequencies.

    Args:
      seq_length: an optional Python int defining the output sequence length.
        if the `position` argument is specified.
      position:   [B, seq_length], optional position for each token in the
        sequence, only required when the sequence is packed.

    Returns:
      [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
    """
    if position is None:
      assert seq_length is not None
      # [1, seqlen]
      position = torch.arange(seq_length, dtype=torch.float32)[None, :]
    else:
      assert position.ndim == 2, position.shape

    num_timescales = self.embedding_dim // 2
    log_timescale_increment = torch.log(
        torch.tensor(float(self.max_timescale) / float(self.min_timescale))
    ) / torch.maximum(
        torch.tensor(num_timescales, dtype=torch.float32) - 1, torch.tensor(1)
    )
    inv_timescales = self.min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32)
        * -log_timescale_increment
    )
    scaled_time = position[:, :, None] * inv_timescales[None, None, :]
    signal = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=2)
    # Force usage of `np` rather than `jnp` to compute static values at trace
    # time.
    signal = F.pad(signal, (0, self.embedding_dim % 2, 0, 0, 0, 0))
    return signal


class MlpBlockWithMask(nn.Module):
  """Transformer MLP / feed-forward block that supports masking."""

  def __init__(
      self,
      mlp_dim: int,
      d_model: int,
      use_bias: bool = True,
      dtype: torch.dtype = torch.float32,
      activation_fn: nn.Module = nn.GELU,
  ):
    super().__init__()

    self.mlp_dim = mlp_dim
    self.d_model = d_model
    self.use_bias = use_bias
    self.dtype = dtype
    self.activation_fn = activation_fn

    self.c_fc = nn.Linear(
        in_features=self.d_model,
        out_features=self.mlp_dim,
        dtype=self.dtype,
        bias=self.use_bias,
    )
    self.c_proj = nn.Linear(
        in_features=self.mlp_dim,
        out_features=self.d_model,
        dtype=self.dtype,
        bias=self.use_bias,
    )

  def __call__(
      self, inputs: torch.Tensor, mlp_mask: torch.Tensor
  ) -> torch.Tensor:
    """Applies Transformer MlpBlock with mask module."""
    x = self.c_fc(inputs)
    x = self.activation_fn()(x)
    x = x * mlp_mask[..., None]  # First masking.
    x = self.c_proj(x)
    x = x * mlp_mask[..., None]  # Second masking.
    return x


class ResidualAttentionBlock(nn.Module):
  """Transformer residual attention block."""

  def __init__(
      self,
      d_model: int,
      n_head: int,
      mlp_dim: int,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    self.mlp_dim = mlp_dim
    self.dtype = dtype

    self.attn = nn.MultiheadAttention(d_model, n_head, dtype=self.dtype)
    self.ln_1 = nn.LayerNorm(d_model, dtype=self.dtype)
    self.mlp = MlpBlockWithMask(
        self.mlp_dim,
        d_model,
        use_bias=True,
        dtype=self.dtype,
        activation_fn=nn.ReLU,
    )
    self.ln_2 = nn.LayerNorm(d_model, dtype=self.dtype)

  def attention(self, x: torch.Tensor, mask: torch.Tensor):
    attn_mask = (
        mask[:, None, None, :]
        .repeat(1, self.n_head, x.shape[0], 1)
        .flatten(0, 1)
    )
    attn_mask[attn_mask == 0] = float('-inf')
    attn_mask[attn_mask == 1] = 0
    return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

  def forward(self, x: torch.Tensor, mask: torch.Tensor):
    x = x + self.attention(self.ln_1(x), mask.permute(1, 0))
    x = x + self.mlp(self.ln_2(x), mask)
    return x, mask


class SequentialMultiInput(nn.Sequential):
  """Sequential module that can take multiple inputs."""

  def forward(self, *inputs):
    for module in self._modules.values():
      if isinstance(inputs, tuple):
        inputs = module(*inputs)
      else:
        inputs = module(inputs)
    return inputs


class Transformer(nn.Module):
  """Transformer implementation."""

  def __init__(
      self,
      width: int,
      layers: int,
      heads: int,
      mlp_dim: int,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.width = width
    self.layers = layers
    self.heads = heads
    self.mlp_dim = mlp_dim
    self.dtype = dtype

    self.resblocks = SequentialMultiInput(*[
        ResidualAttentionBlock(self.width, self.heads, self.mlp_dim, self.dtype)
        for _ in range(self.layers)
    ])

  def forward(self, x: torch.Tensor, mask: torch.Tensor):
    return self.resblocks(x, mask)[0]


class GlobalAvgPooling(nn.Module):
  """Performs a simple global pooling over the input with optional paddings.

  Attributes:
    pooling_dims: A list of dims to perform pooling over.
    keepdims: If True, keep dimension of inputs after pooling.
  """

  pooling_dims: t.Sequence[int]
  epsilon: float = 1e-8

  def __init__(
      self, pooling_dims: t.Sequence[int], epsilon: float = 1e-8
  ):
    super().__init__()
    self.pooling_dims = pooling_dims
    self.epsilon = epsilon

    if not all([p_dims >= 0 for p_dims in self.pooling_dims]):
      raise ValueError('pooling_dims must be non-negative integers.')

  def __call__(
      self,
      inputs: torch.tensor,
      compatible_paddings: torch.tensor,
  ):
    """Applies global average spatial pooling to inputs.

    Args:
      inputs: An input tensor.
      compatible_paddings: paddings of inputs with shapes compatible with
        inputs, e.g. compatible_paddings with shape [B, 1] for inputs with shape
        [B, D].

    Returns:
      Output tensor with global pooling applied.
    """
    padded_value = torch.zeros_like(inputs)
    padded_value = torch.ones_like(inputs) * padded_value
    inputs = torch.where(compatible_paddings > 0, padded_value, inputs)
    valid_inputs = (
        torch.sum(
            1.0 - compatible_paddings,
            self.pooling_dims,
            keepdims=True,
            dtype=inputs.dtype,
        )
        + self.epsilon
    )
    inputs_sum = torch.sum(inputs, self.pooling_dims, keepdims=True)
    outputs = torch.divide(inputs_sum, valid_inputs).type(inputs.dtype)
    outputs = torch.squeeze(outputs, axis=self.pooling_dims)
    return outputs


class TextEncoder(nn.Module):
  """Text encoder implementation."""

  def __init__(
      self,
      config: t.Dict[str, int],
      vocab_size: int,
      dtype: torch.dtype = torch.float32,
      scale_sqrt_depth: bool = True,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.dtype = dtype
    self.scale_sqrt_depth = scale_sqrt_depth

    # The text tower layers are fixed independent of vision tower size.
    self.transformer_layers = config['num_layers']
    self.embedding_dim = config['hidden_size']
    self.transformer_width = config['hidden_size']
    self.mlp_dim = config['mlp_dim']
    self.transformer_heads = config['num_heads']

    self.token_embedding = nn.Embedding(
        self.vocab_size, self.embedding_dim, dtype=self.dtype
    )
    self.pos_embedder = PositionalEmbedding(embedding_dim=self.embedding_dim)
    self.transformer = Transformer(
        width=self.transformer_width,
        layers=self.transformer_layers,
        heads=self.transformer_heads,
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
    )
    self.pooling = GlobalAvgPooling(pooling_dims=[1])
    self.ln_final = nn.LayerNorm(self.transformer_width, dtype=self.dtype)

  def __call__(
      self,
      ids: torch.tensor,
      paddings: torch.tensor,
  ):
    """Applies TextEncoder module."""
    _, seq_length = ids.shape
    mask = (paddings == 0).type(torch.float32)
    mask = mask.permute(1, 0)  # NL -> LN
    x = self.token_embedding(ids)
    if self.scale_sqrt_depth:
      x = x * (self.embedding_dim**0.5)
    x = x + self.pos_embedder(seq_length=seq_length)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x, mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x)
    x = self.pooling(x, compatible_paddings=paddings[:, :, None])
    return x
