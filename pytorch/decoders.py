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

"""DPT decoder heads for dense prediction tasks.

This module provides a shared DPT backbone (ReassembleBlocks + fusion) and
task-specific decoder subclasses for segmentation, depth, and surface normals,
mirroring the Scenic/Flax decoder hierarchy.

Typical usage:
  decoder = SegmentationDecoder(num_classes=150, input_embed_dim=1024)
  load_decoder_weights(decoder, "path/to/checkpoint.npz")
  logits = decoder(intermediate_features, image_size=(480, 640))
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PreActResidualConvUnit(nn.Module):
  """Pre-activation residual convolution unit."""

  def __init__(self, features: int) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=False)
    self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = F.relu(x)
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    return x + residual


class FeatureFusionBlock(nn.Module):
  """Fuses features with optional residual input, then upsamples 2x."""

  def __init__(
      self,
      features: int,
      has_residual: bool = False,
      expand: bool = False,
  ) -> None:
    super().__init__()
    self.has_residual = has_residual
    if has_residual:
      self.residual_unit = PreActResidualConvUnit(features)
    self.main_unit = PreActResidualConvUnit(features)
    out_features = features // 2 if expand else features
    self.out_conv = nn.Conv2d(features, out_features, 1, bias=True)

  def forward(
      self,
      x: torch.Tensor,
      residual: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    if self.has_residual and residual is not None:
      if residual.shape != x.shape:
        residual = F.interpolate(
            residual,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
      residual = self.residual_unit(residual)
      x = x + residual
    x = self.main_unit(x)
    # Upsample 2x with align_corners=True (matches Scenic reference).
    x = F.interpolate(
        x, scale_factor=2, mode="bilinear", align_corners=True
    )
    x = self.out_conv(x)
    return x


class ReassembleBlocks(nn.Module):
  """Projects and resizes intermediate ViT features to different scales."""

  def __init__(
      self,
      input_embed_dim: int = 1024,
      out_channels: Tuple[int, ...] = (128, 256, 512, 1024),
      readout_type: str = "project",
  ) -> None:
    super().__init__()
    self.readout_type = readout_type
    self.out_projections = nn.ModuleList(
        [nn.Conv2d(input_embed_dim, ch, 1) for ch in out_channels]
    )
    self.resize_layers = nn.ModuleList([
        nn.ConvTranspose2d(
            out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0
        ),
        nn.ConvTranspose2d(
            out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0
        ),
        nn.Identity(),
        nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
    ])
    if readout_type == "project":
      self.readout_projects = nn.ModuleList([
          nn.Linear(2 * input_embed_dim, input_embed_dim)
          for _ in out_channels
      ])

  def forward(
      self,
      features: List[Tuple[torch.Tensor, torch.Tensor]],
  ) -> List[torch.Tensor]:
    out = []
    for i, (cls_token, x) in enumerate(features):
      b, d, h, w = x.shape
      if self.readout_type == "project":
        x_flat = x.flatten(2).transpose(1, 2)
        readout = cls_token.unsqueeze(1).expand(-1, x_flat.shape[1], -1)
        x_cat = torch.cat([x_flat, readout], dim=-1)
        x_proj = F.gelu(self.readout_projects[i](x_cat))
        x = x_proj.transpose(1, 2).reshape(b, d, h, w)
      x = self.out_projections[i](x)
      x = self.resize_layers[i](x)
      out.append(x)
    return out


# ---------------------------------------------------------------------------
# Shared DPT head
# ---------------------------------------------------------------------------


class DPTHead(nn.Module):
  """Shared DPT head that produces dense feature maps from ViT intermediates.

  This module performs the common operations for all decoder types:
  1. Reassembles intermediate ViT features at multiple scales.
  2. Projects them to a common channel dimension.
  3. Fuses them bottom-up through residual fusion blocks.
  """

  def __init__(
      self,
      input_embed_dim: int = 1024,
      channels: int = 256,
      post_process_channels: Tuple[int, ...] = (128, 256, 512, 1024),
      readout_type: str = "project",
  ) -> None:
    super().__init__()
    self.reassemble = ReassembleBlocks(
        input_embed_dim=input_embed_dim,
        out_channels=post_process_channels,
        readout_type=readout_type,
    )
    self.convs = nn.ModuleList([
        nn.Conv2d(ch, channels, 3, padding=1, bias=False)
        for ch in post_process_channels
    ])
    self.fusion_blocks = nn.ModuleList([
        FeatureFusionBlock(channels, has_residual=False),
        FeatureFusionBlock(channels, has_residual=True),
        FeatureFusionBlock(channels, has_residual=True),
        FeatureFusionBlock(channels, has_residual=True),
    ])
    self.project = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

  def forward(
      self,
      intermediate_features: List[Tuple[torch.Tensor, torch.Tensor]],
  ) -> torch.Tensor:
    """Produces dense features of shape (B, channels, H', W')."""
    x = self.reassemble(intermediate_features)
    x = [self.convs[i](feat) for i, feat in enumerate(x)]

    out = self.fusion_blocks[0](x[-1])
    for i in range(1, 4):
      out = self.fusion_blocks[i](out, residual=x[-(i + 1)])

    out = self.project(out)
    return out


# ---------------------------------------------------------------------------
# Base decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
  """Base decoder class for dense prediction tasks.

  Wraps a DPTHead and provides a common interface. Subclasses add a
  task-specific output head (e.g. segmentation logits, depth regression).
  """

  def __init__(
      self,
      input_embed_dim: int = 1024,
      channels: int = 256,
      post_process_channels: Tuple[int, ...] = (128, 256, 512, 1024),
      readout_type: str = "project",
  ) -> None:
    super().__init__()
    self.channels = channels
    self.dpt = DPTHead(
        input_embed_dim=input_embed_dim,
        channels=channels,
        post_process_channels=post_process_channels,
        readout_type=readout_type,
    )

  def forward(
      self,
      intermediate_features: List[Tuple[torch.Tensor, torch.Tensor]],
      image_size: Optional[Tuple[int, int]] = None,
  ) -> torch.Tensor:
    """Returns dense features from the DPT head (B, C, H', W')."""
    return self.dpt(intermediate_features)


# ---------------------------------------------------------------------------
# Task-specific decoders
# ---------------------------------------------------------------------------


class SegmentationDecoder(Decoder):
  """Decoder for semantic segmentation.

  Produces per-pixel class logits of shape (B, num_classes, H, W).
  """

  def __init__(
      self,
      num_classes: int = 150,
      **kwargs,
  ) -> None:
    super().__init__(**kwargs)
    self.num_classes = num_classes
    self.head = nn.Linear(self.channels, num_classes)

  def forward(
      self,
      intermediate_features: List[Tuple[torch.Tensor, torch.Tensor]],
      image_size: Optional[Tuple[int, int]] = None,
  ) -> torch.Tensor:
    x = self.dpt(intermediate_features)  # (B, C, H', W')
    x = x.permute(0, 2, 3, 1)  # (B, H', W', C)
    x = self.head(x)  # (B, H', W', num_classes)
    x = x.permute(0, 3, 1, 2)  # (B, num_classes, H', W')
    if image_size is not None:
      x = F.interpolate(
          x, size=image_size, mode="bilinear", align_corners=False
      )
    return x


class DepthDecoder(Decoder):
  """Decoder for monocular depth prediction.

  Produces per-pixel depth values of shape (B, 1, H, W).
  """

  def __init__(
      self,
      **kwargs,
  ) -> None:
    super().__init__(**kwargs)
    self.head = nn.Linear(self.channels, 1)

  def forward(
      self,
      intermediate_features: List[Tuple[torch.Tensor, torch.Tensor]],
      image_size: Optional[Tuple[int, int]] = None,
  ) -> torch.Tensor:
    x = self.dpt(intermediate_features)  # (B, C, H', W')
    x = x.permute(0, 2, 3, 1)  # (B, H', W', C)
    x = self.head(x)  # (B, H', W', 1)
    x = x.permute(0, 3, 1, 2)  # (B, 1, H', W')
    if image_size is not None:
      x = F.interpolate(
          x, size=image_size, mode="bilinear", align_corners=False
      )
    return x


class NormalsDecoder(Decoder):
  """Decoder for surface normals prediction.

  Produces per-pixel 3D normal vectors of shape (B, 3, H, W).
  """

  def __init__(
      self,
      **kwargs,
  ) -> None:
    super().__init__(**kwargs)
    self.head = nn.Linear(self.channels, 3)

  def forward(
      self,
      intermediate_features: List[Tuple[torch.Tensor, torch.Tensor]],
      image_size: Optional[Tuple[int, int]] = None,
  ) -> torch.Tensor:
    x = self.dpt(intermediate_features)  # (B, C, H', W')
    x = x.permute(0, 2, 3, 1)  # (B, H', W', C)
    x = self.head(x)  # (B, H', W', 3)
    x = x.permute(0, 3, 1, 2)  # (B, 3, H', W')
    if image_size is not None:
      x = F.interpolate(
          x, size=image_size, mode="bilinear", align_corners=False
      )
    return x


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------

# Mapping from legacy flat checkpoint keys to the new hierarchical names.
_LEGACY_KEY_PREFIXES = {
    "reassemble.": "dpt.reassemble.",
    "convs.": "dpt.convs.",
    "fusion_blocks.": "dpt.fusion_blocks.",
    "project.": "dpt.project.",
    "segmentation_head.": "head.",
}


def load_decoder_weights(
    model: Decoder,
    checkpoint_path: str,
) -> Decoder:
  """Load pre-converted PyTorch weights into a Decoder.

  The checkpoint should be an .npz file where keys match the model's
  state_dict parameter names and values are NumPy arrays in PyTorch layout
  (already transposed from Flax/JAX format).

  Use ``scripts/convert_segmentation_checkpoint.py`` to produce these
  checkpoints from the original Flax/JAX .zip files.

  Supports both the legacy flat key format (e.g. ``reassemble.…``) and the
  new hierarchical format (e.g. ``dpt.reassemble.…``).

  Args:
    model: A Decoder instance (SegmentationDecoder, DepthDecoder, etc.).
    checkpoint_path: Path to a .npz checkpoint file.

  Returns:
    The model with loaded weights.
  """
  weights = dict(np.load(checkpoint_path, allow_pickle=False))

  sd = {}
  for key, value in weights.items():
    new_key = key
    # Remap legacy flat keys to hierarchical names.
    for old_prefix, new_prefix in _LEGACY_KEY_PREFIXES.items():
      if key.startswith(old_prefix):
        new_key = new_prefix + key[len(old_prefix):]
        break
    sd[new_key] = torch.from_numpy(value)

  model.load_state_dict(sd, strict=True)
  print(f"Loaded decoder weights from {checkpoint_path} ({len(sd)} tensors)")
  return model
