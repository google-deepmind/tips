# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DPT decoder heads for dense prediction tasks (segmentation, depth, etc.)."""

from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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
      features: list[tuple[torch.Tensor, torch.Tensor]],
  ) -> list[torch.Tensor]:
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


class DPTHead(nn.Module):
  """DPT head producing dense features."""

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
      intermediate_features: list[tuple[torch.Tensor, torch.Tensor]],
  ) -> torch.Tensor:
    x = self.reassemble(intermediate_features)
    x = [self.convs[i](feat) for i, feat in enumerate(x)]

    out = self.fusion_blocks[0](x[-1])
    for i in range(1, 4):
      out = self.fusion_blocks[i](out, residual=x[-(i + 1)])

    out = self.project(out)
    return out


class Decoder(nn.Module):
  """Base decoder class for dense prediction tasks."""

  def __init__(
      self,
      input_embed_dim: int = 1024,
      channels: int = 256,
      post_process_channels: Tuple[int, ...] = (128, 256, 512, 1024),
      readout_type: str = "project",
  ) -> None:
    super().__init__()
    self.dpt = DPTHead(
        input_embed_dim=input_embed_dim,
        channels=channels,
        post_process_channels=post_process_channels,
        readout_type=readout_type,
    )

  def forward(
      self,
      intermediate_features: list[tuple[torch.Tensor, torch.Tensor]],
  ) -> torch.Tensor:
    return self.dpt(intermediate_features)


class SegmentationDecoder(Decoder):
  """Decoder for semantic segmentation."""

  def __init__(
      self,
      num_classes: int = 150,
      **kwargs,
  ) -> None:
    super().__init__(**kwargs)
    channels = kwargs.get("channels", 256)
    self.head = nn.Linear(channels, num_classes)

  def forward(
      self,
      intermediate_features: list[tuple[torch.Tensor, torch.Tensor]],
      image_size: Optional[Tuple[int, int]] = None,
  ) -> torch.Tensor:
    x = super().forward(intermediate_features)
    x = x.permute(0, 2, 3, 1)
    x = self.head(x)
    x = x.permute(0, 3, 1, 2)
    if image_size is not None:
      x = F.interpolate(
          x, size=image_size, mode="bilinear", align_corners=False
      )
    return x


class DepthDecoder(Decoder):
  """Decoder for depth prediction."""

  def __init__(
      self,
      **kwargs,
  ) -> None:
    super().__init__(**kwargs)
    channels = kwargs.get("channels", 256)
    self.head = nn.Linear(channels, 1)

  def forward(
      self,
      intermediate_features: list[tuple[torch.Tensor, torch.Tensor]],
      image_size: Optional[Tuple[int, int]] = None,
  ) -> torch.Tensor:
    x = super().forward(intermediate_features)
    x = x.permute(0, 2, 3, 1)
    x = self.head(x)
    x = x.permute(0, 3, 1, 2)
    if image_size is not None:
      x = F.interpolate(
          x, size=image_size, mode="bilinear", align_corners=False
      )
    return x


class NormalsDecoder(Decoder):
  """Decoder for surface normals."""

  def __init__(
      self,
      **kwargs,
  ) -> None:
    super().__init__(**kwargs)
    channels = kwargs.get("channels", 256)
    self.head = nn.Linear(channels, 3)

  def forward(
      self,
      intermediate_features: list[tuple[torch.Tensor, torch.Tensor]],
      image_size: Optional[Tuple[int, int]] = None,
  ) -> torch.Tensor:
    x = super().forward(intermediate_features)
    x = x.permute(0, 2, 3, 1)
    x = self.head(x)
    x = x.permute(0, 3, 1, 2)
    if image_size is not None:
      x = F.interpolate(
          x, size=image_size, mode="bilinear", align_corners=False
      )
    return x


def load_decoder_weights(
    model: Decoder,
    checkpoint_path: str,
) -> Decoder:
  """Load pre-converted PyTorch weights into a Decoder.

  Args:
    model: A Decoder instance (SegmentationDecoder, DepthDecoder, etc.).
    checkpoint_path: Path to a .npz checkpoint file.

  Returns:
    The model with loaded weights.
  """
  weights = dict(np.load(checkpoint_path, allow_pickle=False))
  sd = {}
  for key, value in weights.items():
    if (
        key.startswith("reassemble.")
        or key.startswith("convs.")
        or key.startswith("fusion_blocks.")
        or key.startswith("project.")
    ):
      # Map to the dpt submodule
      sd[f"dpt.{key}"] = torch.from_numpy(value)
    elif key.startswith("segmentation_head."):
      # Map to the head submodule in the subclass
      sd[f"head.{key.replace('segmentation_head.', '')}"] = torch.from_numpy(
          value
      )

  model.load_state_dict(sd, strict=True)
  print(f"Loaded decoder weights ({len(sd)} tensors)")
  return model
