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
task-specific decoder subclasses for segmentation, depth, and surface normals.

The implementation mirrors the Scenic/Flax reference at:
  scenic/projects/dense_features/models/decoders.py
  research/vision/scene_understanding/imsight/modules/dpt.py

Typical usage:
  decoder = SegmentationDecoder(num_classes=150, input_embed_dim=1024)
  load_decoder_weights(decoder, "path/to/checkpoint.zip")
  logits = decoder(intermediate_features, image_size=(480, 640))
"""

import io
from typing import List, Optional, Tuple
import zipfile

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
    # Upsample 2x with align_corners=True (matches Scenic reference)
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

    # 1x1 conv to project to per-level channels
    self.out_projections = nn.ModuleList(
        [nn.Conv2d(input_embed_dim, ch, 1) for ch in out_channels]
    )

    # Spatial resize layers: 4x up, 2x up, identity, 2x down
    # NOTE: Flax ConvTranspose(padding=3) produces same output SIZE as
    # PyTorch ConvTranspose2d(padding=0). The padding semantics differ:
    # Flax padding adds to the input, PyTorch padding crops the output.
    self.resize_layers = nn.ModuleList([
        nn.ConvTranspose2d(
            out_channels[0], out_channels[0],
            kernel_size=4, stride=4, padding=0,
        ),
        nn.ConvTranspose2d(
            out_channels[1], out_channels[1],
            kernel_size=2, stride=2, padding=0,
        ),
        nn.Identity(),
        nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
    ])

    # Readout projection (concatenate cls_token with patch features)
    if readout_type == "project":
      self.readout_projects = nn.ModuleList([
          nn.Linear(2 * input_embed_dim, input_embed_dim)
          for _ in out_channels
      ])

  def forward(
      self,
      features: List[Tuple[torch.Tensor, torch.Tensor]],
  ) -> List[torch.Tensor]:
    """Process list of (cls_token, spatial_features) tuples.

    Args:
      features: list of (cls_token [B,D], patch_feats [B,D,H,W])

    Returns:
      list of tensors at different scales.
    """
    out = []
    for i, (cls_token, x) in enumerate(features):
      b, d, h, w = x.shape

      if self.readout_type == "project":
        # Flatten spatial -> (B, HW, D)
        x_flat = x.flatten(2).transpose(1, 2)
        # Expand cls_token -> (B, HW, D)
        readout = cls_token.unsqueeze(1).expand(-1, x_flat.shape[1], -1)
        # Concat + project + GELU (tanh approx matches JAX default)
        x_cat = torch.cat([x_flat, readout], dim=-1)
        x_proj = F.gelu(
            self.readout_projects[i](x_cat), approximate="tanh"
        )
        # Reshape back to spatial
        x = x_proj.transpose(1, 2).reshape(b, d, h, w)

      # 1x1 projection
      x = self.out_projections[i](x)
      # Spatial resize
      x = self.resize_layers[i](x)
      out.append(x)
    return out


# ---------------------------------------------------------------------------
# DPT head and decoder classes
# ---------------------------------------------------------------------------


class DPTHead(nn.Module):
  """Shared DPT backbone: ReassembleBlocks + convs + fusion + project.

  Matches the Scenic DPTHead from
  ``research/vision/scene_understanding/imsight/modules/dpt.py``.
  """

  def __init__(
      self,
      input_embed_dim: int = 1024,
      channels: int = 256,
      post_process_channels: Tuple[int, ...] = (128, 256, 512, 1024),
      readout_type: str = "project",
      output_activation: bool = False,
  ) -> None:
    super().__init__()
    self.output_activation = output_activation
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
    x = self.reassemble(intermediate_features)
    x = [self.convs[i](feat) for i, feat in enumerate(x)]
    # Fuse bottom-up: start from deepest (x[-1])
    out = self.fusion_blocks[0](x[-1])
    for i in range(1, 4):
      out = self.fusion_blocks[i](out, residual=x[-(i + 1)])
    out = self.project(out)
    if self.output_activation:
      out = F.relu(out)
    return out


class Decoder(nn.Module):
  """Base decoder: DPTHead + linear head.

  Subclasses set ``out_channels`` and may override ``forward`` to add
  task-specific post-processing.
  """

  def __init__(
      self,
      out_channels: int,
      input_embed_dim: int = 1024,
      channels: int = 256,
      post_process_channels: Tuple[int, ...] = (128, 256, 512, 1024),
      readout_type: str = "project",
      output_activation: bool = False,
  ) -> None:
    super().__init__()
    self.channels = channels
    self.out_channels = out_channels
    self.dpt = DPTHead(
        input_embed_dim=input_embed_dim,
        channels=channels,
        post_process_channels=post_process_channels,
        readout_type=readout_type,
        output_activation=output_activation,
    )
    self.head = nn.Linear(channels, out_channels)

  def forward(
      self,
      intermediate_features: List[Tuple[torch.Tensor, torch.Tensor]],
      image_size: Optional[Tuple[int, int]] = None,
  ) -> torch.Tensor:
    x = self.dpt(intermediate_features)
    # (B, C, H, W) -> (B, H, W, C)
    x = x.permute(0, 2, 3, 1)
    x = self.head(x)
    # (B, H, W, C') -> (B, C', H, W)
    x = x.permute(0, 3, 1, 2)
    if image_size is not None:
      x = F.interpolate(
          x, size=image_size, mode="bilinear", align_corners=False
      )
    return x


class SegmentationDecoder(Decoder):
  """Decoder for semantic segmentation.

  Outputs logits of shape ``(B, num_classes, H, W)``.
  """

  def __init__(self, num_classes: int = 150, **kwargs) -> None:
    super().__init__(out_channels=num_classes, **kwargs)


class DepthDecoder(Decoder):
  """Decoder for monocular depth prediction using classification bins.

  Predicts depth by classifying into uniformly spaced bins and computing
  the expectation, following the Scenic reference implementation.

  Outputs depth map of shape ``(B, 1, H, W)``.
  """

  def __init__(
      self,
      num_depth_bins: int = 256,
      min_depth: float = 0.001,
      max_depth: float = 10.0,
      **kwargs,
  ) -> None:
    super().__init__(out_channels=num_depth_bins, **kwargs)
    self.min_depth = min_depth
    self.max_depth = max_depth
    self.num_depth_bins = num_depth_bins
    self.register_buffer(
        "bin_centers",
        torch.linspace(min_depth, max_depth, num_depth_bins),
    )

  def forward(
      self,
      intermediate_features: List[Tuple[torch.Tensor, torch.Tensor]],
      image_size: Optional[Tuple[int, int]] = None,
  ) -> torch.Tensor:
    logits = super().forward(intermediate_features)
    logits = torch.relu(logits) + self.min_depth
    # Linear normalization (could also use softmax)
    probs = logits / torch.sum(logits, dim=1, keepdim=True)
    # Compute expectation to get predicted depth values
    depth_map = torch.einsum("bchw,c->bhw", probs, self.bin_centers)
    if image_size is not None:
      depth_map = F.interpolate(
          depth_map.unsqueeze(1),
          size=image_size,
          mode="bilinear",
          align_corners=False,
      ).squeeze(1)
    return depth_map.unsqueeze(1)


class NormalsDecoder(Decoder):
  """Decoder for surface normal estimation.

  Outputs normal map of shape ``(B, 3, H, W)``.
  Note: outputs are NOT L2-normalized by default, matching the Scenic
  reference. Apply ``F.normalize(out, p=2, dim=1)`` if needed.
  """

  def __init__(self, **kwargs) -> None:
    super().__init__(out_channels=3, **kwargs)


# ---------------------------------------------------------------------------
# Weight loading from Scenic/Flax checkpoints
# ---------------------------------------------------------------------------


def _load_npy_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
  """Load a single .npy array from a zipfile."""
  with zf.open(name) as f:
    return np.load(io.BytesIO(f.read()))


def _conv_flax_to_torch(w: np.ndarray) -> torch.Tensor:
  """Flax Conv kernel (H, W, Cin, Cout) -> PyTorch (Cout, Cin, H, W)."""
  return torch.from_numpy(np.asarray(w).transpose(3, 2, 0, 1).copy())


def _conv_transpose_flax_to_torch(w: np.ndarray) -> torch.Tensor:
  """Flax ConvTranspose kernel -> PyTorch ConvTranspose2d.

  Maps (H, W, Cin, Cout) -> (Cin, Cout, H, W) with 180-degree spatial
  flip. Flax ConvTranspose uses transpose_kernel=False (no kernel flip),
  while PyTorch ConvTranspose2d always flips, so we must pre-flip.
  """
  w_np = np.asarray(w)
  w_flipped = w_np[::-1, ::-1, :, :].copy()
  return torch.from_numpy(w_flipped.transpose(2, 3, 0, 1).copy())


def _dense_flax_to_torch(w: np.ndarray) -> torch.Tensor:
  """Flax Dense kernel (in, out) -> PyTorch Linear (out, in)."""
  return torch.from_numpy(np.asarray(w).T.copy())


def _bias(w: np.ndarray) -> torch.Tensor:
  """Copy a bias parameter as a PyTorch tensor."""
  return torch.from_numpy(np.asarray(w).copy())


def _load_dpt_state_dict(
    zf: zipfile.ZipFile,
    prefix: str = "decoder/dpt/",
) -> dict:
  """Load DPT backbone weights from a Scenic checkpoint zip.

  Args:
    zf: Open zipfile containing .npy arrays.
    prefix: Path prefix inside the zip (default: "decoder/dpt/").

  Returns:
    Dict mapping ``dpt.*`` state_dict keys to tensors.
  """
  npy = lambda name: _load_npy_from_zip(zf, name)
  sd = {}

  # --- ReassembleBlocks ---
  for i in range(4):
    # out_projections (Conv2d 1x1)
    sd[f"dpt.reassemble.out_projections.{i}.weight"] = _conv_flax_to_torch(
        npy(f"{prefix}reassemble_blocks/out_projection_{i}/kernel.npy")
    )
    sd[f"dpt.reassemble.out_projections.{i}.bias"] = _bias(
        npy(f"{prefix}reassemble_blocks/out_projection_{i}/bias.npy")
    )
    # readout_projects (Linear)
    sd[f"dpt.reassemble.readout_projects.{i}.weight"] = _dense_flax_to_torch(
        npy(f"{prefix}reassemble_blocks/readout_projects_{i}/kernel.npy")
    )
    sd[f"dpt.reassemble.readout_projects.{i}.bias"] = _bias(
        npy(f"{prefix}reassemble_blocks/readout_projects_{i}/bias.npy")
    )

  # resize_layers: 0=ConvTranspose, 1=ConvTranspose, 2=Identity, 3=Conv
  sd["dpt.reassemble.resize_layers.0.weight"] = _conv_transpose_flax_to_torch(
      npy(f"{prefix}reassemble_blocks/resize_layers_0/kernel.npy")
  )
  sd["dpt.reassemble.resize_layers.0.bias"] = _bias(
      npy(f"{prefix}reassemble_blocks/resize_layers_0/bias.npy")
  )
  sd["dpt.reassemble.resize_layers.1.weight"] = _conv_transpose_flax_to_torch(
      npy(f"{prefix}reassemble_blocks/resize_layers_1/kernel.npy")
  )
  sd["dpt.reassemble.resize_layers.1.bias"] = _bias(
      npy(f"{prefix}reassemble_blocks/resize_layers_1/bias.npy")
  )
  # resize_layers_2 = Identity (no weights)
  sd["dpt.reassemble.resize_layers.3.weight"] = _conv_flax_to_torch(
      npy(f"{prefix}reassemble_blocks/resize_layers_3/kernel.npy")
  )
  sd["dpt.reassemble.resize_layers.3.bias"] = _bias(
      npy(f"{prefix}reassemble_blocks/resize_layers_3/bias.npy")
  )

  # --- Convs (3x3, no bias) ---
  for i in range(4):
    sd[f"dpt.convs.{i}.weight"] = _conv_flax_to_torch(
        npy(f"{prefix}convs_{i}/kernel.npy")
    )

  # --- Fusion blocks ---
  for i in range(4):
    fb = f"{prefix}fusion_blocks_{i}/"
    if i == 0:
      # No residual unit, only 1 PreActResidualConvUnit
      sd[f"dpt.fusion_blocks.{i}.main_unit.conv1.weight"] = (
          _conv_flax_to_torch(npy(f"{fb}PreActResidualConvUnit_0/conv1/kernel.npy"))
      )
      sd[f"dpt.fusion_blocks.{i}.main_unit.conv2.weight"] = (
          _conv_flax_to_torch(npy(f"{fb}PreActResidualConvUnit_0/conv2/kernel.npy"))
      )
    else:
      # Residual unit (index 0) + main unit (index 1)
      sd[f"dpt.fusion_blocks.{i}.residual_unit.conv1.weight"] = (
          _conv_flax_to_torch(npy(f"{fb}PreActResidualConvUnit_0/conv1/kernel.npy"))
      )
      sd[f"dpt.fusion_blocks.{i}.residual_unit.conv2.weight"] = (
          _conv_flax_to_torch(npy(f"{fb}PreActResidualConvUnit_0/conv2/kernel.npy"))
      )
      sd[f"dpt.fusion_blocks.{i}.main_unit.conv1.weight"] = (
          _conv_flax_to_torch(npy(f"{fb}PreActResidualConvUnit_1/conv1/kernel.npy"))
      )
      sd[f"dpt.fusion_blocks.{i}.main_unit.conv2.weight"] = (
          _conv_flax_to_torch(npy(f"{fb}PreActResidualConvUnit_1/conv2/kernel.npy"))
      )

    # out_conv (Conv2d 1x1)
    sd[f"dpt.fusion_blocks.{i}.out_conv.weight"] = _conv_flax_to_torch(
        npy(f"{fb}Conv_0/kernel.npy")
    )
    sd[f"dpt.fusion_blocks.{i}.out_conv.bias"] = _bias(
        npy(f"{fb}Conv_0/bias.npy")
    )

  # --- Project ---
  sd["dpt.project.weight"] = _conv_flax_to_torch(
      npy(f"{prefix}project/kernel.npy")
  )
  sd["dpt.project.bias"] = _bias(
      npy(f"{prefix}project/bias.npy")
  )

  return sd


# Mapping from Scenic head names to the PyTorch ``head.*`` keys.
_HEAD_NAME_MAP = {
    "segmentation": "pixel_segmentation",
    "depth": "pixel_depth_classif",
    "normals": "pixel_normals",
}


def load_decoder_weights(
    decoder: Decoder,
    checkpoint_path: str,
    head_type: Optional[str] = None,
) -> Decoder:
  """Load Scenic/Flax DPT weights from a zip checkpoint.

  The checkpoint should be a zip file containing .npy arrays produced by
  ``flax.training.save_checkpoint``.

  Args:
    decoder: A ``Decoder`` subclass instance.
    checkpoint_path: Path to the ``.zip`` checkpoint file.
    head_type: One of ``"segmentation"``, ``"depth"``, or ``"normals"``.
      If ``None``, auto-detects from the decoder class.

  Returns:
    The decoder with loaded weights.
  """
  # Auto-detect head type from decoder class
  if head_type is None:
    if isinstance(decoder, SegmentationDecoder):
      head_type = "segmentation"
    elif isinstance(decoder, DepthDecoder):
      head_type = "depth"
    elif isinstance(decoder, NormalsDecoder):
      head_type = "normals"
    else:
      raise ValueError(
          f"Cannot auto-detect head_type for {type(decoder).__name__}. "
          "Pass head_type explicitly."
      )

  scenic_head_name = _HEAD_NAME_MAP[head_type]

  zf = zipfile.ZipFile(checkpoint_path, "r")

  # Load DPT backbone weights
  sd = _load_dpt_state_dict(zf)

  # Load head weights
  sd["head.weight"] = _dense_flax_to_torch(
      _load_npy_from_zip(zf, f"decoder/{scenic_head_name}/kernel.npy")
  )
  sd["head.bias"] = _bias(
      _load_npy_from_zip(zf, f"decoder/{scenic_head_name}/bias.npy")
  )

  zf.close()

  # Add buffers (e.g. bin_centers for DepthDecoder)
  for name, buf in decoder.named_buffers():
    if name not in sd:
      sd[name] = buf

  missing, unexpected = decoder.load_state_dict(sd, strict=True)
  if missing:
    print(f"WARNING: Missing keys: {missing}")
  if unexpected:
    print(f"WARNING: Unexpected keys: {unexpected}")
  print(
      f"Loaded {head_type} decoder weights "
      f"({len(sd)} tensors from {checkpoint_path})"
  )
  return decoder
