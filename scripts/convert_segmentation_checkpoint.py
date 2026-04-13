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

"""Convert Flax/JAX segmentation DPT checkpoints to PyTorch format.

This script reads the original .zip checkpoint (containing .npy files in
Flax/JAX weight layout) and produces a .npz file with weights transposed
to PyTorch conventions. The output can be loaded directly by
`tips.pytorch.decoders.load_segmentation_weights`.

Usage:
  python scripts/convert_segmentation_checkpoint.py \
      --input_zip checkpoints/tips_v2_l14_segmentation_dpt.zip \
      --output_npz checkpoints/tips_v2_l14_segmentation_dpt_pytorch.npz
"""

import argparse
import io
import zipfile

import numpy as np


def _load_npy_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
  with zf.open(name) as f:
    return np.load(io.BytesIO(f.read()))


def _conv_kernel_flax_to_torch(w: np.ndarray) -> np.ndarray:
  """Convert conv kernel: (H, W, C_in, C_out) -> (C_out, C_in, H, W)."""
  return w.transpose(3, 2, 0, 1).copy()


def _conv_transpose_kernel_flax_to_torch(w: np.ndarray) -> np.ndarray:
  """Convert conv-transpose kernel: (H, W, C_out, C_in) -> (C_in, C_out, H, W)."""
  return w.transpose(2, 3, 0, 1).copy()


def _linear_kernel_flax_to_torch(w: np.ndarray) -> np.ndarray:
  """Convert linear kernel: (in, out) -> (out, in)."""
  return w.T.copy()


def _bias(w: np.ndarray) -> np.ndarray:
  return w.copy()


def convert_checkpoint(input_zip: str, output_npz: str) -> None:
  """Convert a Flax/JAX segmentation DPT checkpoint to PyTorch format."""
  zf = zipfile.ZipFile(input_zip, "r")
  npy = lambda name: _load_npy_from_zip(zf, name)
  sd = {}
  prefix = "decoder/dpt/"

  # --- Reassemble blocks ---
  for i in range(4):
    sd[f"reassemble.out_projections.{i}.weight"] = (
        _conv_kernel_flax_to_torch(
            npy(f"{prefix}reassemble_blocks/out_projection_{i}/kernel.npy")
        )
    )
    sd[f"reassemble.out_projections.{i}.bias"] = _bias(
        npy(f"{prefix}reassemble_blocks/out_projection_{i}/bias.npy")
    )
    sd[f"reassemble.readout_projects.{i}.weight"] = (
        _linear_kernel_flax_to_torch(
            npy(f"{prefix}reassemble_blocks/readout_projects_{i}/kernel.npy")
        )
    )
    sd[f"reassemble.readout_projects.{i}.bias"] = _bias(
        npy(f"{prefix}reassemble_blocks/readout_projects_{i}/bias.npy")
    )

  # Resize layers (0=ConvTranspose, 1=ConvTranspose, 2=Identity, 3=Conv)
  sd["reassemble.resize_layers.0.weight"] = (
      _conv_transpose_kernel_flax_to_torch(
          npy(f"{prefix}reassemble_blocks/resize_layers_0/kernel.npy")
      )
  )
  sd["reassemble.resize_layers.0.bias"] = _bias(
      npy(f"{prefix}reassemble_blocks/resize_layers_0/bias.npy")
  )
  sd["reassemble.resize_layers.1.weight"] = (
      _conv_transpose_kernel_flax_to_torch(
          npy(f"{prefix}reassemble_blocks/resize_layers_1/kernel.npy")
      )
  )
  sd["reassemble.resize_layers.1.bias"] = _bias(
      npy(f"{prefix}reassemble_blocks/resize_layers_1/bias.npy")
  )
  sd["reassemble.resize_layers.3.weight"] = _conv_kernel_flax_to_torch(
      npy(f"{prefix}reassemble_blocks/resize_layers_3/kernel.npy")
  )
  sd["reassemble.resize_layers.3.bias"] = _bias(
      npy(f"{prefix}reassemble_blocks/resize_layers_3/bias.npy")
  )

  # --- Convs (no bias) ---
  for i in range(4):
    sd[f"convs.{i}.weight"] = _conv_kernel_flax_to_torch(
        npy(f"{prefix}convs_{i}/kernel.npy")
    )

  # --- Fusion blocks ---
  for i in range(4):
    fb = f"{prefix}fusion_blocks_{i}/"
    if i == 0:
      # Block 0 has no residual unit; only main_unit (PreActResidualConvUnit_0).
      sd[f"fusion_blocks.{i}.main_unit.conv1.weight"] = (
          _conv_kernel_flax_to_torch(
              npy(f"{fb}PreActResidualConvUnit_0/conv1/kernel.npy")
          )
      )
      sd[f"fusion_blocks.{i}.main_unit.conv2.weight"] = (
          _conv_kernel_flax_to_torch(
              npy(f"{fb}PreActResidualConvUnit_0/conv2/kernel.npy")
          )
      )
    else:
      # Blocks 1-3 have both residual_unit (Unit_0) and main_unit (Unit_1).
      sd[f"fusion_blocks.{i}.residual_unit.conv1.weight"] = (
          _conv_kernel_flax_to_torch(
              npy(f"{fb}PreActResidualConvUnit_0/conv1/kernel.npy")
          )
      )
      sd[f"fusion_blocks.{i}.residual_unit.conv2.weight"] = (
          _conv_kernel_flax_to_torch(
              npy(f"{fb}PreActResidualConvUnit_0/conv2/kernel.npy")
          )
      )
      sd[f"fusion_blocks.{i}.main_unit.conv1.weight"] = (
          _conv_kernel_flax_to_torch(
              npy(f"{fb}PreActResidualConvUnit_1/conv1/kernel.npy")
          )
      )
      sd[f"fusion_blocks.{i}.main_unit.conv2.weight"] = (
          _conv_kernel_flax_to_torch(
              npy(f"{fb}PreActResidualConvUnit_1/conv2/kernel.npy")
          )
      )

    # Output conv (shared by all fusion blocks).
    sd[f"fusion_blocks.{i}.out_conv.weight"] = _conv_kernel_flax_to_torch(
        npy(f"{fb}Conv_0/kernel.npy")
    )
    sd[f"fusion_blocks.{i}.out_conv.bias"] = _bias(
        npy(f"{fb}Conv_0/bias.npy")
    )

  # --- Project conv ---
  sd["project.weight"] = _conv_kernel_flax_to_torch(
      npy(f"{prefix}project/kernel.npy")
  )
  sd["project.bias"] = _bias(npy(f"{prefix}project/bias.npy"))

  # --- Segmentation head (linear) ---
  sd["segmentation_head.weight"] = _linear_kernel_flax_to_torch(
      npy("decoder/pixel_segmentation/kernel.npy")
  )
  sd["segmentation_head.bias"] = _bias(
      npy("decoder/pixel_segmentation/bias.npy")
  )

  zf.close()

  # Save as .npz with keys matching PyTorch state_dict names.
  np.savez(output_npz, **sd)
  print(f"Converted {len(sd)} tensors -> {output_npz}")

  # Print summary.
  for key, val in sorted(sd.items()):
    print(f"  {key}: {val.shape} ({val.dtype})")


def main():
  parser = argparse.ArgumentParser(
      description="Convert Flax DPT segmentation checkpoint to PyTorch format."
  )
  parser.add_argument(
      "--input_zip",
      type=str,
      required=True,
      help="Path to the original Flax .zip checkpoint.",
  )
  parser.add_argument(
      "--output_npz",
      type=str,
      required=True,
      help="Path for the output PyTorch-format .npz checkpoint.",
  )
  args = parser.parse_args()
  convert_checkpoint(args.input_zip, args.output_npz)


if __name__ == "__main__":
  main()
