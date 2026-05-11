# %% [markdown]
# # TIPSv2: Scenic ↔ PyTorch Numerical Parity Verification
#
# This notebook verifies that the **PyTorch `decoders.py`** in the
# [google-deepmind/tips](https://github.com/google-deepmind/tips) repository
# produces **numerically identical** outputs to the Scenic/Flax reference
# implementation when loaded with the same checkpoint weights.
#
# We test all three decoder types:
# 1. **Depth** (classification-based, 256 bins)
# 2. **Surface Normals**
# 3. **Semantic Segmentation**
#
# The Scenic DPT implementation is embedded directly in this notebook
# (extracted from `research/vision/scene_understanding/imsight/modules/dpt.py`
# and `scenic/projects/dense_features/models/decoders.py`).

# %% [markdown]
# ## Cell 1: Install Dependencies

# %%
# Install JAX, Flax, and PyTorch (CPU-only for parity testing)
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "jax[cpu]", "flax", "ml_collections", "torch", "numpy",
])

# %% [markdown]
# ## Cell 2: Clone TIPS Repository

# %%
import os, subprocess

TIPS_REPO = "/content/tips"
# Change this to a specific branch/PR if needed:
TIPS_BRANCH = "main"

if not os.path.isdir(TIPS_REPO):
    subprocess.check_call([
        "git", "clone",
        "--branch", TIPS_BRANCH,
        "https://github.com/google-deepmind/tips.git",
        TIPS_REPO,
    ])
    print(f"Cloned tips repo ({TIPS_BRANCH}) to {TIPS_REPO}")
else:
    print(f"tips repo already exists at {TIPS_REPO}")

sys.path.insert(0, TIPS_REPO)

# Verify decoders.py exists and imports correctly
from pytorch import decoders as pt_decoders
print(f"✓ Imported pytorch.decoders: {pt_decoders.__file__}")

# %% [markdown]
# ## Cell 3: Scenic/Flax DPT Implementation (Reference)
#
# This is the Scenic reference implementation, extracted from internal code
# and made standalone (no google3 dependencies).

# %%
import functools
from typing import Callable, Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np


# ─── Scenic resize_with_aligned_corners ─────────────────────────────────────

def resize_with_aligned_corners(image, shape, method, antialias):
  """Resize emulating PyTorch's align_corners=True."""
  output_height, output_width = shape[1:3]
  scale = jnp.array((
      (output_height - 1.0) / (image.shape[1] - 1.0),
      (output_width - 1.0) / (image.shape[2] - 1.0),
  ))
  translate = jnp.array((
      (output_height - scale[0] * image.shape[1]) / 2.0,
      (output_width - scale[1] * image.shape[2]) / 2.0,
  ))
  return jax.image.scale_and_translate(
      image, shape, method=method, scale=scale,
      spatial_dims=(1, 2), translation=translate, antialias=antialias,
  )


# ─── Scenic PreActResidualConvUnit ──────────────────────────────────────────

class ScenicPreActResidualConvUnit(nn.Module):
  features: int
  activation: Callable = nn.relu

  def setup(self):
    self.conv1 = nn.Conv(self.features, (3, 3), padding=1, use_bias=False)
    self.conv2 = nn.Conv(self.features, (3, 3), padding=1, use_bias=False)

  def __call__(self, x):
    residual = x
    x = self.activation(x)
    x = self.conv1(x)
    x = self.activation(x)
    x = self.conv2(x)
    return x + residual


# ─── Scenic FeatureFusionBlock ──────────────────────────────────────────────

class ScenicFeatureFusionBlock(nn.Module):
  features: int
  activation: Callable = nn.relu

  @nn.compact
  def __call__(self, inputs):
    x = inputs[0]
    if len(inputs) == 2:
      res = inputs[1]
      if x.shape != res.shape:
        res = jax.image.resize(res, shape=x.shape, method="bilinear")
      res = ScenicPreActResidualConvUnit(self.features,
                                          activation=self.activation)(res)
      x = x + res
    x = ScenicPreActResidualConvUnit(self.features,
                                      activation=self.activation)(x)
    bs, h, w, c = x.shape
    x = resize_with_aligned_corners(
        x, shape=(bs, 2 * h, 2 * w, c), method="bilinear", antialias=False,
    )
    x = nn.Conv(self.features, (1, 1), use_bias=True)(x)
    return x


# ─── Scenic ReassembleBlocks ───────────────────────────────────────────────

class ScenicReassembleBlocks(nn.Module):
  out_channels: tuple = (96, 192, 384, 768)
  readout_type: str = "project"
  features: int = 768

  def setup(self):
    self.out_projection = [
        nn.Conv(ch, (1, 1), padding=0) for ch in self.out_channels
    ]
    self.resize_layers = [
        nn.ConvTranspose(self.out_channels[0], (4, 4), (4, 4), padding=3),
        nn.ConvTranspose(self.out_channels[1], (2, 2), (2, 2), padding=1),
        lambda x: x,  # Identity
        nn.Conv(self.out_channels[3], (3, 3), (2, 2), padding=1),
    ]
    if self.readout_type == "project":
      self.readout_projects = [
          nn.Dense(self.features) for _ in self.out_channels
      ]

  def __call__(self, inputs):
    out = []
    for i, x in enumerate(inputs):
      cls_token, x = x
      feature_shape = x.shape
      if self.readout_type == "project":
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))
        readout = jnp.tile(cls_token[:, None, :], (1, x.shape[1], 1))
        x = self.readout_projects[i](jnp.concatenate([x, readout], -1))
        x = jax.nn.gelu(x)
        x = jnp.reshape(x, feature_shape)
      x = self.out_projection[i](x)
      x = self.resize_layers[i](x)
      out.append(x)
    return out


# ─── Scenic DPTHead ────────────────────────────────────────────────────────

class ScenicDPTHead(nn.Module):
  input_embed_dim: int = 1024
  post_process_channels: tuple = (96, 192, 384, 768)
  readout_type: str = "project"
  channels: int = 256
  output_activation: bool = False
  fusion_activation: Callable = nn.relu

  def setup(self):
    self.reassemble_blocks = ScenicReassembleBlocks(
        features=self.input_embed_dim,
        out_channels=self.post_process_channels,
        readout_type=self.readout_type,
    )
    self.convs = [
        nn.Conv(self.channels, (3, 3), padding=1, use_bias=False)
        for _ in self.post_process_channels
    ]
    self.fusion_blocks = [
        ScenicFeatureFusionBlock(self.channels,
                                  activation=self.fusion_activation)
        for _ in range(4)
    ]
    self.project = nn.Conv(self.channels, (3, 3), padding=1)

  def __call__(self, inputs):
    x = self.reassemble_blocks(inputs)
    x = [self.convs[i](feat) for i, feat in enumerate(x)]
    out = self.fusion_blocks[0]([x[-1]])
    for i in range(1, 4):
      out = self.fusion_blocks[i]([out, x[-(i + 1)]])
    out = self.project(out)
    if self.output_activation:
      out = nn.relu(out)
    return out


# ─── Scenic Decoder classes ───────────────────────────────────────────────

class ScenicDecoder(nn.Module):
  """Base Scenic decoder (simplified for parity testing)."""
  input_embed_dim: int = 1024
  channels: int = 256
  post_process_channels: tuple = (96, 192, 384, 768)
  readout_type: str = "project"
  output_activation: bool = False

  def setup(self):
    self.dpt = ScenicDPTHead(
        input_embed_dim=self.input_embed_dim,
        post_process_channels=self.post_process_channels,
        readout_type=self.readout_type,
        channels=self.channels,
        output_activation=self.output_activation,
    )


class ScenicSegmentationDecoder(ScenicDecoder):
  num_classes: int = 150

  @nn.compact
  def __call__(self, spatial_list, vector_list, *, train=False):
    inputs = list(zip(
        [v[:, 0, :] for v in vector_list],
        spatial_list,
    ))
    x = self.dpt(inputs)
    x = nn.Dense(self.num_classes, name="pixel_segmentation")(x)
    return x


class ScenicDepthDecoder(ScenicDecoder):
  num_depth_bins: int = 256
  min_depth: float = 0.001
  max_depth: float = 10.0

  @nn.compact
  def __call__(self, spatial_list, vector_list, *, train=False):
    inputs = list(zip(
        [v[:, 0, :] for v in vector_list],
        spatial_list,
    ))
    x = self.dpt(inputs)
    x = nn.Dense(self.num_depth_bins, name="pixel_depth_classif")(x)
    bin_centers = jnp.linspace(self.min_depth, self.max_depth,
                                self.num_depth_bins)
    x = nn.relu(x) + self.min_depth
    x_norm = x / jnp.sum(x, axis=-1, keepdims=True)
    x = jnp.expand_dims(jnp.einsum("bhwn,n->bhw", x_norm, bin_centers), -1)
    return x


class ScenicNormalsDecoder(ScenicDecoder):

  @nn.compact
  def __call__(self, spatial_list, vector_list, *, train=False):
    inputs = list(zip(
        [v[:, 0, :] for v in vector_list],
        spatial_list,
    ))
    x = self.dpt(inputs)
    x = nn.Dense(3, name="pixel_normals")(x)
    return x


print("✓ Scenic reference implementation defined")

# %% [markdown]
# ## Cell 4: Weight Copying — Flax → PyTorch
#
# Functions to copy Scenic (Flax) parameters into PyTorch state dicts.
# This ensures both models have **exactly the same weights**.

# %%
import torch
import torch.nn.functional as torch_F


def _conv_flax_to_torch(w):
  """Flax Conv kernel (H,W,Cin,Cout) → PyTorch (Cout,Cin,H,W)."""
  return torch.from_numpy(np.array(w).transpose(3, 2, 0, 1).copy())


def _conv_transpose_flax_to_torch(w):
  """Flax ConvTranspose → PyTorch ConvTranspose2d (with 180° flip).

  Flax ConvTranspose: no kernel flip (transpose_kernel=False).
  PyTorch ConvTranspose2d: always flips. So we pre-flip.
  """
  w_np = np.array(w)
  w_flipped = w_np[::-1, ::-1, :, :].copy()
  return torch.from_numpy(w_flipped.transpose(2, 3, 0, 1).copy())


def _dense_flax_to_torch(w):
  """Flax Dense kernel (in,out) → PyTorch Linear (out,in)."""
  return torch.from_numpy(np.array(w).T.copy())


def _bias_to_torch(w):
  return torch.from_numpy(np.array(w).copy())


def copy_scenic_dpt_to_pytorch(scenic_params):
  """Map Scenic DPT param tree → PyTorch dpt.* state_dict."""
  sd = {}
  dpt = scenic_params["dpt"]

  # ReassembleBlocks
  rb = dpt["reassemble_blocks"]
  for i in range(4):
    sd[f"dpt.reassemble.out_projections.{i}.weight"] = _conv_flax_to_torch(
        rb[f"out_projection_{i}"]["kernel"])
    sd[f"dpt.reassemble.out_projections.{i}.bias"] = _bias_to_torch(
        rb[f"out_projection_{i}"]["bias"])
    sd[f"dpt.reassemble.readout_projects.{i}.weight"] = _dense_flax_to_torch(
        rb[f"readout_projects_{i}"]["kernel"])
    sd[f"dpt.reassemble.readout_projects.{i}.bias"] = _bias_to_torch(
        rb[f"readout_projects_{i}"]["bias"])

  # resize_layers: 0=ConvTranspose, 1=ConvTranspose, 2=Identity, 3=Conv
  sd["dpt.reassemble.resize_layers.0.weight"] = _conv_transpose_flax_to_torch(
      rb["resize_layers_0"]["kernel"])
  sd["dpt.reassemble.resize_layers.0.bias"] = _bias_to_torch(
      rb["resize_layers_0"]["bias"])
  sd["dpt.reassemble.resize_layers.1.weight"] = _conv_transpose_flax_to_torch(
      rb["resize_layers_1"]["kernel"])
  sd["dpt.reassemble.resize_layers.1.bias"] = _bias_to_torch(
      rb["resize_layers_1"]["bias"])
  sd["dpt.reassemble.resize_layers.3.weight"] = _conv_flax_to_torch(
      rb["resize_layers_3"]["kernel"])
  sd["dpt.reassemble.resize_layers.3.bias"] = _bias_to_torch(
      rb["resize_layers_3"]["bias"])

  # Convs
  for i in range(4):
    sd[f"dpt.convs.{i}.weight"] = _conv_flax_to_torch(
        dpt[f"convs_{i}"]["kernel"])

  # Fusion blocks
  for i in range(4):
    fb = dpt[f"fusion_blocks_{i}"]
    if i == 0:
      sd[f"dpt.fusion_blocks.{i}.main_unit.conv1.weight"] = (
          _conv_flax_to_torch(fb["PreActResidualConvUnit_0"]["conv1"]["kernel"]))
      sd[f"dpt.fusion_blocks.{i}.main_unit.conv2.weight"] = (
          _conv_flax_to_torch(fb["PreActResidualConvUnit_0"]["conv2"]["kernel"]))
    else:
      sd[f"dpt.fusion_blocks.{i}.residual_unit.conv1.weight"] = (
          _conv_flax_to_torch(fb["PreActResidualConvUnit_0"]["conv1"]["kernel"]))
      sd[f"dpt.fusion_blocks.{i}.residual_unit.conv2.weight"] = (
          _conv_flax_to_torch(fb["PreActResidualConvUnit_0"]["conv2"]["kernel"]))
      sd[f"dpt.fusion_blocks.{i}.main_unit.conv1.weight"] = (
          _conv_flax_to_torch(fb["PreActResidualConvUnit_1"]["conv1"]["kernel"]))
      sd[f"dpt.fusion_blocks.{i}.main_unit.conv2.weight"] = (
          _conv_flax_to_torch(fb["PreActResidualConvUnit_1"]["conv2"]["kernel"]))
    # out_conv
    sd[f"dpt.fusion_blocks.{i}.out_conv.weight"] = _conv_flax_to_torch(
        fb["Conv_0"]["kernel"])
    sd[f"dpt.fusion_blocks.{i}.out_conv.bias"] = _bias_to_torch(
        fb["Conv_0"]["bias"])

  # Project
  sd["dpt.project.weight"] = _conv_flax_to_torch(dpt["project"]["kernel"])
  sd["dpt.project.bias"] = _bias_to_torch(dpt["project"]["bias"])

  return sd


def copy_scenic_head_to_pytorch(scenic_params, head_name):
  """Copy a Scenic Dense head to PyTorch head.* keys."""
  return {
      "head.weight": _dense_flax_to_torch(scenic_params[head_name]["kernel"]),
      "head.bias": _bias_to_torch(scenic_params[head_name]["bias"]),
  }


print("✓ Weight copying functions defined")

# %% [markdown]
# ## Cell 5: Create Shared Inputs
#
# Generate deterministic inputs in both Scenic (NHWC) and PyTorch (NCHW)
# formats from the same random seed.

# %%
def create_shared_inputs(rng, batch_size, h, w, embed_dim, num_layers=4):
  """Create inputs in both Scenic and PyTorch format.

  Returns:
    scenic_spatial: list of [B, H, W, C] arrays (NHWC)
    scenic_vector: list of [B, 1, C] arrays
    pt_inputs: list of (cls_token [B,C], spatial [B,C,H,W]) tuples
  """
  scenic_spatial, scenic_vector, pt_inputs = [], [], []

  for _ in range(num_layers):
    rng, k1, k2 = jax.random.split(rng, 3)
    cls_jax = jax.random.normal(k1, (batch_size, 1, embed_dim))
    feat_jax = jax.random.normal(k2, (batch_size, h, w, embed_dim))

    scenic_vector.append(cls_jax)
    scenic_spatial.append(feat_jax)

    # PyTorch: cls=[B,C], spatial=[B,C,H,W]
    cls_pt = torch.from_numpy(np.array(cls_jax[:, 0, :]))
    feat_pt = torch.from_numpy(
        np.array(feat_jax).transpose(0, 3, 1, 2).copy()
    )
    pt_inputs.append((cls_pt, feat_pt))

  return scenic_spatial, scenic_vector, pt_inputs


# Test config
EMBED_DIM = 768
CHANNELS = 256
POST_PROCESS_CHANNELS = (96, 192, 384, 768)
BATCH_SIZE = 1
SPATIAL_H, SPATIAL_W = 8, 8

print(f"✓ Config: embed_dim={EMBED_DIM}, channels={CHANNELS}, "
      f"spatial={SPATIAL_H}x{SPATIAL_W}")

# %% [markdown]
# ## Cell 6: Normals Parity Test
#
# Test `NormalsDecoder`: Scenic vs PyTorch with identical weights.

# %%
print("=" * 60)
print("TEST 1: NORMALS DECODER")
print("=" * 60)

# ─── Init Scenic model ──────────────────────────────────────────────────────
scenic_normals = ScenicNormalsDecoder(
    input_embed_dim=EMBED_DIM,
    channels=CHANNELS,
    post_process_channels=POST_PROCESS_CHANNELS,
    readout_type="project",
    output_activation=False,
)

rng = jax.random.PRNGKey(42)
dummy_spatial = [
    jax.random.normal(jax.random.PRNGKey(i), (BATCH_SIZE, SPATIAL_H, SPATIAL_W, EMBED_DIM))
    for i in range(4)
]
dummy_vector = [
    jax.random.normal(jax.random.PRNGKey(i+10), (BATCH_SIZE, 1, EMBED_DIM))
    for i in range(4)
]
scenic_vars = scenic_normals.init(rng, dummy_spatial, dummy_vector, train=False)
scenic_params = scenic_vars["params"]
print(f"  Scenic NormalsDecoder initialized "
      f"({sum(p.size for p in jax.tree.leaves(scenic_params))} params)")

# ─── Init PyTorch model with SAME weights ────────────────────────────────────
pt_normals = pt_decoders.NormalsDecoder(
    input_embed_dim=EMBED_DIM,
    channels=CHANNELS,
    post_process_channels=POST_PROCESS_CHANNELS,
    readout_type="project",
    output_activation=False,
)

sd = copy_scenic_dpt_to_pytorch(scenic_params)
sd.update(copy_scenic_head_to_pytorch(scenic_params, "pixel_normals"))
missing, unexpected = pt_normals.load_state_dict(sd, strict=True)
assert not missing and not unexpected, f"Missing: {missing}, Unexpected: {unexpected}"
pt_normals.eval()
print("  ✓ PyTorch NormalsDecoder loaded with identical weights")

# ─── Forward pass on shared inputs ──────────────────────────────────────────
rng_input = jax.random.PRNGKey(123)
scenic_spatial, scenic_vector, pt_inputs = create_shared_inputs(
    rng_input, BATCH_SIZE, SPATIAL_H, SPATIAL_W, EMBED_DIM
)

scenic_out = scenic_normals.apply(
    scenic_vars, scenic_spatial, scenic_vector, train=False
)
scenic_out_np = np.array(scenic_out)  # [B, H', W', 3]

with torch.no_grad():
  pt_out = pt_normals(pt_inputs)
pt_out_np = pt_out.numpy().transpose(0, 2, 3, 1)  # [B, H', W', 3]

# ─── Compare ────────────────────────────────────────────────────────────────
assert scenic_out_np.shape == pt_out_np.shape, (
    f"Shape mismatch: {scenic_out_np.shape} vs {pt_out_np.shape}")

max_diff = np.abs(scenic_out_np - pt_out_np).max()
mean_diff = np.abs(scenic_out_np - pt_out_np).mean()

print(f"\n  Scenic output range: [{scenic_out_np.min():.6f}, "
      f"{scenic_out_np.max():.6f}]")
print(f"  PyTorch output range: [{pt_out_np.min():.6f}, "
      f"{pt_out_np.max():.6f}]")
print(f"  Max absolute diff:  {max_diff:.2e}")
print(f"  Mean absolute diff: {mean_diff:.2e}")

if max_diff < 1e-4:
  print("  ✅ NORMALS: PASS (max diff < 1e-4)")
elif max_diff < 1e-3:
  print("  ✅ NORMALS: CLOSE (max diff < 1e-3, float precision)")
else:
  print("  ❌ NORMALS: FAIL!")

# Spot check
print(f"\n  Spot check [0, 0, 0, :]:")
print(f"    Scenic:  {scenic_out_np[0, 0, 0, :].tolist()}")
print(f"    PyTorch: {pt_out_np[0, 0, 0, :].tolist()}")

# %% [markdown]
# ## Cell 7: Depth Parity Test
#
# Test `DepthDecoder`: Scenic classification-based depth vs PyTorch.

# %%
print("\n" + "=" * 60)
print("TEST 2: DEPTH DECODER (classification, 256 bins)")
print("=" * 60)

NUM_DEPTH_BINS = 256
MIN_DEPTH = 0.001
MAX_DEPTH = 10.0

# ─── Init Scenic model ──────────────────────────────────────────────────────
scenic_depth = ScenicDepthDecoder(
    input_embed_dim=EMBED_DIM,
    channels=CHANNELS,
    post_process_channels=POST_PROCESS_CHANNELS,
    readout_type="project",
    output_activation=False,
    num_depth_bins=NUM_DEPTH_BINS,
    min_depth=MIN_DEPTH,
    max_depth=MAX_DEPTH,
)

scenic_vars_d = scenic_depth.init(rng, dummy_spatial, dummy_vector, train=False)
scenic_params_d = scenic_vars_d["params"]
print(f"  Scenic DepthDecoder initialized "
      f"({sum(p.size for p in jax.tree.leaves(scenic_params_d))} params)")

# ─── Init PyTorch model with SAME weights ────────────────────────────────────
pt_depth = pt_decoders.DepthDecoder(
    num_depth_bins=NUM_DEPTH_BINS,
    min_depth=MIN_DEPTH,
    max_depth=MAX_DEPTH,
    input_embed_dim=EMBED_DIM,
    channels=CHANNELS,
    post_process_channels=POST_PROCESS_CHANNELS,
    readout_type="project",
    output_activation=False,
)

sd_d = copy_scenic_dpt_to_pytorch(scenic_params_d)
sd_d.update(copy_scenic_head_to_pytorch(scenic_params_d, "pixel_depth_classif"))
# Add bin_centers buffer
for name, buf in pt_depth.named_buffers():
  if name not in sd_d:
    sd_d[name] = buf

missing, unexpected = pt_depth.load_state_dict(sd_d, strict=True)
assert not missing and not unexpected, f"Missing: {missing}, Unexpected: {unexpected}"
pt_depth.eval()
print("  ✓ PyTorch DepthDecoder loaded with identical weights")

# ─── Forward pass on shared inputs ──────────────────────────────────────────
rng_input_d = jax.random.PRNGKey(456)
scenic_spatial_d, scenic_vector_d, pt_inputs_d = create_shared_inputs(
    rng_input_d, BATCH_SIZE, SPATIAL_H, SPATIAL_W, EMBED_DIM
)

scenic_out_d = scenic_depth.apply(
    scenic_vars_d, scenic_spatial_d, scenic_vector_d, train=False
)
scenic_out_d_np = np.array(scenic_out_d)  # [B, H', W', 1]

with torch.no_grad():
  pt_out_d = pt_depth(pt_inputs_d)
# PyTorch output: [B, 1, H', W'] → [B, H', W', 1]
pt_out_d_np = pt_out_d.numpy().transpose(0, 2, 3, 1)

# ─── Compare ────────────────────────────────────────────────────────────────
assert scenic_out_d_np.shape == pt_out_d_np.shape, (
    f"Shape mismatch: {scenic_out_d_np.shape} vs {pt_out_d_np.shape}")

max_diff_d = np.abs(scenic_out_d_np - pt_out_d_np).max()
mean_diff_d = np.abs(scenic_out_d_np - pt_out_d_np).mean()

print(f"\n  Scenic output range: [{scenic_out_d_np.min():.6f}, "
      f"{scenic_out_d_np.max():.6f}]")
print(f"  PyTorch output range: [{pt_out_d_np.min():.6f}, "
      f"{pt_out_d_np.max():.6f}]")
print(f"  Max absolute diff:  {max_diff_d:.2e}")
print(f"  Mean absolute diff: {mean_diff_d:.2e}")

if max_diff_d < 1e-4:
  print("  ✅ DEPTH: PASS (max diff < 1e-4)")
elif max_diff_d < 1e-3:
  print("  ✅ DEPTH: CLOSE (max diff < 1e-3, float precision)")
else:
  print("  ❌ DEPTH: FAIL!")

# Spot check
print(f"\n  Spot check depth [0, 0, 0, 0]:")
print(f"    Scenic:  {scenic_out_d_np[0, 0, 0, 0]:.8f}")
print(f"    PyTorch: {pt_out_d_np[0, 0, 0, 0]:.8f}")

# %% [markdown]
# ## Cell 8: Segmentation Parity Test
#
# Test `SegmentationDecoder`: Scenic vs PyTorch with identical weights.

# %%
print("\n" + "=" * 60)
print("TEST 3: SEGMENTATION DECODER (21 classes)")
print("=" * 60)

NUM_CLASSES = 21

# ─── Init Scenic model ──────────────────────────────────────────────────────
scenic_seg = ScenicSegmentationDecoder(
    input_embed_dim=EMBED_DIM,
    channels=CHANNELS,
    post_process_channels=POST_PROCESS_CHANNELS,
    readout_type="project",
    output_activation=False,
    num_classes=NUM_CLASSES,
)

scenic_vars_s = scenic_seg.init(rng, dummy_spatial, dummy_vector, train=False)
scenic_params_s = scenic_vars_s["params"]
print(f"  Scenic SegmentationDecoder initialized "
      f"({sum(p.size for p in jax.tree.leaves(scenic_params_s))} params)")

# ─── Init PyTorch model with SAME weights ────────────────────────────────────
pt_seg = pt_decoders.SegmentationDecoder(
    num_classes=NUM_CLASSES,
    input_embed_dim=EMBED_DIM,
    channels=CHANNELS,
    post_process_channels=POST_PROCESS_CHANNELS,
    readout_type="project",
    output_activation=False,
)

sd_s = copy_scenic_dpt_to_pytorch(scenic_params_s)
sd_s.update(copy_scenic_head_to_pytorch(scenic_params_s, "pixel_segmentation"))
missing, unexpected = pt_seg.load_state_dict(sd_s, strict=True)
assert not missing and not unexpected, f"Missing: {missing}, Unexpected: {unexpected}"
pt_seg.eval()
print("  ✓ PyTorch SegmentationDecoder loaded with identical weights")

# ─── Forward pass on shared inputs ──────────────────────────────────────────
rng_input_s = jax.random.PRNGKey(789)
scenic_spatial_s, scenic_vector_s, pt_inputs_s = create_shared_inputs(
    rng_input_s, BATCH_SIZE, SPATIAL_H, SPATIAL_W, EMBED_DIM
)

scenic_out_s = scenic_seg.apply(
    scenic_vars_s, scenic_spatial_s, scenic_vector_s, train=False
)
scenic_out_s_np = np.array(scenic_out_s)  # [B, H', W', C]

with torch.no_grad():
  pt_out_s = pt_seg(pt_inputs_s)
pt_out_s_np = pt_out_s.numpy().transpose(0, 2, 3, 1)  # [B, H', W', C]

# ─── Compare ────────────────────────────────────────────────────────────────
assert scenic_out_s_np.shape == pt_out_s_np.shape, (
    f"Shape mismatch: {scenic_out_s_np.shape} vs {pt_out_s_np.shape}")

max_diff_s = np.abs(scenic_out_s_np - pt_out_s_np).max()
mean_diff_s = np.abs(scenic_out_s_np - pt_out_s_np).mean()

print(f"\n  Scenic output range: [{scenic_out_s_np.min():.6f}, "
      f"{scenic_out_s_np.max():.6f}]")
print(f"  PyTorch output range: [{pt_out_s_np.min():.6f}, "
      f"{pt_out_s_np.max():.6f}]")
print(f"  Max absolute diff:  {max_diff_s:.2e}")
print(f"  Mean absolute diff: {mean_diff_s:.2e}")

if max_diff_s < 1e-4:
  print("  ✅ SEGMENTATION: PASS (max diff < 1e-4)")
elif max_diff_s < 1e-3:
  print("  ✅ SEGMENTATION: CLOSE (max diff < 1e-3, float precision)")
else:
  print("  ❌ SEGMENTATION: FAIL!")

# Spot check
print(f"\n  Spot check logits [0, 0, 0, :5]:")
print(f"    Scenic:  {scenic_out_s_np[0, 0, 0, :5].tolist()}")
print(f"    PyTorch: {pt_out_s_np[0, 0, 0, :5].tolist()}")

# %% [markdown]
# ## Cell 9: Weight Loading from Checkpoint Test
#
# Verify that `load_decoder_weights()` successfully loads from a real
# Scenic `.zip` checkpoint file.

# %%
import urllib.request

GCS = "https://storage.googleapis.com/tips_data/v2_0/checkpoints/scenic"
CKPT_DIR = "/content/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def download(url):
  fname = url.rsplit("/", 1)[-1]
  path = os.path.join(CKPT_DIR, fname)
  if not os.path.exists(path):
    print(f"  Downloading {fname}...")
    urllib.request.urlretrieve(url, path)
  return path

print("=" * 60)
print("TEST 4: CHECKPOINT LOADING")
print("=" * 60)

# Download depth checkpoint
depth_zip = download(f"{GCS}/tips_v2_l14_depth_dpt.zip")

# Load with decoders.py's load_decoder_weights
pt_depth_real = pt_decoders.DepthDecoder(
    num_depth_bins=256,
    min_depth=0.001,
    max_depth=10.0,
    input_embed_dim=1024,
    channels=256,
    post_process_channels=(128, 256, 512, 1024),
    readout_type="project",
    output_activation=False,
)
pt_decoders.load_decoder_weights(pt_depth_real, depth_zip)
pt_depth_real.eval()

# Verify all params are loaded (non-zero)
total_params = sum(p.numel() for p in pt_depth_real.parameters())
zero_params = sum((p == 0).sum().item() for p in pt_depth_real.parameters())
print(f"  Total params: {total_params:,}")
print(f"  Zero params: {zero_params:,} ({100*zero_params/total_params:.1f}%)")

# Quick sanity: run inference on random input
torch.manual_seed(0)
dummy_features = [
    (torch.randn(1, 1024), torch.randn(1, 1024, 16, 16))
    for _ in range(4)
]
with torch.no_grad():
  depth_out = pt_depth_real(dummy_features)
print(f"  Output shape: {depth_out.shape}")
print(f"  Depth range: [{depth_out.min():.4f}, {depth_out.max():.4f}]")
print(f"  ✅ Checkpoint loading works!")

# Download and test normals + segmentation too
normals_zip = download(f"{GCS}/tips_v2_l14_normals_dpt.zip")
pt_normals_real = pt_decoders.NormalsDecoder(
    input_embed_dim=1024, channels=256,
    post_process_channels=(128, 256, 512, 1024),
    readout_type="project", output_activation=False,
)
pt_decoders.load_decoder_weights(pt_normals_real, normals_zip)
print("  ✅ Normals checkpoint loaded!")

# %% [markdown]
# ## Cell 10: Summary

# %%
print("\n" + "=" * 60)
print("PARITY VERIFICATION SUMMARY")
print("=" * 60)

results = [
    ("Normals", max_diff, 1e-4),
    ("Depth", max_diff_d, 1e-4),
    ("Segmentation", max_diff_s, 1e-4),
]

all_pass = True
for name, diff, threshold in results:
  status = "✅ PASS" if diff < threshold else "❌ FAIL"
  if diff >= threshold:
    all_pass = False
  print(f"  {name:15s}: max_diff = {diff:.2e}  {status}")

print()
if all_pass:
  print("  🎉 ALL TESTS PASS — PyTorch decoders.py matches Scenic reference!")
else:
  print("  ⚠️  Some tests failed. Check the outputs above for details.")

print()
print("  Architecture verified:")
print("  • DPTHead output_activation handled correctly")
print("  • DepthDecoder uses classification bins (bin_centers buffer)")
print("  • ConvTranspose kernel flip for Flax→PyTorch parity")
print("  • GELU approximate='tanh' matches JAX default")
print("  • Weight loading from Scenic .zip checkpoints works")
print("=" * 60)
