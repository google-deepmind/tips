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

r"""Running TIPS / TIPSv2 image encoder inference.

Supports both TIPSv1 and TIPSv2 model variants.  TIPSv2 models use a (32, 32)
positional-embedding grid instead of (16, 16).  An optional ``--decoder_path``
flag enables DPT-based dense prediction (segmentation, depth, or normals).

Usage (encoder only):
```python
python run_image_encoder_inference.py \
    --model_path=${PATH_TO_CHECKPOINT} \
    --image_file=${PATH_TO_IMAGE} \
    --model_variant=g
```

Usage (encoder + DPT decoder for segmentation):
```python
python run_image_encoder_inference.py \
    --model_path=${PATH_TO_CHECKPOINT} \
    --image_file=${PATH_TO_IMAGE} \
    --model_variant=L \
    --decoder_path=${PATH_TO_DECODER_CHECKPOINT} \
    --decoder_task=segmentation
```
"""

import argparse
import io

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from tips.pytorch import image_encoder

IMAGE_MEAN = (0, 0, 0)
IMAGE_STD = (1.0, 1.0, 1.0)
PATCH_SIZE = 14

# Mapping from model variant to (embed_dim, post_process_channels).
_DECODER_CONFIGS = {
    'S': (384, (48, 96, 192, 384)),
    'B': (768, (96, 192, 384, 768)),
    'L': (1024, (128, 256, 512, 1024)),
    'So400m': (1152, (144, 288, 576, 1152)),
    'g': (1536, (192, 384, 768, 1536)),
}

# Number of output channels per decoder task.
_DECODER_TASK_OUT_CHANNELS = {
    'segmentation': 150,  # ADE20K classes
    'depth': 1,
    'normals': 3,
}

# Which intermediate layers to extract for the 4-level DPT head.
_INTERMEDIATE_LAYERS = {
    'S': [2, 5, 8, 11],
    'B': [2, 5, 8, 11],
    'L': [4, 11, 17, 23],
    'So400m': [5, 13, 20, 26],
    'g': [9, 19, 29, 39],
}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path', default=None, required=True, help='The path to the model.'
)
parser.add_argument(
    '--image_file',
    default=None,
    required=True,
    help='The path to the image file for inference.',
)
parser.add_argument(
    '--is_low_res',
    action='store_true',
    help='Whether the model is low-resolution (224px instead of 448px).',
)
parser.add_argument(
    '--model_variant',
    default=None,
    required=True,
    choices=['S', 'B', 'L', 'So400m', 'g'],
    help='The variant of the model.',
)
parser.add_argument(
    '--decoder_path',
    default=None,
    help='Optional path to a DPT decoder checkpoint (.npz).  '
         'When provided, runs dense prediction on top of the encoder.',
)
parser.add_argument(
    '--decoder_task',
    default='segmentation',
    choices=list(_DECODER_TASK_OUT_CHANNELS.keys()),
    help='Dense prediction task (only used when --decoder_path is set).',
)


def main(args):

  image_size = 224 if args.is_low_res else 448
  model_def = {
      'S': image_encoder.vit_small,
      'B': image_encoder.vit_base,
      'L': image_encoder.vit_large,
      'So400m': image_encoder.vit_so400m,
      'g': image_encoder.vit_giant2,
  }[args.model_variant]

  ffn_layer = 'swiglu' if args.model_variant == 'g' else 'mlp'

  # Load checkpoint.
  checkpoint = dict(np.load(args.model_path, allow_pickle=False))
  for key in checkpoint:
    checkpoint[key] = torch.tensor(checkpoint[key])

  # Read and pre-process the image.
  with open(args.image_file, 'rb') as fd:
    image_bytes = io.BytesIO(fd.read())
    pil_image = Image.open(image_bytes)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)

  with torch.no_grad():
    model = model_def(
        img_size=image_size,
        patch_size=PATCH_SIZE,
        ffn_layer=ffn_layer,
        block_chunks=0,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    model.load_state_dict(checkpoint)

    # ------------------------------------------------------------------
    # Encoder-only: compute CLS token embeddings.
    # ------------------------------------------------------------------
    outputs = model(input_batch)
    first_cls_token = outputs[0].detach().numpy().squeeze()
    second_cls_token = outputs[1].detach().numpy().squeeze()

    first_cls_token = first_cls_token / np.linalg.norm(
        first_cls_token, ord=2, axis=-1, keepdims=True
    ).clip(min=1e-3)
    second_cls_token = second_cls_token / np.linalg.norm(
        second_cls_token, ord=2, axis=-1, keepdims=True
    ).clip(min=1e-3)
    print('First cls token: ', first_cls_token.tolist())
    print('Second cls token: ', second_cls_token.tolist())

    # ------------------------------------------------------------------
    # Optional: DPT decoder for dense prediction.
    # ------------------------------------------------------------------
    if args.decoder_path is not None:
      from tips.pytorch.decoders import Decoder, load_decoder_weights  # pylint: disable=g-import-not-at-top

      embed_dim, ppc = _DECODER_CONFIGS[args.model_variant]
      out_channels = _DECODER_TASK_OUT_CHANNELS[args.decoder_task]
      layer_ids = _INTERMEDIATE_LAYERS[args.model_variant]

      # Extract intermediate features with class tokens.
      intermediate = model.get_intermediate_layers(
          input_batch,
          n=layer_ids,
          reshape=True,
          return_class_token=True,
      )
      # intermediate is a tuple of (patch_tokens, cls_token) per layer.
      intermediate_features = [
          (cls.unsqueeze(1), patches) for patches, cls in intermediate
      ]

      decoder = Decoder(
          out_channels=out_channels,
          input_embed_dim=embed_dim,
          post_process_channels=ppc,
      )
      decoder = load_decoder_weights(decoder, args.decoder_path)
      decoder.eval()

      original_size = (pil_image.height, pil_image.width)
      dense_output = decoder(intermediate_features, image_size=original_size)
      print(
          f'Dense prediction ({args.decoder_task}): '
          f'shape={tuple(dense_output.shape)}'
      )


if __name__ == '__main__':
  main(parser.parse_args())
