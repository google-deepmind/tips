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

r"""Running TIPS (https://arxiv.org/abs/2410.16512) ViT-g model inference.

Usage:
```python
python run_image_encoder_inference.py --model_path=${PATH_TO_LOW_RES_CHECKPOINT} \
    --image_file=${PATH_TO_IMAGE} --is_low_res --model_variant=g
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
    help='Whether the model is low-resolution.',
)
parser.add_argument(
    '--model_variant',
    default=None,
    required=True,
    help='The variant of the model.',
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

  # Run inference on the image.
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

    # Compute embeddings from two CLS tokens.
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


if __name__ == '__main__':
  main(parser.parse_args())
