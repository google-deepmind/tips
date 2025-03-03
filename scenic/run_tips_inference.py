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

"""Runs TIPS inference."""

import argparse
import os
import cv2
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from tips.scenic.configs import tips_model_config
from tips.scenic.models import text
from tips.scenic.models import tips
from tips.scenic.utils import checkpoint
from tips.scenic.utils import feature_viz


parser = argparse.ArgumentParser()
parser.add_argument(
    '--image_width',
    type=int,
    default=448,
    help='Image width.',
)
parser.add_argument(
    '--variant',
    type=str,
    default='tips_oss_b14_highres_distilled',
    choices=(
        'tips_oss_g14_highres',
        'tips_oss_g14_lowres',
        'tips_oss_so400m14_highres_largetext_distilled',
        'tips_oss_l14_highres_distilled',
        'tips_oss_b14_highres_distilled',
        'tips_oss_s14_highres_distilled',
    ),
    help='Model variant.',
)
parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default='checkpoints/',
    help='The directory of the checkpoints and the tokenizer.',
)
parser.add_argument(
    '--image_path',
    type=str,
    default='images/example_image.jpg',
    help='The path to the image file.'
)


def main() -> None:
  args = parser.parse_args()
  image_width = args.image_width
  image_shape = (image_width,) * 2
  variant = args.variant
  checkpoint_dir = args.checkpoint_dir
  image_path = args.image_path

  # Load the model configuration.
  model_config = tips_model_config.get_config(variant)

  # Load the vision encoder.
  model_vision = tips.VisionEncoder(
      variant=model_config.variant,
      pooling=model_config.pooling,
      num_cls_tokens=model_config.num_cls_tokens)
  init_params_vision = model_vision.init(
      jax.random.PRNGKey(0), jnp.ones([1, *image_shape, 3]), train=False)
  params_vision = checkpoint.load_checkpoint(
      os.path.join(checkpoint_dir, f'{variant}_vision.npz'),
      init_params_vision['params'])

  # Load the text encoder.
  tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.model')
  tokenizer = text.Tokenizer(tokenizer_path)
  model_text = tips.TextEncoder(variant=model_config.variant)
  init_params_text = model_text.init(
      jax.random.PRNGKey(0),
      ids=jnp.ones((4, 64), dtype=jnp.int32),
      paddings=jnp.zeros((4, 64), dtype=jnp.int32),
      train=False)
  init_params_text['params']['temperature_contrastive'] = (
      np.array(0, dtype=np.float32))
  params_text = checkpoint.load_checkpoint(
      os.path.join(checkpoint_dir, f'{variant}_text.npz'),
      init_params_text['params'])

  # Load and preprocess the image.
  image = jnp.array(Image.open(image_path)).astype(jnp.float32) / 255.
  image = jax.image.resize(image, (*image_shape, 3), method='bilinear')
  image = image.astype(jnp.float32)

  # Run inference on the image.
  spatial_features, embeddings_vision = model_vision.apply(
      {'params': params_vision}, image[None], train=False)
  # We choose the first CLS token (the second one is better for dense tasks.).
  cls_token = feature_viz.normalize(embeddings_vision[:, 0, :])

  # Run inference on text.
  text_input = [
      'A ship', 'holidays', 'a toy dinosaur', 'Two astronauts',
      'a real dinosaur', 'A streetview image of burger kings',
      'a streetview image of mc donalds']
  text_ids, text_paddings = tokenizer.tokenize(text_input, max_len=64)
  embeddings_text = model_text.apply(
      {'params': params_text},
      ids=text_ids,
      paddings=text_paddings,
      train=False)
  embeddings_text = feature_viz.normalize(embeddings_text)

  # Compute cosine similariy.
  cos_sim = nn.softmax(
      ((cls_token @ embeddings_text.T) /
       params_text['temperature_contrastive']), axis=-1)
  label_idxs = jnp.argmax(cos_sim, axis=-1)
  cos_sim_max = jnp.max(cos_sim, axis=-1)
  label_predicted = text_input[label_idxs[0].item()]
  similarity = cos_sim_max[0].item()

  # Compute PCA of patch tokens.
  pca_obj = feature_viz.PCAVisualizer(spatial_features)
  image_pca = pca_obj(spatial_features)[0]
  image_pca = np.asarray(jax.image.resize(
      image_pca, (*image_shape, 3), method='nearest'))

  # Display the results.
  cv2.imshow(
      f'{label_predicted},  prob: {similarity*100:.1f}%',
      np.concatenate([image, image_pca], axis=1)[..., ::-1])
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
