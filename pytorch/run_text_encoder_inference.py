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

r"""Running TIPS (https://arxiv.org/abs/2410.16512) text encoder inference.

Usage:
```python
python run_text_encoder_inference.py --model_path=${PATH_TO_LOW_RES_CHECKPOINT} \
    --model_variant=g --tokenizer_path=${PATH_TO_TOKENIZER} \
    --text_input="Hello world."
```
"""

import argparse
import io
import numpy as np
import torch
from tips.pytorch import text_encoder

MAX_LEN = 64
VOCAB_SIZE = 32000

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path', default=None, required=True, help='The path to the model.'
)
parser.add_argument(
    '--model_variant',
    default=None,
    required=True,
    help='The variant of the model.',
)
parser.add_argument(
    '--tokenizer_path',
    default=None,
    required=True,
    help='The path to the tokenizer.',
)
parser.add_argument(
    '--text_input',
    default=None,
    required=True,
    help='The text input to the model.',
)


def get_config(v: str):
  return {
      'hidden_size': {'S': 384, 'B': 768, 'L': 1024, 'So400m': 1152, 'g': 1536}[
          v
      ],
      'mlp_dim': {'S': 1536, 'B': 3072, 'L': 4096, 'So400m': 4304, 'g': 6144}[
          v
      ],
      'num_heads': {'S': 6, 'B': 12, 'L': 16, 'So400m': 16, 'g': 24}[v],
      'num_layers': {'S': 12, 'B': 12, 'L': 12, 'So400m': 27, 'g': 12}[v],
  }


def main(args):

  with open(args.model_path, 'rb') as fin:
    inbuffer = io.BytesIO(fin.read())
  np_weights_text = np.load(inbuffer, allow_pickle=False)

  pytorch_weights_text = {}
  for key, value in np_weights_text.items():
    pytorch_weights_text[key] = torch.from_numpy(value)
  pytorch_weights_text.pop('temperature')

  with torch.no_grad():
    # Define the text model.
    model_text = text_encoder.TextEncoder(
        get_config(args.model_variant),
        vocab_size=VOCAB_SIZE,
    )
    model_text.load_state_dict(pytorch_weights_text)

    tokenizer_obj = text_encoder.Tokenizer(tokenizer_path=args.tokenizer_path)
    text_ids, text_paddings = tokenizer_obj.tokenize(
        [args.text_input], max_len=MAX_LEN
    )
    text_embedding = (
        model_text(torch.from_numpy(text_ids), torch.from_numpy(text_paddings))
        .detach()
        .numpy()
        .squeeze()
    )
    text_embedding = text_embedding / np.linalg.norm(
        text_embedding, ord=2, axis=-1, keepdims=True
    ).clip(min=1e-3)
    print(text_embedding.tolist())


if __name__ == '__main__':
  main(parser.parse_args())
