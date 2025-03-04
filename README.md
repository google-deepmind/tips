# TIPS: Text-Image Pretraining with Spatial awareness (ICLR 2025)

This repository contains the implementation and models introduced in
TIPS: Text-Image Pretraining with Spatial Awareness, published at ICLR 2025.

**Quick Links:**
[Paper](https://arxiv.org/abs/2410.16512) |
[Pytorch Notebook](./pytorch/TIPS_Demo.ipynb) |
[Scenic Notebook](./scenic/notebooks/TIPS_Demo.ipynb)

We provide both Pytorch and Jax (Scenic) implementations:

- `tips/pytorch/`: PyTorch inference for the model. The image tower largely
follows the official [DINOv2 definition](https://github.com/facebookresearch/dinov2).
- `tips/scenic/`: Jax-based inference using the
[scenic library](https://github.com/google-research/scenic).

<p align="center">
  <img
    src="./docs/images/overview.png"
    style="width:75%;"
  >
</p>

**Abstract**
<div style="text-align: justify;">
While image-text representation learning has become very popular
in recent years, existing models tend to lack spatial awareness and have limited
direct applicability for dense understanding tasks. For this reason,
self-supervised image-only pretraining is still the go-to method for many dense
vision applications (e.g. depth estimation, semantic segmentation), despite the
lack of explicit supervisory signals. In this paper, we close this gap between
image-text and self-supervised learning, by proposing a novel general-purpose
image-text model, which can be effectively used off the shelf for dense and
global vision tasks. Our method, which we refer to as Text-Image Pretraining
with Spatial awareness (TIPS), leverages two simple and effective insights.
First, on textual supervision: we reveal that replacing noisy web image captions
by synthetically generated textual descriptions boosts dense understanding
performance significantly, due to a much richer signal for learning spatially
aware representations. We propose an adapted training method that combines noisy
and synthetic captions, resulting in improvements across both dense and global
understanding tasks. Second, on the learning technique: we propose to combine
contrastive image-text learning with self-supervised masked image modeling, to
encourage spatial coherence, unlocking substantial enhancements for downstream
applications. Building on these two ideas, we scale our model using the
transformer architecture, trained on a curated set of public images. Our
experiments are conducted on 8 tasks involving 16 datasets in total,
demonstrating strong off-the-shelf performance on both dense and global
understanding, for several image-only and image-text tasks.
</div>

<p align="center">
  <img
    src="./docs/images/qualitative.png"
    style="width:80%;"
  >
</p>


## Checkpoints
We provide links to all available checkpoints, for both Pytorch and Jax model
definitions, together with representative evals.

 Model size  | #Params vision / text | Pytorch ckp.                                             | Jax ckp.                                                 | PASCAL seg.↑ | NYU-depth↓ | ImageNet-KNN↑ | UNED-KNN↑ | Flickr T→I↑ | Flickr I→T↑
:---------- | :--------------------- | :------------------------------------------------------: | :------------------------------------------------------: | :---------: | :-------: | :----------: | :------: | :--------: | :--------:
g/14-HR     |  1.1B / 389.1M         | [vision][pth-g14-hr-vision] \| [text][pth-g14-hr-text]   | [vision][jax-g14-hr-vision] \| [text][jax-g14-hr-text]   | 83.1        | 0.363     | 83.2         | 68.4     | 93.8       | 83.8
g/14-LR     |  1.1B / 389.1M         | [vision][pth-g14-lr-vision] \| [text][pth-g14-lr-text]   | [vision][jax-g14-lr-vision] \| [text][jax-g14-lr-text]   | 82.0        | 0.390     | 83.6         | 71.5     | 93.4       | 82.1
SO/14-HR    |  412.4M / 448.3M       | [vision][pth-so14-hr-vision] \| [text][pth-so14-hr-text] | [vision][jax-so14-hr-vision] \| [text][jax-so14-hr-text] | 83.7        | 0.362     | 83.0         | 68.6     | 94.2       | 83.8
L/14-HR     |  303.2M / 183.9M       | [vision][pth-l14-hr-vision] \| [text][pth-l14-hr-text]   | [vision][jax-l14-hr-vision] \| [text][jax-l14-hr-text]   | 83.9        | 0.372     | 82.5         | 67.8     | 93.6       | 83.5
B/14-HR     |  85.7M / 109.6M        | [vision][pth-b14-hr-vision] \| [text][pth-b14-hr-text]   | [vision][jax-b14-hr-vision] \| [text][jax-b14-hr-text]   | 82.9        | 0.379     | 80.0         | 62.7     | 91.3       | 79.4
S/14-HR     |  21.6M / 33.6M         | [vision][pth-s14-hr-vision] \| [text][pth-s14-hr-text]   | [vision][jax-s14-hr-vision] \| [text][jax-s14-hr-text]   | 80.6        | 0.425     | 75.1         | 57.7     | 86.3       | 74.7

## Using Pytorch

### Installation
Manage dependencies with a custom environment (eg. Conda)

```bash
conda create -n tips python=3.11

# Activate the environment.
conda activate tips
```

Install Pytorch dependencies.

```bash
# Install pytorch (change to GPU version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies.
pip install tensorflow_text mediapy jax jaxlib scikit-learn

# Optionally, install Jupyter to use the notebook.
pip install jupyter
```

Clone the code from this repo.

```bash
git clone https://github.com/google-deepmind/tips.git

# Add the current directory to PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Download the checkpoints locally. The script downloads all released checkpoints.
Please adjust accordingly.

```bash
cd tips/pytorch/checkpoints
chmod +x download_checkpoints.sh
./download_checkpoints.sh
cd ../../..
```

### Usage (Pytorch)

To run inference on one image and get the L2-normalized image embedding from the
1st and 2nd CLS token, one can use the following:

```bash
cd tips/pytorch && \
python run_image_encoder_inference.py \
  --model_path=${PATH_TO_CHECKPOINT} \
  --image_file=${PATH_TO_IMAGE} \
  --model_variant=${MODEL_VARIANT}
```

One can use `is_low_res` to specify whether a low-resolution or high-resolution
checkpoint is used.

To run text model inference and get the L2-normalized text embedding, please use
the following cmd

```bash
cd tips/pytorch && \
python run_text_encoder_inference.py \
  --model_path=${PATH_TO_CHECKPOINT} \
  --tokenizer_path=${PATH_TO_TOKENIZER} \
  --model_variant=${MODEL_VARIANT} \
  --text_input=${TEXT_INPUT}
```

We also provide a simple notebook demo:

```bash
jupyter-notebook
```
Then navigate to `tips/pytorch/TIPS_Demo.ipynb`.

## Using Jax (Scenic)

### Installation
Similar to using Pytorch, manage dependencies with a custom environment.

```bash
conda create -n tips python=3.11

# Activate the environment.
conda activate tips
```

```bash
# Install scenic.
git clone https://github.com/google-research/scenic.git scenic_src
cd scenic_src
pip install .
cd ..
rm -rf scenic_src

# Install other dependencies.
pip install pillow scikit-learn opencv-python tensorflow_text

# Optionally, install Jupyter to use the notebook.
pip install jupyter mediapy

# In case of using CUDA, install the CUDA-supported JAX libraries.
# For example, for CUDA 12 run:
# pip install --upgrade "jax[cuda12_pip]" -f \
#   https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Clone the code from the this repo.

```bash
git clone https://github.com/google-deepmind/tips.git

# Add the current directory to PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Download the checkpoints (different files from Pytorch).

```bash
cd tips/scenic/checkpoints
chmod +x download_checkpoints.sh
./download_checkpoints.sh
cd ../../..
```

### Usage (Jax)

To run inference on an image, use the following script:

```bash
cd tips/scenic
python run_tips_inference.py
```

Alternatively, try the demo in the notebook:

```bash
jupyter-notebook
```
Then navigate to `tips/scenic/notebooks/TIPS_Demo.ipynb`.

## Citing this work

The paper can be found on [arXiv](https://arxiv.org/abs/2410.16512).
Please consider citing this work using:

```
@InProceedings{tips_paper,
    Title={{TIPS: Text-Image Pretraining with Spatial Awareness}},
    Author={Maninis, Kevis-Kokitsi and Chen, Kaifeng and Ghosh, Soham and Karpur, Arjun and Chen, Koert and Xia, Ye and Cao, Bingyi and Salz, Daniel and Han, Guangxing and Dlabal, Jan and Gnanapragasam, Dan and Seyedhosseini, Mojtaba and Zhou, Howard and Araujo, Andr\'e},
    Booktitle={ICLR},
    year={2025},
}
```

## License and disclaimer

Copyright 2025 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

[jax-g14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_g14_highres_vision.npz
[jax-g14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_g14_highres_text.npz
[jax-g14-lr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_g14_lowres_vision.npz
[jax-g14-lr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_g14_lowres_text.npz
[jax-so14-hr-vision]: https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_so400m14_highres_largetext_distilled_vision.npz
[jax-so14-hr-text]:   https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_so400m14_highres_largetext_distilled_text.npz
[jax-l14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_l14_highres_distilled_vision.npz
[jax-l14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_l14_highres_distilled_text.npz
[jax-b14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_b14_highres_distilled_vision.npz
[jax-b14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_b14_highres_distilled_text.npz
[jax-s14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_s14_highres_distilled_vision.npz
[jax-s14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_s14_highres_distilled_text.npz

[pth-g14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_g14_highres_vision.npz
[pth-g14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_g14_highres_text.npz
[pth-g14-lr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_g14_lowres_vision.npz
[pth-g14-lr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_g14_lowres_text.npz
[pth-so14-hr-vision]: https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_so400m14_highres_largetext_distilled_vision.npz
[pth-so14-hr-text]:   https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_so400m14_highres_largetext_distilled_text.npz
[pth-l14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_l14_highres_distilled_vision.npz
[pth-l14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_l14_highres_distilled_text.npz
[pth-b14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_b14_highres_distilled_vision.npz
[pth-b14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_b14_highres_distilled_text.npz
[pth-s14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_s14_highres_distilled_vision.npz
[pth-s14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_s14_highres_distilled_text.npz
