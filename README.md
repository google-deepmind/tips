[![Demo-Colab-Pytorch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/tips/blob/main/pytorch/TIPS_Demo.ipynb)
[![Demo-HF](https://img.shields.io/badge/Demo-HF-orange)](https://huggingface.co/spaces/google/TIPSv2)
[![Models-HF](https://img.shields.io/badge/Models-HF-orange)](https://huggingface.co/collections/google/tipsv2)
[![Webpage](https://img.shields.io/badge/Webpage-yellow)](https://gdm-tipsv2.github.io/)
[![Paper](https://img.shields.io/badge/TIPSv2-arXiv.XXXX.YYYYY-B3181B.svg)](https://arxiv.org/abs/XXXX.YYYYY)
[![Paper](https://img.shields.io/badge/TIPSv1-arXiv.2410.16512-B3181B.svg)](https://arxiv.org/abs/2410.16512)
<br/>

# TIPS / TIPSv2

This repository contains the implementation and models introduced in:
* TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment, CVPR 2026
* TIPS: Text-Image Pretraining with Spatial Awareness, ICLR 2025

<p align="center">
  <img
    src="./docs/images/overview.png"
    style="width:75%;"
  >
</p>

The **TIPS** series of models (**T**ext-**I**mage **P**retraining with **S**patial Awareness) are foundational image-text encoders built for general-purpose computer vision and multimodal applications. Our models were validated on a comprehensive suite of 9 tasks and 20 datasets, displaying excellent performance that matches or exceeds other recent vision encoders. with particularly strong spatial awareness.

We recommend using the latest version, TIPSv2, but still provide the earlier TIPSv1 for completeness. For a more detailed overview, please visit the <a href="https://gdm-tipsv2.github.io/">Project Webpage</a> and check out the papers:
[![Paper](https://img.shields.io/badge/v2-arXiv.XXXX.YYYYY-B3181B.svg)](https://arxiv.org/abs/XXXX.YYYYY)
[![Paper](https://img.shields.io/badge/v1-arXiv.2410.16512-B3181B.svg)](https://arxiv.org/abs/2410.16512)

See also our [demos](#demos) for a quick start.

<p align="center">
  <img
    src="./docs/images/pca.png"
    style="width:60%;"
  >
</p>


## How to use
We provide both Pytorch and Jax (Scenic) implementations:

- `tips/pytorch/`: PyTorch inference for the model.
- `tips/scenic/`: Jax-based inference using the
[scenic library](https://github.com/google-research/scenic).

We provide links to all available checkpoints, for both Pytorch and Jax model
definitions, together with representative evals.

### v2 models
| Model size | #Params vision / text | Pytorch ckp. | Jax ckp. | PASCAL seg.↑ | NYU-depth↓ | ImageNet-KNN↑ | Flickr I→T↑ | Flickr T→I↑ | ADE150-ZS↑ |
| :--------- | :-------------------- | :----------: | :------: | :---------: | :-------: | :----------: | :------: | :--------: | :--------: |
| g/14       | 1.1B / 389.1M         | [vision][v2-pth-g14-vision] \| [text][v2-pth-g14-text]  | [vision][v2-jax-g14-vision] \| [text][v2-jax-g14-text]  | 85.1 | 0.334 | 83.7 | 95.1 | 85.9 | 17.8 |
| SO/14      | 412.4M / 448.3M       | [vision][v2-pth-so14-vision] \| [text][v2-pth-so14-text]| [vision][v2-jax-so14-vision] \| [text][v2-jax-so14-text]| 85.2 | 0.339 | 82.8 | 94.8 | 84.0 | 23.3 |
| L/14       | 303.2M / 183.9M       | [vision][v2-pth-l14-vision] \| [text][v2-pth-l14-text]  | [vision][v2-jax-l14-vision] \| [text][v2-jax-l14-text]  | 85.1 | 0.339 | 82.5 | 95.4 | 83.3 | 24.7 |
| B/14       | 85.7M / 109.6M        | [vision][v2-pth-b14-vision] \| [text][v2-pth-b14-text]  | [vision][v2-jax-b14-vision] \| [text][v2-jax-b14-text]  | 84.0 | 0.374 | 79.8 | 92.6 | 80.0 | 17.4 |


### v1 models
| Model size | #Params vision / text | Pytorch ckp.                                            | Jax ckp.                                                 | PASCAL seg.↑ | NYU-depth↓ | ImageNet-KNN↑ | UNED-KNN↑ | Flickr I→T↑ | Flickr T→I↑ |
| :--------- | :-------------------- | :------------------------------------------------------: | :------------------------------------------------------: | :---------: | :-------: | :----------: | :------: | :--------: | :--------: |
| g/14-HR    | 1.1B / 389.1M         | [vision][v1-pth-g14-hr-vision] \| [text][v1-pth-g14-hr-text] | [vision][v1-jax-g14-hr-vision] \| [text][v1-jax-g14-hr-text] | 83.1        | 0.363     | 83.2         | 68.4     | 93.8       | 83.8       |
| g/14-LR    | 1.1B / 389.1M         | [vision][v1-pth-g14-lr-vision] \| [text][v1-pth-g14-lr-text] | [vision][v1-jax-g14-lr-vision] \| [text][v1-jax-g14-lr-text] | 82.0        | 0.390     | 83.6         | 71.5     | 93.4       | 82.1       |
| SO/14-HR   | 412.4M / 448.3M       | [vision][v1-pth-so14-hr-vision] \| [text][v1-pth-so14-hr-text]| [vision][v1-jax-so14-hr-vision] \| [text][v1-jax-so14-hr-text]| 83.7        | 0.362     | 83.0         | 68.6     | 94.2       | 83.8       |
| L/14-HR    | 303.2M / 183.9M       | [vision][v1-pth-l14-hr-vision] \| [text][v1-pth-l14-hr-text] | [vision][v1-jax-l14-hr-vision] \| [text][v1-jax-l14-hr-text] | 83.9        | 0.372     | 82.5         | 67.8     | 93.6       | 83.5       |
| B/14-HR    | 85.7M / 109.6M        | [vision][v1-pth-b14-hr-vision] \| [text][v1-pth-b14-hr-text] | [vision][v1-jax-b14-hr-vision] \| [text][v1-jax-b14-hr-text] | 82.9        | 0.379     | 80.0         | 62.7     | 91.3       | 79.4       |
| S/14-HR    | 21.6M / 33.6M         | [vision][v1-pth-s14-hr-vision] \| [text][v1-pth-s14-hr-text] | [vision][v1-jax-s14-hr-vision] \| [text][v1-jax-s14-hr-text] | 80.6        | 0.425     | 75.1         | 57.7     | 86.3       | 74.7       |


## Demos

[![Demo-HF](https://img.shields.io/badge/Demo-HF-orange)](https://huggingface.co/spaces/google/TIPSv2): Demo on Zero-shot segmentation / Depth and Normals / Feature visualization / Supervised seg <br>
[![Inference-Colab-Pytorch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/tips/blob/main/pytorch/TIPS_Demo.ipynb): Inference Colab in Pytorch <br>
[![Inference-Colab-Jax](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/tips/blob/main/scenic/notebooks/TIPS_Demo.ipynb): Inference Colab in Jax <br>

We also provide task-specific notebooks:

[![ZS-Pytorch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/tips/blob/main/pytorch/TIPS_zeroshot_segmentation.ipynb): Zero-shot segmentation visualization (Pytorch) <br>
[![FG-Seg-Pytorch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/tips/blob/main/pytorch/TIPS_foreground_segmentation_demo.ipynb): Train a linear head for foreground segmentation(Pytorch) <br>
[![DPT-Pytorch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/tips/blob/main/pytorch/TIPSv2_Segmentation_Depth_Normal_DPT_Pytorch.ipynb): Inference using a DPT head for segmentation, depth, normals (Pytorch) <br>


## Local Installation
To install locally instead of using the Colabs/HF, please follow the instructions below.

### Installation (Pytorch)
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

## Installation (JAX/Scenic)

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

## Citing this work

The manuscripts for TIPS v1 and v2 can be found on arXiv ([v1](https://arxiv.org/abs/2410.16512), [v2](https://arxiv.org/abs/TODO)).

Please consider citing this work using:

```
@InProceedings{tips_v2_paper,
    Title={{TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment}},
    Author={Cao, Bingyi and Chen, Koert and Maninis, Kevis-Kokitsi and Chen, Kaifeng and Karpur, Arjun and Xia, Ye and Dua, Sahil and Dabral, Tanmaya and Han, Guangxing and Han, Bohyung and Ainslie, Joshua and Bewley, Alex and Jacob, Mithun and Wagner, Ren\'e and Ramos, Washington and Choromanski, Krzysztof and Seyedhosseini, Mojtaba and Zhou, Howard and Araujo, Andr\'e},
    Booktitle={CVPR},
    year={2026},
}

@InProceedings{tips_v1_paper,
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

[v2-jax-g14-vision]:  https://storage.googleapis.com/tips_data/v2_0/checkpoints/scenic/tips_v2_g14_vision.npz
[v2-jax-g14-text]:    https://storage.googleapis.com/tips_data/v2_0/checkpoints/scenic/tips_v2_g14_text.npz
[v2-jax-so14-vision]: https://storage.googleapis.com/tips_data/v2_0/checkpoints/scenic/tips_v2_so14_vision.npz
[v2-jax-so14-text]:   https://storage.googleapis.com/tips_data/v2_0/checkpoints/scenic/tips_v2_so14_text.npz
[v2-jax-l14-vision]:  https://storage.googleapis.com/tips_data/v2_0/checkpoints/scenic/tips_v2_l14_vision.npz
[v2-jax-l14-text]:    https://storage.googleapis.com/tips_data/v2_0/checkpoints/scenic/tips_v2_l14_text.npz
[v2-jax-b14-vision]:  https://storage.googleapis.com/tips_data/v2_0/checkpoints/scenic/tips_v2_b14_vision.npz
[v2-jax-b14-text]:    https://storage.googleapis.com/tips_data/v2_0/checkpoints/scenic/tips_v2_b14_text.npz

[v2-pth-g14-vision]:  https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_g14_vision.npz
[v2-pth-g14-text]:    https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_g14_text.npz
[v2-pth-so14-vision]: https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_so14_vision.npz
[v2-pth-so14-text]:   https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_so14_text.npz
[v2-pth-l14-vision]:  https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_l14_vision.npz
[v2-pth-l14-text]:    https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_l14_text.npz
[v2-pth-b14-vision]:  https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_b14_vision.npz
[v2-pth-b14-text]:    https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_b14_text.npz

[v1-jax-g14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_g14_highres_vision.npz
[v1-jax-g14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_g14_highres_text.npz
[v1-jax-g14-lr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_g14_lowres_vision.npz
[v1-jax-g14-lr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_g14_lowres_text.npz
[v1-jax-so14-hr-vision]: https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_so400m14_highres_largetext_distilled_vision.npz
[v1-jax-so14-hr-text]:   https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_so400m14_highres_largetext_distilled_text.npz
[v1-jax-l14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_l14_highres_distilled_vision.npz
[v1-jax-l14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_l14_highres_distilled_text.npz
[v1-jax-b14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_b14_highres_distilled_vision.npz
[v1-jax-b14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_b14_highres_distilled_text.npz
[v1-jax-s14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_s14_highres_distilled_vision.npz
[v1-jax-s14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/scenic/tips_oss_s14_highres_distilled_text.npz

[v1-pth-g14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_g14_highres_vision.npz
[v1-pth-g14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_g14_highres_text.npz
[v1-pth-g14-lr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_g14_lowres_vision.npz
[v1-pth-g14-lr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_g14_lowres_text.npz
[v1-pth-so14-hr-vision]: https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_so400m14_highres_largetext_distilled_vision.npz
[v1-pth-so14-hr-text]:   https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_so400m14_highres_largetext_distilled_text.npz
[v1-pth-l14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_l14_highres_distilled_vision.npz
[v1-pth-l14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_l14_highres_distilled_text.npz
[v1-pth-b14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_b14_highres_distilled_vision.npz
[v1-pth-b14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_b14_highres_distilled_text.npz
[v1-pth-s14-hr-vision]:  https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_s14_highres_distilled_vision.npz
[v1-pth-s14-hr-text]:    https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/tips_oss_s14_highres_distilled_text.npz
