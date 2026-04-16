## MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model (CVPR 2026)
[![arXiv](https://img.shields.io/badge/arXiv%20papr-2602.06393-b31b1b.svg)](https://arxiv.org/abs/2602.06393)

Welcom to the official Pytorch implementation of MuCo!

**Authors**:

**[Geonmo Gu](https://geonm.github.io/)<sup>1,3</sup>, [Byeongho Heo](https://sites.google.com/view/byeongho-heo/home)<sup>1</sup>, [Jaemyung Yu](https://sites.google.com/view/jaemyungyu)<sup>1</sup>, [Jaehui Hwang](https://j-h-hwang.github.io/)<sup>1</sup>, [Taekyung Kim](https://scholar.google.co.kr/citations?user=u-9bdkwAAAAJ&hl=en)<sup>1</sup>, [Sangmin Lee](https://sites.google.com/view/pixel-lab-ai)<sup>3</sup>, HeeJae Jun<sup>2</sup>, Yoohoon Kang<sup>2</sup>, [Sangdoo Yun](https://sangdooyun.github.io/)\*<sup>1</sup>, [Dongyoon Han](https://sites.google.com/site/dyhan0920/)\*<sup>1</sup>**

<sup>1</sup> NAVER AI Lab <sup>2</sup> NAVER AI Search Platform <sup>3</sup> Korea University

\* Corresponding authors.


## 🚀 News
- **April 16, 2026** - Models and evaluation code are released!
- **February 21, 2026** - MuCo is accepted to CVPR 2026!

## 📚 M3T dataset
[[🤗 naver-ai/M3T]](https://huggingface.co/datasets/naver-ai/M3T)

## 📂 MuCo Models

| Model | Avg. MMEB |
| :--- | :---: |
| [[🤗 MuCo-2B]](https://huggingface.co/naver-ai/MuCo-2B) | 70.1 |
| [[🤗 MuCo-7B]](https://huggingface.co/naver-ai/MuCo-7B) | 74.2 |

> **Note:** Performance has been further optimized during the code release preparation. 😊

---


## 🛠️ Installation

```bash
$ pip install -r requirements.txt
```

## 🗂️ Dataset Preparation
```bash
$ DATASET_FOLDER="./dataset" # Set your path here
$ hf download naver-ai/M3T --repo-type dataset --local-dir ${DATASET_FOLDER}
$ cd ${DATASET_FOLDER}
$ python download_M3T_images.py
$ python download_and_unzip_MMEB_images.py
```

## 🔥 Training

Scripts will be updated soon.

## 💯 Evaluation

```bash
torchrun --nproc_per_node=8 --master_port=10000 eval_mmeb.py \
    --pooling eos \
    --normalize \
    --per_device_eval_batch_size 64 \
    --model_name naver-ai/MuCo-2B \
    --data_basedir ${DATASET_FOLDER}/MMEB_eval \
    --encode_output_path ./results/MuCo-2B_MMEB
```
