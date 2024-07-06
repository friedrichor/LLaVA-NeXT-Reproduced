# LLaVA-NeXT-Reproduced

English | [简体中文](README_zh-CN.md)

---

This repository **(mainly for personal learning)** is intended to reproduce the training code and scripts of LLaVA-NeXT, since as of today (2024.07.06), there are not yet official LLaVA-NeXT training code and scripts publicly available.

This code is based on the currently released code of [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) (including the LLaVA-NeXT model and associated data processing functions), and refer to the training code and scripts of the previous version ([LLaVA-v1.5](https://github.com/haotian-liu/LLaVA)), aiming to reproduce the training of LLaVA-NeXT with minimal changes.

> In this repository, we copy the training code from [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) directly and modify it. Specifically, only `llava/train/train.py` and `llava/train/llava_trainer.py` have been modified, and the changes are annotated so that the readers can know the difference between this code and the code of LLaVA-v1.5.  
> For training scripts, they are listed in `scripts/next`. Currently, only the training scripts for vicuna-v1.5-7b (as the LLM backbone) are available, and we will update scripts for other models in the future.

## ⏳ Models & Scripts

### 🔧 Installation

#### 1. **Clone this repository and navigate to the LLaVA-NeXT folder:**
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
```

#### 2. **Install the training package:**
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## 📁 Data Preparation

You can follow [Data.md](docs/Data.md) to access the dataset.

## 🚆 Training

### Hyperparameters

We use a similar set of hyperparameters as Vicuna in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Projector lr  | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-NeXT-Vicuna-7B | 256 | 1e-3 | 1 | 4096 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | LLM lr |  Projector lr |  Vision Tower lr | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA-NeXT-Vicuna-7B | 128 | 2e-5 | 2e-5 | 2e-6 | 1 | 4096 | 0 |

### Pretrain (feature alignment)

```bash
bash scripts/next/vicuna-7b/pretrain.sh 
```

Pretrain takes around 4 hours for LLaVA-NeXT-Vicuna-7B on 8x H100 (80G)

### Visual Instruction Tuning

```bash
bash scripts/next/vicuna-7b/finetune.sh
```

Visual instruction tuning (use llava_v1_5_mix665k) takes around 11.5 hours for LLaVA-NeXT-Vicuna-7B on 8x H100 (80G)

### Convert Model Weightss

If you want to evaluate model performance via [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit), you will also need to convert the model weights to the hf version. Because the model weights saved after training can only be loaded by this code, not via the transformers library.

```bash
bash convert_llava_next_weights_to_hf.sh
```

## 🙏 Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT)
