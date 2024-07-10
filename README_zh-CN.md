# LLaVA-NeXT-Reproduced

[English](README.md) | 简体中文

---

由于截止到目前为止 (2024.07.06) 官方还没有公开 LLaVA-NeXT 的训练代码和脚本，这个仓库旨在复现 LLaVA-NeXT 的训练代码和脚本 (主要用于个人学习)。

我们引入 [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) 目前提供的代码 (包括LLaVA-NeXT模型和相关的数据处理函数)， 并参考之前版本 ([LLaVA-v1.5](https://github.com/haotian-liu/LLaVA)) 以及的训练代码和脚本，旨在以最小的改动来复现 LLaVA-NeXT 的训练。

> 在这个仓库中，我们直接将 [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) 的训练代码复制过来并进行更改。具体来说，只有 `llava/train/train.py` 和 `llava/train/llava_trainer.py` 被更改，且更改的部分均有注释，以便读者了解这个代码与LLaVA-v1.5的代码的区别。  
> 
> 对于训练脚本，我们添加的脚本在 `scripts/next`，目前提供了 vicuna-v1.5-7b 和 llama3-8b 为 LLM 基座的训练脚本，未来我们将补充其它模型的脚本。

## ⏳ ToDo

- [X] Reproduce LLaVA-Next-Vicuna-7B
- [X] Reproduce LLaVA-Next-LLaMA3-8B
- [ ] Reproduce LLaVA-Next-Qwen2-7B
- [ ] Support for SigLIP as the vision tower

## 🔧 安装

### 1. **克隆仓库并进入 LLaVA-NeXT 文件夹:**
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
```

### 2. **安装环境:**
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## 📁 数据准备

您可以按照 [Data.md](docs/Data.md) 获取数据集。

## 🚆 训练

### 超参数

在微调过程中，我们使用了与 Vicuna 类似的一组超参数。 下面提供了预训练和微调中使用的两个超参数。

1. Pretraining

| Hyperparameter       | Global Batch Size | Projector lr | Epochs | Max length | Weight decay |
| -------------------- | ----------------: | -----------: | -----: | ---------: | -----------: |
| LLaVA-NeXT-Vicuna-7B |               256 |         1e-3 |      1 |       4096 |            0 |

2. Finetuning


| Hyperparameter       | Global Batch Size | LLM lr | Projector lr | Vision Tower lr | Epochs | Max length | Weight decay |
| -------------------- | ----------------: | -----: | -----------: | --------------: | -----: | ---------: | -----------: |
| LLaVA-NeXT-Vicuna-7B |               128 |   2e-5 |         2e-5 |            2e-6 |      1 |       4096 |            0 |

### 预训练 (特征对齐)

```bash
bash scripts/next/vicuna-7b/pretrain.sh 
```

| Model                | Data           | Hardware      | Training Time  |
| -------------------- | -------------: | ------------: | -------------: |
| LLaVA-NeXT-Vicuna-7B | LLaVA-Pretrain | 8x H100 (80G) | ~ 4   hours    |
| LLaVA-NeXT-LLaMA3-8B | LLaVA-Pretrain | 8x H100 (80G) | ~ 4.5 hours    |

### 视觉指令微调

```bash
bash scripts/next/vicuna-7b/finetune.sh
```

| Model                | Data                  | Hardware      | Training Time  |
| -------------------- | --------------------: | ------------: | -------------: |
| LLaVA-NeXT-Vicuna-7B | LLaVA-v1.5-mix665K    | 8x H100 (80G) | ~ 11.5 hours   |
| LLaVA-NeXT-Vicuna-7B | Open-LLaVA-NeXT-mix1M | 8x H100 (80G) | ~ 17.5 hours   |
| LLaVA-NeXT-LLaMA3-8B | Open-LLaVA-NeXT-mix1M | 8x H100 (80G) | ~ 24.5 hours   |

### 转换模型权重

如果你想要通过 🤗 Transformers 库来加载这个权重 (如更方便地生成回复或通过 [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 来测试模型)，你需要转换模型权重为 hf 版本。因为训练后保存的模型权重只能由当前这个仓库的代码加载，而不能通过 🤗 Transformers 库来加载。

```bash
bash convert_llava_next_weights_to_hf.sh
```

## 🙏 致谢

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT)
