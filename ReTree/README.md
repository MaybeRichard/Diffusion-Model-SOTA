# ReTree: Denoising Diffusion Probabilistic Model for Retinal Image Generation

基于DDPM的视网膜图像生成与分割模型，支持两种条件注入方式：**Concatenation** 和 **SPADE**。

> 原始论文: [Denoising Diffusion Probabilistic Model for Retinal Image Generation and Segmentation](https://ieeexplore.ieee.org/document/10233905) (IEEE ICCP 2023)

## 模型架构

- **DDPM (Concatenation)**: 将mask与噪声图像在通道维度拼接作为输入
- **DDPM_seg (SPADE)**: 通过空间自适应归一化(SPADEGroupNorm)注入条件信息，保留更多空间细节

![Architecture](ddpms.jpg)

## 环境配置

```bash
conda env create -f environment.yml
conda activate diffusion
```

## 数据集结构

```
data_root/
    images/
        img_001.png, img_002.png, ...
    masks/
        seg_001.png, seg_002.png, ...
```

原始数据集: [ReTree Dataset (Kaggle)](https://www.kaggle.com/datasets/alnuritoalimanov/retree-dataset)

## 快速开始

### 使用脚本运行

```bash
# 训练 Concatenation 版本
bash run.sh train_concat

# 训练 SPADE 版本
bash run.sh train_spade

# 训练两个版本
bash run.sh train_all

# 使用训练好的模型采样
bash run.sh sample_concat
bash run.sh sample_spade

# 训练并采样所有版本
bash run.sh all
```

### 直接使用Python

**训练 Concatenation 版本:**
```bash
python train_mask2img.py \
    --mode train \
    --data_root /path/to/dataset \
    --image_size 256 \
    --batch_size 4 \
    --epochs 500 \
    --lr 3e-4 \
    --output_dir ./outputs/mask2img \
    --device cuda:0
```

**训练 SPADE 版本:**
```bash
python train_mask2img_spade.py \
    --mode train \
    --data_root /path/to/dataset \
    --image_size 256 \
    --batch_size 4 \
    --epochs 500 \
    --lr 3e-4 \
    --output_dir ./outputs/mask2img_spade \
    --device cuda:0
```

**生成图像:**
```bash
python train_mask2img.py \
    --mode generate \
    --data_root /path/to/dataset \
    --checkpoint ./outputs/mask2img/checkpoints/best_model.pth.tar \
    --output_dir ./outputs/generated \
    --device cuda:0
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image_size` | 256 | 图像分辨率 |
| `--batch_size` | 4 | 批次大小 |
| `--epochs` | 500 | 训练轮数 |
| `--lr` | 3e-4 | 学习率 |
| `--noise_steps` | 1000 | 扩散步数 |
| `--emb_dim` | 256 | 时间嵌入维度 |
| `--sample_interval` | 500 | 每N个batch生成样本 |

## Citation

```bibtex
@inproceedings{alimanov2023denoising,
  title={Denoising Diffusion Probabilistic Model for Retinal Image Generation and Segmentation},
  author={Alimanov, Alnur and Islam, Md Baharul},
  booktitle={2023 IEEE International Conference on Computational Photography (ICCP)},
  pages={1--12},
  year={2023},
  organization={IEEE}
}
```
