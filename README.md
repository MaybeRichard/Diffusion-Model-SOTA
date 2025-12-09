# Diffusion-Model-SOTA

SOTA Diffusion Models for 2D Image Generation / 2D图像生成的SOTA扩散模型

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

This repository contains implementations of state-of-the-art diffusion models for 2D image generation, specifically designed for medical imaging and other domain-specific applications.

### Models Included

| Model | Description | Use Case |
|-------|-------------|----------|
| **LDM** | Class-Conditional Latent Diffusion Model | Generate images by class label |
| **ControlNet** | Mask-Guided Image Generation | Generate images following mask structure |

### Requirements

```bash
pip install torch torchvision diffusers accelerate tqdm tensorboard pillow
```

Or install from requirements.txt in each model directory.

---

### LDM (Latent Diffusion Model)

Class-conditional image generation using latent diffusion with Classifier-Free Guidance (CFG).

#### Architecture
- **VAE**: Pretrained `stabilityai/sd-vae-ft-mse` (frozen)
- **UNet**: Diffusers UNet2DConditionModel (trainable)
- **ClassEmbedder**: Learnable class embeddings with CFG dropout

#### Data Structure
```
data_root/
    class_name_1/
        image1.png
        image2.png
    class_name_2/
        image1.png
        image2.png
```

#### Training

```bash
cd ldm

# Single GPU
python train.py \
    --data_root /path/to/data \
    --output_dir ./ldm_checkpoints \
    --image_size 256 \
    --batch_size 8 \
    --num_epochs 200 \
    --learning_rate 1e-4 \
    --cfg_dropout 0.1 \
    --mixed_precision fp16

# Multi-GPU (using accelerate)
accelerate launch --multi_gpu train.py \
    --data_root /path/to/data \
    --batch_size 8 \
    --num_epochs 200 \
    --mixed_precision fp16
```

#### Inference

```bash
cd ldm

# Generate all classes
python inference.py \
    --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
    --output_dir ./generated \
    --num_samples 10 \
    --cfg_scale 3.0 \
    --save_grid

# Generate specific class
python inference.py \
    --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
    --class_label "class_name" \
    --num_samples 20 \
    --cfg_scale 4.0 \
    --seed 42
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--cfg_scale` | CFG strength (higher = stronger class features) | 3.0 |
| `--num_inference_steps` | Sampling steps | 50 |
| `--cfg_dropout` | Dropout probability for CFG training | 0.1 |
| `--batch_size` | Inference batch size (prevents OOM) | 16 |

---

### ControlNet

Mask-guided image generation. Generates images that follow the structure defined by input masks.

#### Recommended Training Pipeline

1. **First**: Train LDM to learn the domain distribution
2. **Then**: Train ControlNet with pretrained LDM UNet weights

This two-stage approach is crucial for domain-specific applications (e.g., medical imaging).

#### Architecture
- **VAE**: Pretrained `stabilityai/sd-vae-ft-mse` (frozen)
- **UNet**: Loaded from LDM checkpoint (frozen by default)
- **ControlNet**: Initialized from UNet encoder (trainable)
- **DummyEncoder**: Simple embedding for unconditional generation

#### Data Structure
```
data_root/
    images/
        001.png
        002.png
    masks/
        001.png  # Corresponding mask
        002.png
```

Or with different naming prefixes:
```
data_root/
    images/
        Img_001.png
        Img_002.png
    masks/
        seg_001.png  # Prefix mapping: Img_XXX -> seg_XXX
        seg_002.png
```

#### Training

```bash
cd controlnet

# With pretrained LDM (recommended)
python train.py \
    --data_root /path/to/data \
    --images_folder images \
    --masks_folder masks \
    --ldm_checkpoint ../ldm/ldm_checkpoints/checkpoint_best.pt \
    --output_dir ./controlnet_checkpoints \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-5 \
    --freeze_unet \
    --mixed_precision fp16

# With different naming prefixes
python train.py \
    --data_root /path/to/data \
    --image_prefix "Img_" \
    --mask_prefix "seg_" \
    --ldm_checkpoint ../ldm/ldm_checkpoints/checkpoint_best.pt \
    --batch_size 8 \
    --num_epochs 100
```

#### Inference

```bash
cd controlnet

# Single mask
python inference.py \
    --checkpoint ./controlnet_checkpoints/checkpoint_best.pt \
    --ldm_checkpoint ../ldm/ldm_checkpoints/checkpoint_best.pt \
    --mask /path/to/mask.png \
    --output_dir ./generated \
    --controlnet_scale 1.0 \
    --save_comparison

# Batch generation from folder
python inference.py \
    --checkpoint ./controlnet_checkpoints/checkpoint_best.pt \
    --ldm_checkpoint ../ldm/ldm_checkpoints/checkpoint_best.pt \
    --mask_folder /path/to/masks/ \
    --num_images 1 \
    --save_comparison
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--controlnet_scale` | Mask influence (0=ignore, 1=normal, 1.5+=strict) | 1.0 |
| `--ldm_checkpoint` | **Required**: LDM checkpoint for UNet weights | - |
| `--freeze_unet` | Freeze UNet during training | True |
| `--mask_channels` | Mask channels (1=grayscale, 3=RGB) | 1 |

---

### Output Information

Both models provide detailed output during training and inference:

**Training Output**:
- Model parameter summary (total, trainable, frozen)
- Per-component breakdown (VAE, UNet, ControlNet, etc.)

**Inference Output**:
- Timing statistics
- Samples generated, inference steps, total time
- Average time per image and per step

---

<a name="chinese"></a>
## 中文

本仓库包含用于2D图像生成的SOTA扩散模型实现，专为医学影像等领域特定应用设计。

### 包含的模型

| 模型 | 描述 | 用途 |
|------|------|------|
| **LDM** | 类别条件潜在扩散模型 | 根据类别标签生成图像 |
| **ControlNet** | 掩码引导图像生成 | 生成遵循掩码结构的图像 |

### 环境要求

```bash
pip install torch torchvision diffusers accelerate tqdm tensorboard pillow
```

或从各模型目录下的 requirements.txt 安装。

---

### LDM（潜在扩散模型）

使用潜在扩散和分类器自由引导（CFG）进行类别条件图像生成。

#### 架构
- **VAE**: 预训练 `stabilityai/sd-vae-ft-mse`（冻结）
- **UNet**: Diffusers UNet2DConditionModel（可训练）
- **ClassEmbedder**: 可学习的类别嵌入，支持CFG dropout

#### 数据结构
```
data_root/
    类别名称1/
        image1.png
        image2.png
    类别名称2/
        image1.png
        image2.png
```

#### 训练

```bash
cd ldm

# 单卡训练
python train.py \
    --data_root /path/to/data \
    --output_dir ./ldm_checkpoints \
    --image_size 256 \
    --batch_size 8 \
    --num_epochs 200 \
    --learning_rate 1e-4 \
    --cfg_dropout 0.1 \
    --mixed_precision fp16

# 多卡并行训练（使用accelerate）
accelerate launch --multi_gpu train.py \
    --data_root /path/to/data \
    --batch_size 8 \
    --num_epochs 200 \
    --mixed_precision fp16
```

#### 推理

```bash
cd ldm

# 生成所有类别
python inference.py \
    --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
    --output_dir ./generated \
    --num_samples 10 \
    --cfg_scale 3.0 \
    --save_grid

# 生成指定类别
python inference.py \
    --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
    --class_label "类别名称" \
    --num_samples 20 \
    --cfg_scale 4.0 \
    --seed 42
```

#### 关键参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--cfg_scale` | CFG强度（越高类别特征越强） | 3.0 |
| `--num_inference_steps` | 采样步数 | 50 |
| `--cfg_dropout` | CFG训练时的dropout概率 | 0.1 |
| `--batch_size` | 推理批次大小（防止显存不足） | 16 |

---

### ControlNet

掩码引导图像生成。生成遵循输入掩码定义结构的图像。

#### 推荐训练流程

1. **首先**：训练LDM学习领域数据分布
2. **然后**：使用预训练的LDM UNet权重训练ControlNet

这种两阶段方法对于领域特定应用（如医学影像）至关重要。

#### 架构
- **VAE**: 预训练 `stabilityai/sd-vae-ft-mse`（冻结）
- **UNet**: 从LDM checkpoint加载（默认冻结）
- **ControlNet**: 从UNet编码器初始化（可训练）
- **DummyEncoder**: 用于无条件生成的简单嵌入

#### 数据结构
```
data_root/
    images/
        001.png
        002.png
    masks/
        001.png  # 对应的掩码
        002.png
```

或使用不同的命名前缀：
```
data_root/
    images/
        Img_001.png
        Img_002.png
    masks/
        seg_001.png  # 前缀映射: Img_XXX -> seg_XXX
        seg_002.png
```

#### 训练

```bash
cd controlnet

# 使用预训练LDM（推荐）
python train.py \
    --data_root /path/to/data \
    --images_folder images \
    --masks_folder masks \
    --ldm_checkpoint ../ldm/ldm_checkpoints/checkpoint_best.pt \
    --output_dir ./controlnet_checkpoints \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-5 \
    --freeze_unet \
    --mixed_precision fp16

# 使用不同的命名前缀
python train.py \
    --data_root /path/to/data \
    --image_prefix "Img_" \
    --mask_prefix "seg_" \
    --ldm_checkpoint ../ldm/ldm_checkpoints/checkpoint_best.pt \
    --batch_size 8 \
    --num_epochs 100
```

#### 推理

```bash
cd controlnet

# 单个掩码
python inference.py \
    --checkpoint ./controlnet_checkpoints/checkpoint_best.pt \
    --ldm_checkpoint ../ldm/ldm_checkpoints/checkpoint_best.pt \
    --mask /path/to/mask.png \
    --output_dir ./generated \
    --controlnet_scale 1.0 \
    --save_comparison

# 从文件夹批量生成
python inference.py \
    --checkpoint ./controlnet_checkpoints/checkpoint_best.pt \
    --ldm_checkpoint ../ldm/ldm_checkpoints/checkpoint_best.pt \
    --mask_folder /path/to/masks/ \
    --num_images 1 \
    --save_comparison
```

#### 关键参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--controlnet_scale` | 掩码影响程度（0=忽略，1=正常，1.5+=严格） | 1.0 |
| `--ldm_checkpoint` | **必须**：LDM checkpoint用于加载UNet权重 | - |
| `--freeze_unet` | 训练时冻结UNet | True |
| `--mask_channels` | 掩码通道数（1=灰度，3=RGB） | 1 |

---

### 输出信息

两个模型在训练和推理时都提供详细输出：

**训练输出**：
- 模型参数量统计（总计、可训练、冻结）
- 各组件分解（VAE、UNet、ControlNet等）

**推理输出**：
- 时间统计
- 生成样本数、推理步数、总耗时
- 平均每张图像时间和每步时间

---

### Project Structure / 项目结构

```
Diffusion-Model-SOTA/
├── README.md
├── .gitignore
├── ldm/
│   ├── dataset.py      # Dataset loading
│   ├── models.py       # Model definitions
│   ├── train.py        # Training script
│   ├── inference.py    # Inference script
│   ├── requirements.txt
│   └── scripts/
│       ├── train.sh
│       └── inference.sh
└── controlnet/
    ├── dataset.py      # Image-mask pair loading
    ├── models.py       # ControlNet model
    ├── train.py        # Training script
    ├── inference.py    # Inference script
    ├── requirements.txt
    └── scripts/
        ├── train.sh
        └── inference.sh
```

---

### License

MIT License

### Citation

If you use this code in your research, please cite the original papers:

```bibtex
@article{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  journal={CVPR},
  year={2022}
}

@article{zhang2023adding,
  title={Adding conditional control to text-to-image diffusion models},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  journal={ICCV},
  year={2023}
}
```
