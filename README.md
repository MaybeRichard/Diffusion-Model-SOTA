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
| **DDPM** | Denoising Diffusion Probabilistic Model | Basic diffusion model (pixel-space) |
| **LDM** | Class-Conditional Latent Diffusion Model | Generate images by class label |
| **ControlNet** | Mask-Guided Image Generation | Generate images following mask structure |
| **DiT** | Diffusion Transformer (Meta) | SOTA class-conditional generation with Transformer backbone |

### Requirements

```bash
pip install torch torchvision diffusers accelerate tqdm tensorboard pillow
```

Or install from requirements.txt in each model directory.

---

### DDPM (Denoising Diffusion Probabilistic Model)

Classic pixel-space diffusion model. Includes both unconditional and class-conditional variants.

#### Architecture
- **UNet**: Custom UNet with self-attention layers
- **Time Embedding**: Sinusoidal positional encoding
- **Class Embedding**: Learned embeddings (conditional version)
- **EMA**: Exponential moving average for stable generation

#### Variants
- `ddpm.py`: Unconditional DDPM
- `ddpm_conditional.py`: Class-conditional DDPM with CFG

#### Data Structure
ImageNet-style folder structure:
```
data_root/
    class_0/
        image1.png
        image2.png
    class_1/
        image1.png
        image2.png
```

#### Training

```python
# Unconditional DDPM - modify ddpm.py launch() function:
args.run_name = "DDPM_Uncondtional"
args.epochs = 500
args.batch_size = 12
args.image_size = 64
args.dataset_path = "/path/to/dataset"
args.device = "cuda"
args.lr = 3e-4

# Then run:
python ddpm.py

# Conditional DDPM - modify ddpm_conditional.py launch() function:
args.run_name = "DDPM_conditional"
args.epochs = 300
args.batch_size = 14
args.image_size = 64
args.num_classes = 10
args.dataset_path = "/path/to/dataset"

# Then run:
python ddpm_conditional.py
```

```bash
# Multi-GPU DDP training
torchrun --nnodes=1 --nproc_per_node=4 ddpm_conditional_ddp.py \
    --mode train_ddp \
    --dataset-path /path/to/dataset \
    --num-classes 8 \
    --image-size 256 \
    --batch-size 8 \
    --epochs 100

# Resume training
torchrun --nnodes=1 --nproc_per_node=4 ddpm_conditional_ddp.py \
    --mode train_ddp \
    --dataset-path /path/to/dataset \
    --resume
```

#### Inference

```python
import torch
from modules import UNet_conditional, count_parameters
from ddpm_conditional import Diffusion

device = "cuda"
img_size = 64
num_classes = 10

# Load model
model = UNet_conditional(num_classes=num_classes, img_size=img_size).to(device)
ckpt = torch.load("./models/DDPM_conditional/ckpt.pt", weights_only=False)
model.load_state_dict(ckpt)

# Sample
diffusion = Diffusion(img_size=img_size, device=device)
labels = torch.Tensor([0, 1, 2, 3]).long().to(device)
samples = diffusion.sample(model, n=4, labels=labels, cfg_scale=3, log_time=True)
```

```bash
# Multi-GPU DDP sampling for FID evaluation
torchrun --nnodes=1 --nproc_per_node=4 ddpm_conditional_ddp.py \
    --mode sample_ddp \
    --ckpt ./models/DDPM_conditional/ckpt.pt \
    --use-ema \
    --num-samples 8000 \
    --num-classes 8 \
    --batch-size 8 \
    --output-dir ./samples

# Single-GPU sampling
python ddpm_conditional_ddp.py \
    --mode sample \
    --ckpt ./models/DDPM_conditional/ema_ckpt.pt \
    --num-samples 100 \
    --class-labels 0 1 2 3 \
    --cfg-scale 3
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `noise_steps` | Number of diffusion steps | 1000 |
| `cfg_scale` | CFG strength (conditional only) | 3 |
| `image_size` | Image resolution | 64 |
| `--use-ema` | Use EMA model for sampling | False |
| `--resume` | Resume training from checkpoint | False |

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

# Batch generation per class (for FID evaluation)
python inference_per_class.py \
    --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
    --output_dir ./generated_per_class \
    --total_samples 8000 \
    --batch_size 16 \
    --cfg_scale 3.0
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

### DiT (Diffusion Transformer)

**From Meta AI (facebookresearch/DiT)**

State-of-the-art diffusion model using Transformer architecture instead of U-Net. Achieves FID 2.27 on ImageNet 256x256.

#### Architecture
- **VAE**: Pretrained `stabilityai/sd-vae-ft-ema` (frozen)
- **DiT**: Transformer-based denoiser (trainable)
- **Patchify**: Converts latent to patch tokens
- Supports multiple model sizes: XL, L, B, S with patch sizes 2, 4, 8

#### Available Models

| Model | Params | FID-50K | Recommended Use |
|-------|--------|---------|-----------------|
| DiT-XL/2 | 675M | 2.27 | Large datasets (ImageNet) |
| DiT-L/2 | 458M | - | Medium datasets |
| DiT-B/2 | 130M | 68.4 | **Custom datasets (default)** |
| DiT-S/2 | 33M | - | Fast experiments |

#### Data Structure
ImageNet-style folder structure:
```
data_root/
    class_0/
        image1.png
        image2.png
    class_1/
        image1.png
        image2.png
```

#### Training

```bash
cd dit

# Multi-GPU training with torchrun (required)
# DiT-B/2 is recommended for custom datasets
torchrun --nnodes=1 --nproc_per_node=4 train.py \
    --model DiT-B/2 \
    --data-path /path/to/data \
    --results-dir ./results \
    --image-size 256 \
    --num-classes 8 \
    --global-batch-size 256 \
    --epochs 1400

# Large model for bigger datasets
torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --model DiT-XL/2 \
    --data-path /path/to/data \
    --num-classes 8 \
    --global-batch-size 256
```

#### Inference

```bash
cd dit

# Basic sampling (generates 8 images across all classes)
python sample.py \
    --model DiT-B/2 \
    --ckpt ./results/000-DiT-B-2/checkpoints/0050000.pt \
    --num-classes 8 \
    --output-dir ./samples

# Generate specific classes
python sample.py \
    --model DiT-B/2 \
    --ckpt ./results/000-DiT-B-2/checkpoints/0050000.pt \
    --num-classes 8 \
    --class-labels 0 1 2 3 \
    --output-dir ./samples

# Generate many samples of one class
python sample.py \
    --model DiT-B/2 \
    --ckpt ./results/000-DiT-B-2/checkpoints/0050000.pt \
    --num-classes 8 \
    --class-labels 0 \
    --num-samples 50 \
    --batch-size 16 \
    --output-dir ./samples

# Batch sampling for FID evaluation (DDP)
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py \
    --model DiT-B/2 \
    --ckpt ./results/000-DiT-B-2/checkpoints/0050000.pt \
    --num-classes 8 \
    --num-fid-samples 50000 \
    --cfg-scale 1.5
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model variant (DiT-XL/2, DiT-B/2, etc.) | DiT-B/2 |
| `--image-size` | Image resolution (256 or 512) | 256 |
| `--num-classes` | Number of classes | 8 |
| `--cfg-scale` | CFG strength | 4.0 |
| `--num-sampling-steps` | DDPM sampling steps | 250 |
| `--class-labels` | Specific classes to generate (e.g., 0 1 2) | All |
| `--num-samples` | Total images to generate | 8 |
| `--batch-size` | Sampling batch size (prevents OOM) | 8 |
| `--output-dir` | Output directory for samples | samples |

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

### Evaluation Metrics

We provide a unified evaluation script to compute standard generative model metrics.

#### Supported Metrics

| Metric | Model | Description |
|--------|-------|-------------|
| **FID** | InceptionV3 | Fréchet Inception Distance - measures distribution similarity |
| **Precision** | VGG16 | Fraction of generated images that fall within real data manifold |
| **Recall** | VGG16 | Fraction of real images covered by generated data manifold |

#### Usage

```python
# Modify paths in evaluation/precision_recall.py:
REAL_IMAGES_DIR = "/path/to/real/images"
GEN_IMAGES_DIR = "/path/to/generated/images"
BATCH_SIZE = 32

# Run evaluation
cd evaluation
python precision_recall.py
```

#### Output Example

```
========================================
FINAL EVALUATION REPORT
Real Data: real_images
Gen Data:  generated_images
========================================
FID (InceptionV3):     12.3456
Precision (VGG16):     0.8521
Recall (VGG16):        0.7834
========================================
```

#### Interpretation

- **FID**: Lower is better. Measures overall quality and diversity.
  - < 10: Excellent
  - 10-50: Good
  - > 50: Poor

- **Precision**: Higher is better (0-1). High precision = generated images look realistic.

- **Recall**: Higher is better (0-1). High recall = generated images cover the diversity of real data.

---

<a name="chinese"></a>
## 中文

本仓库包含用于2D图像生成的SOTA扩散模型实现，专为医学影像等领域特定应用设计。

### 包含的模型

| 模型 | 描述 | 用途 |
|------|------|------|
| **DDPM** | 去噪扩散概率模型 | 基础扩散模型（像素空间） |
| **LDM** | 类别条件潜在扩散模型 | 根据类别标签生成图像 |
| **ControlNet** | 掩码引导图像生成 | 生成遵循掩码结构的图像 |
| **DiT** | Diffusion Transformer (Meta) | 使用Transformer架构的SOTA类别条件生成 |

### 环境要求

```bash
pip install torch torchvision diffusers accelerate tqdm tensorboard pillow
```

或从各模型目录下的 requirements.txt 安装。

---

### DDPM（去噪扩散概率模型）

经典的像素空间扩散模型。包含无条件和类别条件两种变体。

#### 架构
- **UNet**: 自定义UNet，带自注意力层
- **时间嵌入**: 正弦位置编码
- **类别嵌入**: 可学习嵌入（条件版本）
- **EMA**: 指数移动平均，用于稳定生成

#### 变体
- `ddpm.py`: 无条件DDPM
- `ddpm_conditional.py`: 类别条件DDPM，支持CFG

#### 数据结构
ImageNet风格的文件夹结构：
```
data_root/
    class_0/
        image1.png
        image2.png
    class_1/
        image1.png
        image2.png
```

#### 训练

```python
# 无条件DDPM - 修改 ddpm.py 中的 launch() 函数：
args.run_name = "DDPM_Uncondtional"
args.epochs = 500
args.batch_size = 12
args.image_size = 64
args.dataset_path = "/path/to/dataset"
args.device = "cuda"
args.lr = 3e-4

# 然后运行：
python ddpm.py

# 条件DDPM - 修改 ddpm_conditional.py 中的 launch() 函数：
args.run_name = "DDPM_conditional"
args.epochs = 300
args.batch_size = 14
args.image_size = 64
args.num_classes = 10
args.dataset_path = "/path/to/dataset"

# 然后运行：
python ddpm_conditional.py
```

```bash
# 多卡DDP训练
torchrun --nnodes=1 --nproc_per_node=4 ddpm_conditional_ddp.py \
    --mode train_ddp \
    --dataset-path /path/to/dataset \
    --num-classes 8 \
    --image-size 256 \
    --batch-size 8 \
    --epochs 100

# 断点续训
torchrun --nnodes=1 --nproc_per_node=4 ddpm_conditional_ddp.py \
    --mode train_ddp \
    --dataset-path /path/to/dataset \
    --resume
```

#### 推理

```python
import torch
from modules import UNet_conditional, count_parameters
from ddpm_conditional import Diffusion

device = "cuda"
img_size = 64
num_classes = 10

# 加载模型
model = UNet_conditional(num_classes=num_classes, img_size=img_size).to(device)
ckpt = torch.load("./models/DDPM_conditional/ckpt.pt", weights_only=False)
model.load_state_dict(ckpt)

# 采样
diffusion = Diffusion(img_size=img_size, device=device)
labels = torch.Tensor([0, 1, 2, 3]).long().to(device)
samples = diffusion.sample(model, n=4, labels=labels, cfg_scale=3, log_time=True)
```

```bash
# 多卡DDP采样（用于FID评估）
torchrun --nnodes=1 --nproc_per_node=4 ddpm_conditional_ddp.py \
    --mode sample_ddp \
    --ckpt ./models/DDPM_conditional/ckpt.pt \
    --use-ema \
    --num-samples 8000 \
    --num-classes 8 \
    --batch-size 8 \
    --output-dir ./samples

# 单卡采样
python ddpm_conditional_ddp.py \
    --mode sample \
    --ckpt ./models/DDPM_conditional/ema_ckpt.pt \
    --num-samples 100 \
    --class-labels 0 1 2 3 \
    --cfg-scale 3
```

#### 关键参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `noise_steps` | 扩散步数 | 1000 |
| `cfg_scale` | CFG强度（仅条件版本） | 3 |
| `image_size` | 图像分辨率 | 64 |
| `--use-ema` | 使用EMA模型采样 | False |
| `--resume` | 从checkpoint恢复训练 | False |

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

# 按类别批量生成（用于FID评估）
python inference_per_class.py \
    --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
    --output_dir ./generated_per_class \
    --total_samples 8000 \
    --batch_size 16 \
    --cfg_scale 3.0
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

### DiT（Diffusion Transformer）

**来自 Meta AI (facebookresearch/DiT)**

使用Transformer架构替代U-Net的SOTA扩散模型。在ImageNet 256x256上达到FID 2.27。

#### 架构
- **VAE**: 预训练 `stabilityai/sd-vae-ft-ema`（冻结）
- **DiT**: 基于Transformer的去噪器（可训练）
- **Patchify**: 将latent转换为patch tokens
- 支持多种模型大小：XL、L、B、S，patch大小可选2、4、8

#### 可用模型

| 模型 | 参数量 | FID-50K | 推荐用途 |
|------|--------|---------|----------|
| DiT-XL/2 | 675M | 2.27 | 大数据集（ImageNet） |
| DiT-L/2 | 458M | - | 中等数据集 |
| DiT-B/2 | 130M | 68.4 | **自定义数据集（默认）** |
| DiT-S/2 | 33M | - | 快速实验 |

#### 数据结构
ImageNet风格的文件夹结构：
```
data_root/
    class_0/
        image1.png
        image2.png
    class_1/
        image1.png
        image2.png
```

#### 训练

```bash
cd dit

# 多卡训练（使用torchrun，必须）
# DiT-B/2 推荐用于自定义数据集
torchrun --nnodes=1 --nproc_per_node=4 train.py \
    --model DiT-B/2 \
    --data-path /path/to/data \
    --results-dir ./results \
    --image-size 256 \
    --num-classes 8 \
    --global-batch-size 256 \
    --epochs 1400

# 大模型用于更大数据集
torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --model DiT-XL/2 \
    --data-path /path/to/data \
    --num-classes 8 \
    --global-batch-size 256
```

#### 推理

```bash
cd dit

# 基础采样（生成8张图像，分布在所有类别）
python sample.py \
    --model DiT-B/2 \
    --ckpt ./results/000-DiT-B-2/checkpoints/0050000.pt \
    --num-classes 8 \
    --output-dir ./samples

# 生成指定类别
python sample.py \
    --model DiT-B/2 \
    --ckpt ./results/000-DiT-B-2/checkpoints/0050000.pt \
    --num-classes 8 \
    --class-labels 0 1 2 3 \
    --output-dir ./samples

# 生成单个类别的多张图像
python sample.py \
    --model DiT-B/2 \
    --ckpt ./results/000-DiT-B-2/checkpoints/0050000.pt \
    --num-classes 8 \
    --class-labels 0 \
    --num-samples 50 \
    --batch-size 16 \
    --output-dir ./samples

# 批量采样用于FID评估（DDP）
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py \
    --model DiT-B/2 \
    --ckpt ./results/000-DiT-B-2/checkpoints/0050000.pt \
    --num-classes 8 \
    --num-fid-samples 50000 \
    --cfg-scale 1.5
```

#### 关键参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model` | 模型变体（DiT-XL/2、DiT-B/2等） | DiT-B/2 |
| `--image-size` | 图像分辨率（256或512） | 256 |
| `--num-classes` | 类别数量 | 8 |
| `--cfg-scale` | CFG强度 | 4.0 |
| `--num-sampling-steps` | DDPM采样步数 | 250 |
| `--class-labels` | 指定生成的类别（如 0 1 2） | 全部 |
| `--num-samples` | 生成图像总数 | 8 |
| `--batch-size` | 采样批次大小（防止OOM） | 8 |
| `--output-dir` | 输出目录 | samples |

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

### 评估指标

我们提供统一的评估脚本来计算标准的生成模型指标。

#### 支持的指标

| 指标 | 模型 | 描述 |
|------|------|------|
| **FID** | InceptionV3 | Fréchet Inception Distance - 衡量分布相似性 |
| **Precision** | VGG16 | 生成图像落入真实数据流形的比例 |
| **Recall** | VGG16 | 真实图像被生成数据流形覆盖的比例 |

#### 使用方法

```python
# 修改 evaluation/precision_recall.py 中的路径：
REAL_IMAGES_DIR = "/path/to/real/images"
GEN_IMAGES_DIR = "/path/to/generated/images"
BATCH_SIZE = 32

# 运行评估
cd evaluation
python precision_recall.py
```

#### 输出示例

```
========================================
FINAL EVALUATION REPORT
Real Data: real_images
Gen Data:  generated_images
========================================
FID (InceptionV3):     12.3456
Precision (VGG16):     0.8521
Recall (VGG16):        0.7834
========================================
```

#### 指标解读

- **FID**: 越低越好。衡量整体质量和多样性。
  - < 10: 优秀
  - 10-50: 良好
  - > 50: 较差

- **Precision**: 越高越好（0-1）。高精度 = 生成图像看起来真实。

- **Recall**: 越高越好（0-1）。高召回 = 生成图像覆盖真实数据的多样性。

---

### Project Structure / 项目结构

```
Diffusion-Model-SOTA/
├── README.md
├── .gitignore
├── evaluation/        # Evaluation metrics
│   ├── precision_recall.py  # FID, Precision, Recall
│   ├── requirements.txt
│   └── evaluate.sh
├── ddpm/              # Pixel-space DDPM
│   ├── ddpm.py        # Unconditional DDPM
│   ├── ddpm_conditional.py  # Class-conditional DDPM
│   ├── ddpm_conditional_ddp.py  # Multi-GPU DDP training
│   ├── modules.py     # UNet architectures
│   ├── utils.py       # Utilities
│   ├── LICENSE
│   └── scripts/
│       ├── train.sh
│       └── sample.sh
├── ldm/
│   ├── dataset.py      # Dataset loading
│   ├── models.py       # Model definitions
│   ├── train.py        # Training script
│   ├── inference.py    # Inference script
│   ├── inference_per_class.py  # Batch inference per class
│   ├── requirements.txt
│   └── scripts/
│       ├── train.sh
│       ├── inference.sh
│       └── inference_per_class.sh
├── controlnet/
│   ├── dataset.py      # Image-mask pair loading
│   ├── models.py       # ControlNet model
│   ├── train.py        # Training script
│   ├── inference.py    # Inference script
│   ├── requirements.txt
│   └── scripts/
│       ├── train.sh
│       └── inference.sh
└── dit/                # From facebookresearch/DiT
    ├── models.py       # DiT model definitions
    ├── train.py        # DDP training script
    ├── sample.py       # Single-GPU sampling
    ├── sample_ddp.py   # Multi-GPU sampling
    ├── download.py     # Pretrained weights download
    ├── diffusion/      # Diffusion utilities
    │   ├── __init__.py
    │   ├── gaussian_diffusion.py
    │   ├── diffusion_utils.py
    │   ├── respace.py
    │   └── timestep_sampler.py
    ├── environment.yml
    ├── LICENSE.txt     # CC-BY-NC License
    └── scripts/
        ├── train.sh
        └── sample.sh
```

---

### License

MIT License

### Citation

If you use this code in your research, please cite the original papers:

```bibtex
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={NeurIPS},
  year={2020}
}

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

@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```
