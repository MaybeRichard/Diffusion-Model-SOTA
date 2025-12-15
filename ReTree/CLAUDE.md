# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReTree is a Denoising Diffusion Probabilistic Model (DDPM) implementation for retinal image generation and segmentation. The project generates synthetic retinal fundus images and corresponding segmentation masks for diabetic retinopathy lesions (MA, EX, HE, SE - Microaneurysms, Exudates, Hemorrhages, Soft Exudates).

Based on the paper: "Denoising Diffusion Probabilistic Model for Retinal Image Generation and Segmentation" (IEEE ICCP 2023).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate diffusion
```

Key dependencies: PyTorch 1.12.1, CUDA 11.3, pytorch-lightning, einops, albumentations, torchgeometry.

## Training Commands

**Mask-to-image generation with standard concatenation conditioning:**
```bash
python train_mask2img.py --mode train --data_root /path/to/dataset --device cuda:0
```

**Mask-to-image generation with SPADE conditioning:**
```bash
python train_mask2img_spade.py --mode train --data_root /path/to/dataset --device cuda:0
```

**Original training scripts (use hardcoded paths):**
- `train_ddpm_mask.py`: Train mask generation DDPM (structure → lesion masks)
- `train_ddpm_image.py`: Train image generation DDPM (masks → retinal images)
- `train_ddpm_inpaint.py`: Inpainting-style generation with masked regions
- `train_classifier.py`: Train discriminator for quality filtering

Note: Original scripts (`train_ddpm_*.py`) use hardcoded paths. Toggle `training = True/False` in `__main__` to switch between training and inference modes.

## Architecture

### DDPM_model.py - Core Model Components

- `DDPM`: Main diffusion model with U-Net architecture, transformer blocks (LSA attention), and sinusoidal time embedding. Supports conditional generation via channel concatenation of masks and optional class label embedding.
- `DDPM_seg`: SPADE-conditioned variant using `SPADEGroupNorm` for spatially-adaptive normalization. Injects condition information by modulating normalization scale/bias rather than concatenation, preserving more spatial detail.
- `Discriminator`: Transformer-augmented discriminator for adversarial training/quality filtering.
- `Down`/`Up` blocks: Use 4x pooling/upsampling with residual `DoubleConv` blocks.

### Conditioning Methods

1. **Concatenation (`DDPM`)**: Mask concatenated with noisy image as input channels
2. **SPADE (`DDPM_seg`)**: Mask modulates GroupNorm layers via learned scale/bias - better for semantic-guided generation

### utils.py - Data and Utilities

- `Diffusion`: Handles noise scheduling (linear or cosine), forward diffusion (noise addition), and reverse sampling.
- Dataset classes for various training scenarios:
  - `DDPMDataset`: Full pipeline with images + structure + 4 lesion channels
  - `DDPMDataset_sep`: Separate lesion channels with CSV class labels
  - `MaskToImageDataset`: Simple image-mask pairs (in training scripts)
- `VGGLoss`: Perceptual loss using VGG19 features.
- `save_checkpoint`/`load_checkpoint`: Model persistence.

### Data Flow

1. **Mask Generation**: Structure masks + class labels → `DDPM` → Lesion masks (MA, EX, HE, SE)
2. **Image Generation**: Structure + Lesion masks → `DDPM_seg` → Retinal fundus images

## Key Parameters

- `noise_steps`: 1000 (default diffusion timesteps)
- `image_size`: 256 (typical training resolution)
- `emb_dim/time_dim`: 256 (time embedding dimension)
- `img_channel`: Varies by task (4 for lesion masks, 3+1 for image+mask, etc.)
- `dim`: 4096 (base channel dimension, scaled by powers of 2 in encoder/decoder)

## Dataset Structure

Expected structure for `train_mask2img*.py`:
```
data_root/
    images/
        img_001.png, img_002.png, ...
    masks/
        seg_001.png, seg_002.png, ...
```

For original scripts, paths are hardcoded (e.g., `/data/xiaoyi/DR_lesions/...`) with structure:
- `image_1024_png/`: Original fundus images
- `mask_1024_png_structure/`: Vessel/structure masks
- `mask_1024_png_MA/`, `_EX/`, `_HE/`, `_SE/`: Individual lesion masks
