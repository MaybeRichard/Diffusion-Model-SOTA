# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Segmentation-guided diffusion model for medical image generation (MICCAI 2024). Trains and samples from diffusion models conditioned on multiclass segmentation masks, with optional mask-ablated training for handling incomplete masks.

## Key Commands

### Training

```bash
# Unconditional model
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset my_dataset \
    --img_dir /path/to/images \
    --train_batch_size 16 \
    --num_epochs 400

# Segmentation-guided model
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset my_dataset \
    --img_dir /path/to/images \
    --seg_dir /path/to/masks \
    --segmentation_guided \
    --num_segmentation_classes 4 \
    --train_batch_size 16 \
    --num_epochs 400

# Add --use_ablated_segmentations for mask-ablated training
```

### Sampling/Evaluation

```bash
# Sample many images
python3 main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset my_dataset \
    --segmentation_guided \
    --seg_dir /path/to/masks \
    --num_segmentation_classes 4 \
    --eval_sample_size 100

# Single batch evaluation
python3 main.py --mode eval [same args as above]
```

## Architecture

- **main.py**: Entry point, handles argument parsing, dataset loading, model initialization
- **training.py**: `TrainingConfig` dataclass and `train_loop()` function
- **eval.py**: Sampling pipelines (`SegGuidedDDPMPipeline`, `SegGuidedDDIMPipeline`) and evaluation functions
- **utils.py**: Helper utilities (image grid creation)

## Key Configurations

Training hyperparameters in `TrainingConfig` (training.py:22-57):
- `learning_rate`: 1e-4 (reduce to 2e-5 if outputs are noisy)
- `save_image_epochs`: 20
- `save_model_epochs`: 30

## Data Format

- Images: train/val/test subdirectories, PIL-readable formats or .npy for multi-channel
- Segmentation masks: `{seg_dir}/all/{train,val,test}/`, integer class values (0, 1, 2, ...)
- Mask filenames must match corresponding image filenames

## Output Structure

Models save to `{model_type}-{dataset}-{img_size}[-segguided][-ablated]/`:
- `unet/`: Model checkpoint (diffusion_pytorch_model.safetensors, config.json)
- `samples/`: Generated image grids during training
- `samples_many_{N}/`: Individual samples from eval_many mode
