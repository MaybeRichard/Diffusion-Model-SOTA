#!/bin/bash
# DiT Sampling Script

# ============ Sample with Pre-trained Model ============
# Auto-downloads DiT-XL/2 pretrained on ImageNet
python sample.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --num-sampling-steps 250 \
    --cfg-scale 4.0 \
    --seed 0

# ============ Sample with 512x512 ============
# python sample.py \
#     --model DiT-XL/2 \
#     --image-size 512 \
#     --num-sampling-steps 250 \
#     --cfg-scale 4.0

# ============ Sample with Custom Checkpoint ============
# Use your own trained model
# python sample.py \
#     --model DiT-XL/2 \
#     --image-size 256 \
#     --ckpt ./results/000-DiT-XL-2/checkpoints/0050000.pt \
#     --cfg-scale 4.0

# ============ Sample with Different CFG Scale ============
# Higher CFG = stronger class conditioning
# python sample.py \
#     --model DiT-XL/2 \
#     --image-size 256 \
#     --cfg-scale 7.5

# ============ Fewer Sampling Steps (faster) ============
# python sample.py \
#     --model DiT-XL/2 \
#     --image-size 256 \
#     --num-sampling-steps 50

# ============ Batch Sampling for Evaluation (DDP) ============
# Sample 50K images for FID evaluation
# torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py \
#     --model DiT-XL/2 \
#     --num-fid-samples 50000 \
#     --image-size 256

# ============ Custom Classes (for custom trained model) ============
# python sample.py \
#     --model DiT-XL/2 \
#     --ckpt /path/to/checkpoint.pt \
#     --num-classes 8 \
#     --image-size 256
