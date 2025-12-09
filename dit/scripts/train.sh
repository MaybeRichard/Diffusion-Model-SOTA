#!/bin/bash
# DiT Training Script
# DiT uses torchrun for distributed training

DATA_PATH="/path/to/imagenet/train"  # ImageNet format: class folders
RESULTS_DIR="./results"

# ============ Single Node Multi-GPU Training ============
# Train DiT-XL/2 with N GPUs
# Replace N with your number of GPUs
torchrun --nnodes=1 --nproc_per_node=N train.py \
    --model DiT-XL/2 \
    --data-path $DATA_PATH \
    --results-dir $RESULTS_DIR \
    --image-size 256 \
    --global-batch-size 256 \
    --epochs 1400

# ============ Available Models ============
# DiT-XL/2  (largest, best quality)
# DiT-XL/4
# DiT-XL/8
# DiT-L/2
# DiT-L/4
# DiT-L/8
# DiT-B/2
# DiT-B/4
# DiT-B/8
# DiT-S/2
# DiT-S/4
# DiT-S/8

# ============ Training DiT-B/4 (smaller model) ============
# torchrun --nnodes=1 --nproc_per_node=4 train.py \
#     --model DiT-B/4 \
#     --data-path $DATA_PATH \
#     --image-size 256 \
#     --global-batch-size 128

# ============ Training with 512x512 resolution ============
# torchrun --nnodes=1 --nproc_per_node=N train.py \
#     --model DiT-XL/2 \
#     --data-path $DATA_PATH \
#     --image-size 512 \
#     --global-batch-size 128

# ============ Custom num_classes ============
# For custom datasets with different number of classes
# torchrun --nnodes=1 --nproc_per_node=N train.py \
#     --model DiT-XL/2 \
#     --data-path $DATA_PATH \
#     --num-classes 8 \
#     --image-size 256 \
#     --global-batch-size 64
