#!/bin/bash
# DiT Training Script
# DiT uses torchrun for distributed training

DATA_PATH="/path/to/dataset/train"  # ImageNet format: class folders
RESULTS_DIR="./results"

# ============ Single Node Multi-GPU Training ============
# Train DiT-B/2 with N GPUs (default for custom datasets)
# Replace N with your number of GPUs
torchrun --nnodes=1 --nproc_per_node=N train.py \
    --model DiT-B/2 \
    --data-path $DATA_PATH \
    --results-dir $RESULTS_DIR \
    --image-size 256 \
    --num-classes 8 \
    --global-batch-size 256 \
    --epochs 1400

# ============ Available Models ============
# DiT-XL/2  (largest, best quality, ~675M params)
# DiT-XL/4
# DiT-XL/8
# DiT-L/2   (~458M params)
# DiT-L/4
# DiT-L/8
# DiT-B/2   (~130M params, default for custom datasets)
# DiT-B/4
# DiT-B/8
# DiT-S/2   (~33M params, fastest)
# DiT-S/4
# DiT-S/8

# ============ Training DiT-XL/2 (large model) ============
# torchrun --nnodes=1 --nproc_per_node=8 train.py \
#     --model DiT-XL/2 \
#     --data-path $DATA_PATH \
#     --image-size 256 \
#     --num-classes 8 \
#     --global-batch-size 256

# ============ Training with 512x512 resolution ============
# torchrun --nnodes=1 --nproc_per_node=N train.py \
#     --model DiT-B/2 \
#     --data-path $DATA_PATH \
#     --image-size 512 \
#     --num-classes 8 \
#     --global-batch-size 128

# ============ Smaller batch size (limited GPU memory) ============
# torchrun --nnodes=1 --nproc_per_node=2 train.py \
#     --model DiT-B/2 \
#     --data-path $DATA_PATH \
#     --num-classes 8 \
#     --global-batch-size 64

# ============ More frequent checkpoints ============
# torchrun --nnodes=1 --nproc_per_node=N train.py \
#     --model DiT-B/2 \
#     --data-path $DATA_PATH \
#     --num-classes 8 \
#     --ckpt-every 10000 \
#     --log-every 50
