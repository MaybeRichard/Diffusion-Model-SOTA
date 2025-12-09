#!/bin/bash
# DiT Sampling Script

CHECKPOINT="./results/000-DiT-B-2/checkpoints/0050000.pt"
OUTPUT_DIR="./samples"
NUM_CLASSES=8

# ============ Basic Sampling ============
# Generate 8 images evenly distributed across all classes
python sample.py \
    --model DiT-B/2 \
    --ckpt $CHECKPOINT \
    --num-classes $NUM_CLASSES \
    --image-size 256 \
    --num-sampling-steps 250 \
    --cfg-scale 4.0 \
    --output-dir $OUTPUT_DIR \
    --seed 0

# ============ Generate Specific Classes ============
# Generate images for specific class labels
# python sample.py \
#     --model DiT-B/2 \
#     --ckpt $CHECKPOINT \
#     --num-classes $NUM_CLASSES \
#     --class-labels 0 1 2 3 \
#     --output-dir $OUTPUT_DIR

# ============ Generate Many Samples ============
# Generate 100 images (will cycle through classes)
# python sample.py \
#     --model DiT-B/2 \
#     --ckpt $CHECKPOINT \
#     --num-classes $NUM_CLASSES \
#     --num-samples 100 \
#     --batch-size 16 \
#     --output-dir $OUTPUT_DIR

# ============ Generate Many Samples of Specific Class ============
# Generate 50 images of class 0
# python sample.py \
#     --model DiT-B/2 \
#     --ckpt $CHECKPOINT \
#     --num-classes $NUM_CLASSES \
#     --class-labels 0 \
#     --num-samples 50 \
#     --batch-size 16 \
#     --output-dir $OUTPUT_DIR

# ============ Adjust CFG Scale ============
# Higher CFG = stronger class conditioning
# python sample.py \
#     --model DiT-B/2 \
#     --ckpt $CHECKPOINT \
#     --num-classes $NUM_CLASSES \
#     --cfg-scale 7.5 \
#     --output-dir $OUTPUT_DIR

# ============ Fewer Sampling Steps (faster) ============
# python sample.py \
#     --model DiT-B/2 \
#     --ckpt $CHECKPOINT \
#     --num-classes $NUM_CLASSES \
#     --num-sampling-steps 50 \
#     --output-dir $OUTPUT_DIR

# ============ Batch Sampling for FID Evaluation (DDP) ============
# Sample 50K images for FID evaluation using multiple GPUs
# torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py \
#     --model DiT-B/2 \
#     --ckpt $CHECKPOINT \
#     --num-classes $NUM_CLASSES \
#     --num-fid-samples 50000 \
#     --per-proc-batch-size 32 \
#     --cfg-scale 1.5

# ============ DDP Sampling with Specific Classes ============
# torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py \
#     --model DiT-B/2 \
#     --ckpt $CHECKPOINT \
#     --num-classes $NUM_CLASSES \
#     --class-labels 0 1 2 3 4 5 6 7 \
#     --num-fid-samples 10000

# ============ Sample with Pretrained ImageNet Model ============
# Auto-downloads DiT-XL/2 pretrained on ImageNet (1000 classes)
# python sample.py \
#     --model DiT-XL/2 \
#     --image-size 256 \
#     --num-classes 1000 \
#     --num-sampling-steps 250 \
#     --cfg-scale 4.0
