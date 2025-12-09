#!/bin/bash
# LDM训练脚本示例

DATA_ROOT="/home/richard/Documents/Code/dataset/dataset_8/train"
OUTPUT_DIR="./ldm_checkpoints"

# ============ 单卡训练 ============
# python train.py \
#     --data_root $DATA_ROOT \
#     --output_dir $OUTPUT_DIR \
#     --image_size 256 \
#     --batch_size 8 \
#     --num_epochs 200 \
#     --learning_rate 1e-4 \
#     --cfg_dropout 0.1 \
#     --save_every 10 \
#     --sample_every 5 \
#     --mixed_precision fp16

# ============ 多卡并行训练 ============
# 使用所有可用GPU
accelerate launch --multi_gpu train.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --batch_size 8 \
    --num_epochs 200 \
    --mixed_precision fp16

# 指定2张GPU
# accelerate launch --multi_gpu --num_processes=2 train.py \
#     --data_root $DATA_ROOT \
#     --output_dir $OUTPUT_DIR \
#     --batch_size 8 \
#     --num_epochs 200 \
#     --mixed_precision fp16

# 指定特定GPU (例如GPU 0和1)
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 train.py \
#     --data_root $DATA_ROOT \
#     --output_dir $OUTPUT_DIR \
#     --batch_size 8 \
#     --num_epochs 200 \
#     --mixed_precision fp16

# 如果显存不足，可以减小batch_size并增加gradient_accumulation_steps
# python train.py \
#     --data_root ../output_dir \
#     --output_dir ./ldm_checkpoints \
#     --batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --mixed_precision fp16

# 从checkpoint恢复训练
# python train.py \
#     --data_root ../output_dir \
#     --output_dir ./ldm_checkpoints \
#     --resume ./ldm_checkpoints/checkpoint_epoch_50.pt
