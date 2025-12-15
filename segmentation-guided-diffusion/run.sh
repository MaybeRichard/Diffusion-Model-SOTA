#!/bin/bash

# ============================================================
# Segmentation-Guided Diffusion Model 训练和推理脚本
# ============================================================

# ------------------------------------------------------------
# 基础配置 (根据需要修改)
# ------------------------------------------------------------
CUDA_DEVICES="0"                    # GPU设备号
IMG_SIZE=256                        # 图像尺寸
NUM_IMG_CHANNELS=1                  # 图像通道数 (1=灰度, 3=RGB)
DATASET_NAME="my_dataset"           # 数据集名称
IMG_DIR="/path/to/dataset/images"   # 图像目录
SEG_DIR="/path/to/dataset/masks"    # 掩码目录
NUM_SEG_CLASSES=4                   # 分割类别数 (包括背景0)
MODEL_TYPE="DDIM"                   # 模型类型: DDIM 或 DDPM
TRAIN_BATCH_SIZE=16                 # 训练batch size
EVAL_BATCH_SIZE=8                   # 推理batch size
NUM_EPOCHS=400                      # 训练轮数

# ============================================================
# 训练命令
# ============================================================

# --- 1. Segmentation-Guided 训练 (基础版) ---
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 main.py \
    --mode train \
    --model_type ${MODEL_TYPE} \
    --img_size ${IMG_SIZE} \
    --num_img_channels ${NUM_IMG_CHANNELS} \
    --dataset ${DATASET_NAME} \
    --img_dir ${IMG_DIR} \
    --seg_dir ${SEG_DIR} \
    --segmentation_guided \
    --num_segmentation_classes ${NUM_SEG_CLASSES} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS}

# --- 2. Segmentation-Guided 训练 (Mask-Ablated版，适合类别不完整的掩码) ---
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 main.py \
#     --mode train \
#     --model_type ${MODEL_TYPE} \
#     --img_size ${IMG_SIZE} \
#     --num_img_channels ${NUM_IMG_CHANNELS} \
#     --dataset ${DATASET_NAME} \
#     --img_dir ${IMG_DIR} \
#     --seg_dir ${SEG_DIR} \
#     --segmentation_guided \
#     --num_segmentation_classes ${NUM_SEG_CLASSES} \
#     --use_ablated_segmentations \
#     --train_batch_size ${TRAIN_BATCH_SIZE} \
#     --eval_batch_size ${EVAL_BATCH_SIZE} \
#     --num_epochs ${NUM_EPOCHS}

# --- 3. 从checkpoint恢复训练 ---
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 main.py \
#     --mode train \
#     --model_type ${MODEL_TYPE} \
#     --img_size ${IMG_SIZE} \
#     --num_img_channels ${NUM_IMG_CHANNELS} \
#     --dataset ${DATASET_NAME} \
#     --img_dir ${IMG_DIR} \
#     --seg_dir ${SEG_DIR} \
#     --segmentation_guided \
#     --num_segmentation_classes ${NUM_SEG_CLASSES} \
#     --train_batch_size ${TRAIN_BATCH_SIZE} \
#     --eval_batch_size ${EVAL_BATCH_SIZE} \
#     --num_epochs ${NUM_EPOCHS} \
#     --resume_epoch 100

# --- 4. 无条件模型训练 (不使用分割掩码) ---
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 main.py \
#     --mode train \
#     --model_type ${MODEL_TYPE} \
#     --img_size ${IMG_SIZE} \
#     --num_img_channels ${NUM_IMG_CHANNELS} \
#     --dataset ${DATASET_NAME} \
#     --img_dir ${IMG_DIR} \
#     --train_batch_size ${TRAIN_BATCH_SIZE} \
#     --eval_batch_size ${EVAL_BATCH_SIZE} \
#     --num_epochs ${NUM_EPOCHS}

# ============================================================
# 推理/采样命令
# ============================================================

# --- 5. 批量生成图像 (eval_many模式) ---
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 main.py \
#     --mode eval_many \
#     --model_type ${MODEL_TYPE} \
#     --img_size ${IMG_SIZE} \
#     --num_img_channels ${NUM_IMG_CHANNELS} \
#     --dataset ${DATASET_NAME} \
#     --seg_dir ${SEG_DIR} \
#     --segmentation_guided \
#     --num_segmentation_classes ${NUM_SEG_CLASSES} \
#     --eval_batch_size ${EVAL_BATCH_SIZE} \
#     --eval_sample_size 100

# --- 6. 批量生成图像 (Mask-Ablated模型) ---
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 main.py \
#     --mode eval_many \
#     --model_type ${MODEL_TYPE} \
#     --img_size ${IMG_SIZE} \
#     --num_img_channels ${NUM_IMG_CHANNELS} \
#     --dataset ${DATASET_NAME} \
#     --seg_dir ${SEG_DIR} \
#     --segmentation_guided \
#     --num_segmentation_classes ${NUM_SEG_CLASSES} \
#     --use_ablated_segmentations \
#     --eval_batch_size ${EVAL_BATCH_SIZE} \
#     --eval_sample_size 100

# --- 7. 单批次评估 (生成一个batch的图像网格) ---
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 main.py \
#     --mode eval \
#     --model_type ${MODEL_TYPE} \
#     --img_size ${IMG_SIZE} \
#     --num_img_channels ${NUM_IMG_CHANNELS} \
#     --dataset ${DATASET_NAME} \
#     --seg_dir ${SEG_DIR} \
#     --segmentation_guided \
#     --num_segmentation_classes ${NUM_SEG_CLASSES} \
#     --eval_batch_size ${EVAL_BATCH_SIZE}

# --- 8. 使用空白掩码采样 ---
# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 main.py \
#     --mode eval \
#     --model_type ${MODEL_TYPE} \
#     --img_size ${IMG_SIZE} \
#     --num_img_channels ${NUM_IMG_CHANNELS} \
#     --dataset ${DATASET_NAME} \
#     --seg_dir ${SEG_DIR} \
#     --segmentation_guided \
#     --num_segmentation_classes ${NUM_SEG_CLASSES} \
#     --use_ablated_segmentations \
#     --eval_blank_mask \
#     --eval_batch_size ${EVAL_BATCH_SIZE}
