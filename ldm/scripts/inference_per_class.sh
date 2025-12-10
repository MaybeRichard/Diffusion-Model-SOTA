#!/bin/bash

# 按类别生成图像的推理脚本
# 用法: bash inference_per_class.sh [GPU_ID] [TOTAL_SAMPLES]
# 示例: bash inference_per_class.sh 0 8000
# bash ./scripts/inference_per_class.sh "4,5,6,7" 8000
# 默认参数
# 获取第1个参数，如果未提供则默认为 0
GPU_ID=${1:-0}
TOTAL_SAMPLES=${2:-8000}
CHECKPOINT="./ldm_checkpoints/checkpoint_best.pt"
OUTPUT_DIR="./generated_per_class"
NUM_INFERENCE_STEPS=50
CFG_SCALE=3.0
BATCH_SIZE=16

echo "========================================"
echo "按类别生成图像"
echo "========================================"
echo "GPU ID: ${GPU_ID}"
echo "总样本数: ${TOTAL_SAMPLES}"
echo "Checkpoint: ${CHECKPOINT}"
echo "输出目录: ${OUTPUT_DIR}"
echo "采样步数: ${NUM_INFERENCE_STEPS}"
echo "CFG强度: ${CFG_SCALE}"
echo "批大小: ${BATCH_SIZE}"
echo "========================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_per_class.py \
    --checkpoint ${CHECKPOINT} \
    --output_dir ${OUTPUT_DIR} \
    --total_samples ${TOTAL_SAMPLES} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --cfg_scale ${CFG_SCALE} \
    --batch_size ${BATCH_SIZE}

echo "完成!"
