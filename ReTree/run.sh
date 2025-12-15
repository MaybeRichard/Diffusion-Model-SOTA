#!/bin/bash
#===============================================================================
# Mask-guided DDPM 训练与采样脚本
# 包含两种条件注入方式: Concatenation 和 SPADE
#===============================================================================

# 配置参数 (根据需要修改)
DATA_ROOT="/home/richard/Documents/dataset/DUKE"
IMAGE_SIZE=256
BATCH_SIZE=4
EPOCHS=500
LR=3e-4
DEVICE="cuda:0"
NUM_WORKERS=4
SAMPLE_INTERVAL=500

#===============================================================================
# Concatenation 版本
#===============================================================================
CONCAT_OUTPUT="./outputs/mask2img"
CONCAT_CHECKPOINT="${CONCAT_OUTPUT}/checkpoints/best_model.pth.tar"

# 训练 Concatenation 版本
train_concat() {
    echo "=========================================="
    echo "Training Concatenation version..."
    echo "=========================================="
    python train_mask2img.py \
        --mode train \
        --data_root ${DATA_ROOT} \
        --image_size ${IMAGE_SIZE} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --num_workers ${NUM_WORKERS} \
        --sample_interval ${SAMPLE_INTERVAL} \
        --output_dir ${CONCAT_OUTPUT} \
        --device ${DEVICE}
}

# 采样 Concatenation 版本
sample_concat() {
    echo "=========================================="
    echo "Sampling with Concatenation version..."
    echo "=========================================="
    python train_mask2img.py \
        --mode generate \
        --data_root ${DATA_ROOT} \
        --image_size ${IMAGE_SIZE} \
        --checkpoint ${CONCAT_CHECKPOINT} \
        --output_dir ${CONCAT_OUTPUT}/generated \
        --device ${DEVICE}
}

#===============================================================================
# SPADE 版本
#===============================================================================
SPADE_OUTPUT="./outputs/mask2img_spade"
SPADE_CHECKPOINT="${SPADE_OUTPUT}/checkpoints/best_model.pth.tar"

# 训练 SPADE 版本
train_spade() {
    echo "=========================================="
    echo "Training SPADE version..."
    echo "=========================================="
    python train_mask2img_spade.py \
        --mode train \
        --data_root ${DATA_ROOT} \
        --image_size ${IMAGE_SIZE} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --num_workers ${NUM_WORKERS} \
        --sample_interval ${SAMPLE_INTERVAL} \
        --output_dir ${SPADE_OUTPUT} \
        --device ${DEVICE}
}

# 采样 SPADE 版本
sample_spade() {
    echo "=========================================="
    echo "Sampling with SPADE version..."
    echo "=========================================="
    python train_mask2img_spade.py \
        --mode generate \
        --data_root ${DATA_ROOT} \
        --image_size ${IMAGE_SIZE} \
        --checkpoint ${SPADE_CHECKPOINT} \
        --output_dir ${SPADE_OUTPUT}/generated \
        --device ${DEVICE}
}

#===============================================================================
# 主函数
#===============================================================================
usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  train_concat    训练 Concatenation 版本"
    echo "  train_spade     训练 SPADE 版本"
    echo "  train_all       训练两个版本 (顺序执行)"
    echo "  sample_concat   使用 Concatenation 模型采样"
    echo "  sample_spade    使用 SPADE 模型采样"
    echo "  sample_all      使用两个模型采样"
    echo "  all             训练并采样所有版本"
    echo ""
    echo "Examples:"
    echo "  $0 train_concat"
    echo "  $0 sample_spade"
    echo "  $0 all"
}

case "$1" in
    train_concat)
        train_concat
        ;;
    train_spade)
        train_spade
        ;;
    train_all)
        train_concat
        train_spade
        ;;
    sample_concat)
        sample_concat
        ;;
    sample_spade)
        sample_spade
        ;;
    sample_all)
        sample_concat
        sample_spade
        ;;
    all)
        train_concat
        sample_concat
        train_spade
        sample_spade
        ;;
    *)
        usage
        exit 1
        ;;
esac

echo "Done!"
