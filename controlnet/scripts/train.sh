#!/bin/bash
# ControlNet训练脚本示例

# =====================================================
# 推荐训练流程（针对OCT等专业领域图像）：
# 1. 先训练LDM (ldm项目): 学习领域图像的生成分布
# 2. 再训练ControlNet: 加载LDM的UNet权重
# =====================================================

# 数据集路径
DATA_ROOT="/path/to/your/dataset"  # 需要包含images和masks子文件夹
OUTPUT_DIR="./controlnet_checkpoints"

# LDM checkpoint路径（来自ldm项目的训练结果）
LDM_CHECKPOINT="../ldm/ldm_checkpoints/checkpoint_best.pt"

# ============ 推荐方式：从预训练LDM加载 ============
# 如果图像和掩码文件名相同（如 001.png 对应 001.png）
python train.py \
    --data_root $DATA_ROOT \
    --images_folder images \
    --masks_folder masks \
    --output_dir $OUTPUT_DIR \
    --ldm_checkpoint $LDM_CHECKPOINT \
    --image_size 256 \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-5 \
    --freeze_unet \
    --save_every 10 \
    --sample_every 5 \
    --mixed_precision fp16

# ============ 文件名前缀不同的情况 ============
# 如果图像命名为 Img_XXX，掩码命名为 seg_XXX
# python train.py \
#     --data_root $DATA_ROOT \
#     --images_folder images \
#     --masks_folder masks \
#     --image_prefix "Img_" \
#     --mask_prefix "seg_" \
#     --ldm_checkpoint $LDM_CHECKPOINT \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --mixed_precision fp16

# ============ 多卡并行训练 ============
# accelerate launch --multi_gpu train.py \
#     --data_root $DATA_ROOT \
#     --ldm_checkpoint $LDM_CHECKPOINT \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --mixed_precision fp16

# ============ 不推荐：从头训练（仅大数据集） ============
# 如果没有预训练LDM，UNet将随机初始化
# 这需要更多数据和更长的训练时间
# python train.py \
#     --data_root $DATA_ROOT \
#     --output_dir $OUTPUT_DIR \
#     --batch_size 8 \
#     --num_epochs 500 \
#     --learning_rate 1e-4

# ============ 从checkpoint恢复训练 ============
# python train.py \
#     --data_root $DATA_ROOT \
#     --ldm_checkpoint $LDM_CHECKPOINT \
#     --output_dir $OUTPUT_DIR \
#     --resume ./controlnet_checkpoints/checkpoint_epoch_50.pt
