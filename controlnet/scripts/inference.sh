#!/bin/bash
# ControlNet推理脚本示例

CHECKPOINT="./controlnet_checkpoints/checkpoint_best.pt"
LDM_CHECKPOINT="../ldm/ldm_checkpoints/checkpoint_best.pt"  # 必须提供！
OUTPUT_DIR="./generated"

# ============ 单个掩码生成 ============
# 从单个掩码生成1张图像
python inference.py \
    --checkpoint $CHECKPOINT \
    --ldm_checkpoint $LDM_CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --mask /path/to/mask.png \
    --num_inference_steps 30 \
    --controlnet_scale 1.0 \
    --save_comparison

# 从单个掩码生成多张图像
# python inference.py \
#     --checkpoint $CHECKPOINT \
#     --ldm_checkpoint $LDM_CHECKPOINT \
#     --output_dir $OUTPUT_DIR \
#     --mask /path/to/mask.png \
#     --num_images 5 \
#     --seed 42

# ============ 批量生成 ============
# 从文件夹中所有掩码生成图像
# python inference.py \
#     --checkpoint $CHECKPOINT \
#     --ldm_checkpoint $LDM_CHECKPOINT \
#     --output_dir $OUTPUT_DIR \
#     --mask_folder /path/to/masks/ \
#     --num_images 1 \
#     --save_comparison

# ============ 调整ControlNet强度 ============
# controlnet_scale: 控制掩码对生成的影响程度
# 0.0: 完全忽略掩码
# 1.0: 正常强度（默认）
# 1.5-2.0: 更严格遵循掩码结构

# python inference.py \
#     --checkpoint $CHECKPOINT \
#     --ldm_checkpoint $LDM_CHECKPOINT \
#     --mask /path/to/mask.png \
#     --controlnet_scale 1.5

# ============ RGB掩码 ============
# 如果掩码是RGB格式（如语义分割的彩色掩码）
# python inference.py \
#     --checkpoint $CHECKPOINT \
#     --ldm_checkpoint $LDM_CHECKPOINT \
#     --mask /path/to/mask.png \
#     --mask_channels 3
