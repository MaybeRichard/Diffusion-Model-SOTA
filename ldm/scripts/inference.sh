#!/bin/bash
# LDM推理脚本示例

# 使用最优模型生成所有类别的图像
python inference.py \
    --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
    --output_dir ./generated \
    --num_samples 10 \
    --num_inference_steps 50 \
    --cfg_scale 3.0 \
    --save_individual \
    --save_grid

# 生成指定类别的图像 (例如: AMD)
# python inference.py \
#     --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
#     --output_dir ./generated \
#     --class_label AMD \
#     --num_samples 20 \
#     --cfg_scale 4.0 \
#     --seed 42

# 生成健康类别
# python inference.py \
#     --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
#     --class_label 健康 \
#     --num_samples 50 \
#     --cfg_scale 3.0

# 使用不同的CFG强度（较高的值=更强的类别特征）
# python inference.py \
#     --checkpoint ./ldm_checkpoints/checkpoint_best.pt \
#     --class_label DR \
#     --num_samples 10 \
#     --cfg_scale 7.5
