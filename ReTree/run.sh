#===============================================================================
# Mask-guided DDPM 训练与采样命令
# 复制需要的命令直接运行即可
#===============================================================================

#===============================================================================
# Concatenation 版本
#===============================================================================

# --- 训练 Concat ---
python train_mask2img.py \
    --mode train \
    --data_root /data2/sichengli/Data/test/Despeckle/RETOUCH-3 \
    --image_size 256 \
    --batch_size 4 \
    --epochs 500 \
    --lr 3e-4 \
    --output_dir ./outputs/mask2img \
    --device cuda:0

# --- 采样 Concat ---
python train_mask2img.py \
    --mode generate \
    --data_root /data2/sichengli/Data/test/Despeckle/RETOUCH-3 \
    --checkpoint ./outputs/mask2img/checkpoints/best_model.pth.tar \
    --output_dir ./outputs/mask2img/generated \
    --device cuda:0

#===============================================================================
# SPADE 版本
#===============================================================================

# --- 训练 SPADE ---
python train_mask2img_spade.py \
    --mode train \
    --data_root /data2/sichengli/Data/test/Despeckle/RETOUCH-3 \
    --image_size 256 \
    --batch_size 4 \
    --epochs 500 \
    --lr 3e-4 \
    --output_dir ./outputs/mask2img_spade \
    --device cuda:0

# --- 采样 SPADE ---
python train_mask2img_spade.py \
    --mode generate \
    --data_root /data2/sichengli/Data/test/Despeckle/RETOUCH-3 \
    --checkpoint ./outputs/mask2img_spade/checkpoints/best_model.pth.tar \
    --output_dir ./outputs/mask2img_spade/generated \
    --device cuda:0
