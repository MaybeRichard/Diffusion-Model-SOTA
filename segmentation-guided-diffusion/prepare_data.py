"""
数据准备脚本：将原始数据整理为训练所需的目录结构

原始结构:
    image/img_XXX.png
    mask/seg_XXX.png

目标结构:
    dataset/images/{train,val}/XXX.png
    dataset/masks/all/{train,val}/XXX.png
"""
import os
import shutil

# ============ 配置这里 ============
src_img_dir = "/path/to/image"      # 原始图像目录
src_mask_dir = "/path/to/mask"      # 原始掩码目录
dst_base = "/path/to/dataset"       # 目标目录
val_count = 8                       # val集数量（仅用于训练时可视化）
img_prefix = "img_"                 # 图像文件名前缀
mask_prefix = "seg_"                # 掩码文件名前缀
# =================================

# 创建目录
for split in ["train", "val"]:
    os.makedirs(os.path.join(dst_base, "images", split), exist_ok=True)
    os.makedirs(os.path.join(dst_base, "masks", "all", split), exist_ok=True)

# 获取所有图像ID
img_files = sorted(os.listdir(src_img_dir))
ids = [f.replace(img_prefix, "").replace(".png", "") for f in img_files if f.startswith(img_prefix)]

print(f"找到 {len(ids)} 个图像")

# 复制文件
for i, id_ in enumerate(ids):
    split = "val" if i < val_count else "train"

    # 图像
    src_img = os.path.join(src_img_dir, f"{img_prefix}{id_}.png")
    dst_img = os.path.join(dst_base, "images", split, f"{id_}.png")
    shutil.copy(src_img, dst_img)

    # 掩码
    src_mask = os.path.join(src_mask_dir, f"{mask_prefix}{id_}.png")
    dst_mask = os.path.join(dst_base, "masks", "all", split, f"{id_}.png")
    shutil.copy(src_mask, dst_mask)

print(f"完成! train: {len(ids)-val_count}, val: {val_count}")
print(f"目标目录: {dst_base}")
