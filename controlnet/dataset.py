"""
ControlNet数据集加载模块 - 用于图像-掩码对训练

支持的目录结构:
data_root/
    images/
        001.png
        002.png
        ...
    masks/
        001.png  (与images中的文件名对应)
        002.png
        ...
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np


class ControlNetDataset(Dataset):
    """
    ControlNet图像-掩码对数据集

    掩码会被用作控制条件，引导图像生成
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        images_folder: str = "images",
        masks_folder: str = "masks",
        augment: bool = True,
        mask_channels: int = 1,  # 掩码通道数，1为灰度，3为RGB
        image_prefix: str = None,  # 图像文件前缀，如 "Img_"
        mask_prefix: str = None,   # 掩码文件前缀，如 "seg_"
    ):
        """
        Args:
            data_root: 数据集根目录
            image_size: 图像大小
            images_folder: 图像文件夹名称
            masks_folder: 掩码文件夹名称
            augment: 是否进行数据增强
            mask_channels: 掩码通道数
            image_prefix: 图像文件名前缀（用于前缀不同的配对，如Img_001对应seg_001）
            mask_prefix: 掩码文件名前缀
        """
        self.data_root = data_root
        self.image_size = image_size
        self.mask_channels = mask_channels
        self.image_prefix = image_prefix
        self.mask_prefix = mask_prefix

        self.images_dir = os.path.join(data_root, images_folder)
        self.masks_dir = os.path.join(data_root, masks_folder)

        # 检查目录是否存在
        if not os.path.isdir(self.images_dir):
            raise ValueError(f"图像目录不存在: {self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise ValueError(f"掩码目录不存在: {self.masks_dir}")

        # 获取所有图像文件
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])

        # 验证掩码文件存在
        self.samples = []
        missing_masks = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            img_ext = os.path.splitext(img_file)[1]
            mask_file = None

            # 如果指定了前缀映射（如 Img_XXX -> seg_XXX）
            if image_prefix and mask_prefix and base_name.startswith(image_prefix):
                # 提取ID部分，替换前缀
                id_part = base_name[len(image_prefix):]
                mask_base = mask_prefix + id_part

                for ext in [img_ext, '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                    candidate = mask_base + ext
                    if os.path.exists(os.path.join(self.masks_dir, candidate)):
                        mask_file = candidate
                        break
            else:
                # 默认：相同文件名
                for ext in [img_ext, '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                    candidate = base_name + ext
                    if os.path.exists(os.path.join(self.masks_dir, candidate)):
                        mask_file = candidate
                        break

            if mask_file:
                self.samples.append((img_file, mask_file))
            else:
                missing_masks.append(img_file)

        if missing_masks:
            print(f"警告: {len(missing_masks)} 张图像缺少对应掩码，已跳过")
            if len(missing_masks) <= 5:
                for f in missing_masks:
                    print(f"  - {f}")

        print(f"共加载 {len(self.samples)} 对图像-掩码")

        # 设置transform
        self.augment = augment

        # 图像transform
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到[-1, 1]
        ])

        # 掩码transform
        # 注意：ControlNet期望条件图像也在[-1, 1]范围
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, mask_file = self.samples[idx]

        # 加载图像
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        # 加载掩码
        mask_path = os.path.join(self.masks_dir, mask_file)
        if self.mask_channels == 1:
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.open(mask_path).convert('RGB')

        # 数据增强（同步对图像和掩码进行翻转）
        if self.augment and torch.rand(1).item() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # 应用transform
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # 如果掩码是单通道，扩展为3通道（ControlNet需要）
        if self.mask_channels == 1:
            mask = mask.repeat(3, 1, 1)

        return {
            "pixel_values": image,      # 目标图像 [3, H, W]
            "conditioning": mask,        # 掩码条件 [3, H, W]
        }


class ControlNetDatasetWithClass(Dataset):
    """
    带类别标签的ControlNet数据集

    支持同时使用掩码控制和类别条件

    目录结构:
    data_root/
        class_name_1/
            images/
            masks/
        class_name_2/
            images/
            masks/
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        images_folder: str = "images",
        masks_folder: str = "masks",
        augment: bool = True,
        mask_channels: int = 1,
        class_names: list = None
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.mask_channels = mask_channels

        # 获取类别名称
        if class_names is None:
            self.class_names = sorted([
                d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ])
        else:
            self.class_names = class_names

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        print(f"发现 {self.num_classes} 个类别: {self.class_names}")

        # 收集所有样本
        self.samples = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

        for class_name in self.class_names:
            class_dir = os.path.join(data_root, class_name)
            images_dir = os.path.join(class_dir, images_folder)
            masks_dir = os.path.join(class_dir, masks_folder)

            if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
                print(f"  跳过 {class_name}: 缺少images或masks文件夹")
                continue

            class_idx = self.class_to_idx[class_name]
            count = 0

            for img_file in os.listdir(images_dir):
                if os.path.splitext(img_file)[1].lower() not in image_extensions:
                    continue

                base_name = os.path.splitext(img_file)[0]
                mask_file = None

                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                    candidate = base_name + ext
                    if os.path.exists(os.path.join(masks_dir, candidate)):
                        mask_file = candidate
                        break

                if mask_file:
                    self.samples.append((
                        os.path.join(images_dir, img_file),
                        os.path.join(masks_dir, mask_file),
                        class_idx
                    ))
                    count += 1

            print(f"  {class_name}: {count} 对")

        print(f"共加载 {len(self.samples)} 对图像-掩码")

        # 设置transform
        self.augment = augment

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, class_idx = self.samples[idx]

        image = Image.open(img_path).convert('RGB')

        if self.mask_channels == 1:
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.open(mask_path).convert('RGB')

        if self.augment and torch.rand(1).item() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        if self.mask_channels == 1:
            mask = mask.repeat(3, 1, 1)

        return {
            "pixel_values": image,
            "conditioning": mask,
            "class_labels": class_idx
        }


def get_dataloader(
    data_root: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    augment: bool = True,
    images_folder: str = "images",
    masks_folder: str = "masks",
    mask_channels: int = 1,
    shuffle: bool = True,
    with_class: bool = False,
    class_names: list = None,
    image_prefix: str = None,
    mask_prefix: str = None
) -> tuple:
    """
    创建数据加载器

    Args:
        with_class: 是否使用带类别的数据集
        image_prefix: 图像文件名前缀（如 "Img_"）
        mask_prefix: 掩码文件名前缀（如 "seg_"）

    Returns:
        dataloader, dataset
    """
    if with_class:
        dataset = ControlNetDatasetWithClass(
            data_root=data_root,
            image_size=image_size,
            images_folder=images_folder,
            masks_folder=masks_folder,
            augment=augment,
            mask_channels=mask_channels,
            class_names=class_names
        )
    else:
        dataset = ControlNetDataset(
            data_root=data_root,
            image_size=image_size,
            images_folder=images_folder,
            masks_folder=masks_folder,
            augment=augment,
            mask_channels=mask_channels,
            image_prefix=image_prefix,
            mask_prefix=mask_prefix
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader, dataset


if __name__ == "__main__":
    # 测试数据集加载
    print("测试基础数据集...")

    # 创建测试目录结构
    import tempfile
    import shutil

    test_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(test_dir, "images"))
    os.makedirs(os.path.join(test_dir, "masks"))

    # 创建测试图像
    for i in range(5):
        img = Image.new('RGB', (256, 256), color=(i*50, i*50, i*50))
        img.save(os.path.join(test_dir, "images", f"test_{i}.png"))

        mask = Image.new('L', (256, 256), color=i*50)
        mask.save(os.path.join(test_dir, "masks", f"test_{i}.png"))

    dataloader, dataset = get_dataloader(test_dir, batch_size=2, image_size=256)

    print(f"数据集大小: {len(dataset)}")
    print(f"批次数量: {len(dataloader)}")

    batch = next(iter(dataloader))
    print(f"图像shape: {batch['pixel_values'].shape}")
    print(f"掩码shape: {batch['conditioning'].shape}")

    # 清理
    shutil.rmtree(test_dir)
    print("\n测试完成!")
