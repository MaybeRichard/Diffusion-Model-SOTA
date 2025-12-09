"""
OCT图像数据集加载模块 - 用于类别条件LDM训练
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class OCTDataset(Dataset):
    """
    OCT图像数据集，支持类别条件训练

    目录结构要求:
    data_root/
        class_name_1/
            image1.png
            image2.png
        class_name_2/
            ...
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        augment: bool = True,
        class_names: list = None
    ):
        """
        Args:
            data_root: 数据集根目录
            image_size: 图像大小
            augment: 是否进行数据增强
            class_names: 类别名称列表，如果为None则自动从目录获取
        """
        self.data_root = data_root
        self.image_size = image_size

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

        # 收集所有图像路径和标签
        self.samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_idx = self.class_to_idx[class_name]
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        class_idx
                    ))

        print(f"共加载 {len(self.samples)} 张图像")

        # 统计各类别数量
        class_counts = {}
        for _, label in self.samples:
            class_name = self.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for name, count in class_counts.items():
            print(f"  {name}: {count} 张")

        # 设置transform
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 加载图像
        image = Image.open(img_path)

        # 转换为RGB（处理灰度图）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 应用transform
        image = self.transform(image)

        return {
            "pixel_values": image,
            "class_labels": label
        }


def get_dataloader(
    data_root: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    augment: bool = True,
    class_names: list = None,
    shuffle: bool = True
) -> tuple:
    """
    创建数据加载器

    Returns:
        dataloader, dataset
    """
    dataset = OCTDataset(
        data_root=data_root,
        image_size=image_size,
        augment=augment,
        class_names=class_names
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
    data_root = "./output_dir"
    dataloader, dataset = get_dataloader(data_root, batch_size=4, image_size=256)

    print(f"\n数据集大小: {len(dataset)}")
    print(f"批次数量: {len(dataloader)}")
    print(f"类别映射: {dataset.class_to_idx}")

    # 测试一个batch
    batch = next(iter(dataloader))
    print(f"\n图像shape: {batch['pixel_values'].shape}")
    print(f"标签shape: {batch['class_labels'].shape}")
    print(f"标签值: {batch['class_labels']}")
