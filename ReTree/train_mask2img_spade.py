"""
Mask-guided Image Generation using DDPM with SPADE conditioning
SPADE: Spatially-Adaptive Denormalization，通过空间自适应归一化注入条件信息

用法: python train_mask2img_spade.py --data_root /path/to/dataset --device cuda:0

与Concatenation版本的区别:
- Concatenation: 将mask与噪声图像在通道维度拼接作为输入
- SPADE: mask通过调制归一化层的scale和bias参数来注入条件信息，保留更多空间细节

数据集结构:
    data_root/
        images/
            img_001.png
            ...
        masks/
            seg_001.png
            ...
"""

import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from torch import optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import random

from utils import save_images, save_checkpoint, load_checkpoint
from DDPM_model import DDPM_seg

torch.manual_seed(42)


class MaskToImageDataset(Dataset):
    """
    加载 image 和对应的 mask 用于训练
    image: img_XXX.png (RGB)
    mask: seg_XXX.png (单通道灰度图)
    """
    def __init__(self, image_dir, mask_dir, img_size=256, augment=True):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.*")))
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        # 图像变换 (RGB)
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size),
                                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Mask变换 (灰度图)
        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size),
                                          interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 加载图像
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')

        # 根据图像文件名找到对应的mask
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace('img_', 'seg_')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 如果命名规则不匹配，尝试直接用相同文件名
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, img_name)

        mask = Image.open(mask_path).convert('L')

        # 应用变换
        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        # 数据增强
        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        return img, mask


class Diffusion:
    """扩散过程管理"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # 使用cosine schedule
        self.beta = self._cosine_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def _cosine_schedule(self):
        """Cosine noise schedule"""
        return torch.abs(
            torch.cos(torch.linspace(0, torch.pi / 2, self.noise_steps)) * self.beta_end
            - (self.beta_end - self.beta_start)
        )

    def noise_images(self, x, t):
        """前向扩散: 给图像添加噪声"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        """随机采样时间步"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    @torch.no_grad()
    def sample(self, model, mask, n=1):
        """
        从mask生成图像 (反向扩散) - SPADE版本
        Args:
            model: DDPM_seg模型
            mask: 条件mask [B, 1, H, W]
            n: 生成数量
        """
        model.eval()
        mask = mask[:n]

        # 从纯噪声开始
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

        for i in tqdm(reversed(range(1, self.noise_steps)), desc="Sampling", leave=False):
            t = (torch.ones(n) * i).long().to(self.device)

            # SPADE: mask作为structure条件传入
            predicted_noise = model(x=x, t=t, structure=mask)

            # 计算系数
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            # 添加噪声 (最后一步不加)
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # 反向扩散一步
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        model.train()

        # 转换到 [0, 255]
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    device = args.device

    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    # 数据集
    image_dir = os.path.join(args.data_root, "images")
    mask_dir = os.path.join(args.data_root, "masks")

    dataset = MaskToImageDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_size=args.image_size,
        augment=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # SPADE模型: DDPM_seg
    # img_channels = 3(RGB) + 1(mask) = 4 (forward时会自动concat)
    # con_channel = 1 (单通道mask用于SPADE条件注入)
    model = DDPM_seg(
        img_channels=4,      # 输入: RGB + mask (forward时会concat)
        time_dim=args.emb_dim,
        out_channel=3,       # 输出: RGB
        con_channel=1,       # SPADE条件通道数 (单通道mask)
        device=device
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 加载checkpoint
    if args.resume:
        load_checkpoint(args.resume, model, optimizer, args.lr, device)
        print(f"Resumed from: {args.resume}")

    # 损失函数
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # 扩散过程
    diffusion = Diffusion(
        noise_steps=args.noise_steps,
        img_size=args.image_size,
        device=device
    )

    # 训练循环
    best_loss = float("inf")
    sample_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0

        for i, (images, masks) in enumerate(pbar):
            images = images.to(device)  # [B, 3, H, W]
            masks = masks.to(device)    # [B, 1, H, W]

            # 采样时间步
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            # 前向扩散: 给图像添加噪声
            x_t, noise = diffusion.noise_images(images, t)

            # SPADE: mask作为structure条件
            # DDPM_seg的forward会自动concat x和structure
            predicted_noise = model(x=x_t, t=t, structure=masks)

            # 计算损失
            loss = mse_loss(noise, predicted_noise) + l1_loss(noise, predicted_noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), avg_loss=epoch_loss/(i+1))

            # 定期生成样本
            if i % args.sample_interval == 0 and i > 0:
                sample_imgs = diffusion.sample(model, masks, n=min(4, masks.shape[0]))
                mask_vis = ((masks.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

                save_images(sample_imgs,
                           os.path.join(args.output_dir, "samples", f"ep{epoch}_s{sample_counter}_gen.png"))
                save_images(mask_vis,
                           os.path.join(args.output_dir, "samples", f"ep{epoch}_s{sample_counter}_mask.png"))
                sample_counter += 1

        # 只保存最佳模型
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer,
                os.path.join(args.output_dir, "checkpoints", "best_model.pth.tar")
            )
            print(f"Saved best model at epoch {epoch} with loss {best_loss:.6f}")


@torch.no_grad()
def generate(args):
    """从mask生成图像 - SPADE版本"""
    device = args.device

    # 加载SPADE模型
    model = DDPM_seg(
        img_channels=4,
        time_dim=args.emb_dim,
        out_channel=3,
        con_channel=1,
        device=device
    ).to(device)

    load_checkpoint(args.checkpoint, model, None, None, device)
    model.eval()

    diffusion = Diffusion(
        noise_steps=args.noise_steps,
        img_size=args.image_size,
        device=device
    )

    # 加载mask
    mask_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.image_size, args.image_size),
                                      interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    mask_paths = sorted(glob(os.path.join(args.mask_dir, "*.*")))

    for idx, mask_path in enumerate(tqdm(mask_paths, desc="Generating")):
        mask = Image.open(mask_path).convert('L')
        mask = mask_transform(mask).unsqueeze(0).to(device)

        # 生成图像
        generated = diffusion.sample(model, mask, n=1)

        # 保存
        name = os.path.splitext(os.path.basename(mask_path))[0]
        save_images(generated, os.path.join(args.output_dir, f"{name}_generated.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mask-guided DDPM with SPADE conditioning")

    # 模式选择
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                        help='train: 训练模型, generate: 生成图像')

    # 数据相关
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录 (包含 images/ 和 masks/ 子目录)')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='生成模式: mask目录路径')

    # 模型相关
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--emb_dim', type=int, default=256, help='时间嵌入维度')
    parser.add_argument('--noise_steps', type=int, default=1000, help='扩散步数')

    # 训练相关
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    # 保存相关
    parser.add_argument('--output_dir', type=str, default='./outputs/mask2img_spade',
                        help='输出目录')
    parser.add_argument('--sample_interval', type=int, default=500, help='每N个batch生成样本')

    # 恢复训练/生成
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='生成模式: 模型checkpoint路径')

    # 设备
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        if args.checkpoint is None:
            raise ValueError("生成模式需要指定 --checkpoint")
        if args.mask_dir is None:
            args.mask_dir = os.path.join(args.data_root, "masks")
        generate(args)


'''
# 训练SPADE版本
python train_mask2img_spade.py \
      --mode train \
      --data_root /home/richard/Documents/dataset/DUKE \
      --image_size 256 \
      --batch_size 4 \
      --epochs 500 \
      --lr 3e-4 \
      --output_dir ./outputs/mask2img_spade \
      --device cuda:0
'''
