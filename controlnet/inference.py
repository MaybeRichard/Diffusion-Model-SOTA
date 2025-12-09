"""
ControlNet推理脚本

功能：
- 加载训练好的模型
- 使用掩码引导生成图像
- 支持单张掩码或批量掩码
- 支持控制强度调节
"""

import os
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from diffusers import AutoencoderKL

from models import create_unet, create_controlnet_from_unet, DummyEncoder, create_inference_scheduler


def print_timing_stats(stats, title="采样时间统计"):
    """打印采样时间统计"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"  生成样本数:     {stats['num_samples']}")
    print(f"  采样步数:       {stats['num_steps']}")
    print(f"  总耗时:         {stats['total_time']:.2f} 秒")
    print(f"  平均每张图像:   {stats['avg_time_per_image']:.3f} 秒")
    print(f"  平均每步:       {stats['avg_time_per_step']*1000:.2f} 毫秒")
    print(f"{'='*50}\n")


class ControlNetInference:
    """ControlNet推理类"""

    def __init__(
        self,
        checkpoint_path: str,
        ldm_checkpoint_path: str = None,
        vae_pretrained: str = "stabilityai/sd-vae-ft-mse",
        device: str = None
    ):
        """
        Args:
            checkpoint_path: ControlNet训练好的checkpoint路径
            ldm_checkpoint_path: LDM checkpoint路径（用于加载UNet权重，必须提供！）
            vae_pretrained: 预训练VAE模型
            device: 推理设备
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载ControlNet checkpoint
        print(f"加载ControlNet checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # 获取配置
        self.config = checkpoint.get("config", {})
        self.image_size = self.config.get("image_size", 256)
        self.embed_dim = self.config.get("embed_dim", 768)

        # 如果没有提供ldm_checkpoint，尝试从config中获取
        if ldm_checkpoint_path is None:
            ldm_checkpoint_path = self.config.get("ldm_checkpoint", None)

        print(f"图像大小: {self.image_size}")

        # 加载VAE
        print(f"加载VAE: {vae_pretrained}")
        self.vae = AutoencoderKL.from_pretrained(vae_pretrained, torch_dtype=torch.float32)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae = self.vae.to(self.device)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # 创建并加载UNet
        print("加载UNet...")
        self.unet = create_unet(cross_attention_dim=self.embed_dim)

        # 优先从LDM checkpoint加载UNet权重
        if ldm_checkpoint_path:
            print(f"  从LDM checkpoint加载UNet: {ldm_checkpoint_path}")
            ldm_checkpoint = torch.load(ldm_checkpoint_path, map_location="cpu", weights_only=False)
            self.unet.load_state_dict(ldm_checkpoint["unet"])
        elif "unet" in checkpoint:
            print("  从ControlNet checkpoint加载UNet")
            self.unet.load_state_dict(checkpoint["unet"])
        else:
            raise ValueError(
                "UNet权重未找到！请提供--ldm_checkpoint参数指向训练LDM时的checkpoint"
            )

        self.unet.eval()
        self.unet = self.unet.to(self.device)

        # 创建并加载ControlNet
        print("加载ControlNet...")
        self.controlnet = create_controlnet_from_unet(self.unet, conditioning_channels=3)
        self.controlnet.load_state_dict(checkpoint["controlnet"])
        self.controlnet.eval()
        self.controlnet = self.controlnet.to(self.device)

        # 创建并加载DummyEncoder
        print("加载DummyEncoder...")
        self.dummy_encoder = DummyEncoder(embed_dim=self.embed_dim)
        self.dummy_encoder.load_state_dict(checkpoint["dummy_encoder"])
        self.dummy_encoder.eval()
        self.dummy_encoder = self.dummy_encoder.to(self.device)

        # 创建推理调度器
        self.scheduler = create_inference_scheduler()

        # 掩码预处理（归一化到[-1, 1]，与训练时一致）
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])

        print("模型加载完成!")

    def load_mask(self, mask_path: str, channels: int = 1) -> torch.Tensor:
        """
        加载掩码图像

        Args:
            mask_path: 掩码图像路径
            channels: 掩码通道数

        Returns:
            mask: [1, 3, H, W] tensor
        """
        if channels == 1:
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.open(mask_path).convert('RGB')

        mask = self.mask_transform(mask)

        # 扩展为3通道
        if channels == 1:
            mask = mask.repeat(3, 1, 1)

        return mask.unsqueeze(0)

    def load_masks_from_folder(self, folder_path: str, channels: int = 1) -> torch.Tensor:
        """
        从文件夹加载所有掩码

        Args:
            folder_path: 掩码文件夹路径
            channels: 掩码通道数

        Returns:
            masks: [N, 3, H, W] tensor
        """
        mask_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))
        ])

        masks = []
        for f in mask_files:
            mask = self.load_mask(os.path.join(folder_path, f), channels)
            masks.append(mask)

        return torch.cat(masks, dim=0), mask_files

    @torch.no_grad()
    def generate(
        self,
        conditioning: torch.Tensor,
        num_inference_steps: int = 30,
        controlnet_conditioning_scale: float = 1.0,
        seed: int = None,
        num_images_per_mask: int = 1
    ) -> torch.Tensor:
        """
        使用掩码生成图像

        Args:
            conditioning: 控制条件（掩码）[N, 3, H, W]
            num_inference_steps: 采样步数
            controlnet_conditioning_scale: ControlNet强度
            seed: 随机种子
            num_images_per_mask: 每个掩码生成的图像数量

        Returns:
            images: 生成的图像 [N * num_images_per_mask, 3, H, W]
        """
        if seed is not None:
            torch.manual_seed(seed)

        conditioning = conditioning.to(self.device)
        num_masks = conditioning.shape[0]

        # 如果每个掩码生成多张图像，复制掩码
        if num_images_per_mask > 1:
            conditioning = conditioning.repeat_interleave(num_images_per_mask, dim=0)

        num_samples = conditioning.shape[0]
        print(f"生成 {num_samples} 张图像 (掩码数: {num_masks}, 每掩码: {num_images_per_mask})")

        # 记录开始时间
        torch.cuda.synchronize() if self.device != "cpu" else None
        start_time = time.time()

        # 初始化随机噪声
        latent_size = self.image_size // self.vae_scale_factor
        latents = torch.randn(
            num_samples, 4, latent_size, latent_size,
            device=self.device
        )

        # 获取encoder hidden states
        encoder_hidden_states = self.dummy_encoder(num_samples).to(self.device)

        # 设置推理调度器
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 采样循环
        for t in tqdm(self.scheduler.timesteps, desc="采样中"):
            timestep = t.expand(num_samples)

            # ControlNet前向传播
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latents,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=conditioning,
                return_dict=False
            )

            # 应用conditioning scale
            down_block_res_samples = [
                sample * controlnet_conditioning_scale
                for sample in down_block_res_samples
            ]
            mid_block_res_sample = mid_block_res_sample * controlnet_conditioning_scale

            # UNet前向传播
            noise_pred = self.unet(
                latents,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ).sample

            # 调度器步进
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 解码latent到图像
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample

        # 记录结束时间
        torch.cuda.synchronize() if self.device != "cpu" else None
        end_time = time.time()

        # 计算时间统计
        total_time = end_time - start_time
        avg_time_per_image = total_time / num_samples
        avg_time_per_step = total_time / num_inference_steps

        # 保存时间统计到实例变量
        self.last_generation_stats = {
            "total_time": total_time,
            "num_samples": num_samples,
            "num_steps": num_inference_steps,
            "avg_time_per_image": avg_time_per_image,
            "avg_time_per_step": avg_time_per_step
        }

        return images

    @torch.no_grad()
    def generate_from_path(
        self,
        mask_path: str,
        num_inference_steps: int = 30,
        controlnet_conditioning_scale: float = 1.0,
        seed: int = None,
        num_images: int = 1,
        mask_channels: int = 1
    ) -> torch.Tensor:
        """
        从掩码路径生成图像

        Args:
            mask_path: 掩码图像路径
            其他参数同generate()

        Returns:
            images: 生成的图像
        """
        mask = self.load_mask(mask_path, channels=mask_channels)
        return self.generate(
            conditioning=mask,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed,
            num_images_per_mask=num_images
        )


def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet推理脚本")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="ControlNet Checkpoint路径")
    parser.add_argument("--ldm_checkpoint", type=str, required=True,
                        help="LDM Checkpoint路径（必须！用于加载UNet权重）")
    parser.add_argument("--output_dir", type=str, default="./generated",
                        help="输出目录")

    # 输入
    parser.add_argument("--mask", type=str, default=None,
                        help="单个掩码图像路径")
    parser.add_argument("--mask_folder", type=str, default=None,
                        help="掩码文件夹路径（批量生成）")
    parser.add_argument("--mask_channels", type=int, default=1,
                        help="掩码通道数")

    # 生成设置
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="采样步数")
    parser.add_argument("--controlnet_scale", type=float, default=1.0,
                        help="ControlNet强度（0-2，越高越严格遵循掩码）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--num_images", type=int, default=1,
                        help="每个掩码生成的图像数量")

    # 模型设置
    parser.add_argument("--vae_pretrained", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="预训练VAE模型")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备")

    # 输出设置
    parser.add_argument("--save_comparison", action="store_true",
                        help="是否保存掩码-图像对比图")

    return parser.parse_args()


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化推理
    inference = ControlNetInference(
        checkpoint_path=args.checkpoint,
        ldm_checkpoint_path=args.ldm_checkpoint,
        vae_pretrained=args.vae_pretrained,
        device=args.device
    )

    if args.mask:
        # 单个掩码生成
        print(f"\n从掩码生成: {args.mask}")

        mask = inference.load_mask(args.mask, channels=args.mask_channels)
        images = inference.generate(
            conditioning=mask,
            num_inference_steps=args.num_inference_steps,
            controlnet_conditioning_scale=args.controlnet_scale,
            seed=args.seed,
            num_images_per_mask=args.num_images
        )

        # 打印时间统计
        print_timing_stats(inference.last_generation_stats)

        # 保存生成的图像
        mask_name = Path(args.mask).stem
        for i, img in enumerate(images):
            img_path = output_dir / f"{mask_name}_gen_{i:04d}.png"
            save_image(img, img_path, normalize=True, value_range=(-1, 1))
        print(f"生成 {len(images)} 张图像，保存到: {output_dir}")

        # 保存对比图
        if args.save_comparison:
            # 掩码 | 生成图像
            mask_expanded = mask.repeat(args.num_images, 1, 1, 1)
            comparison = torch.cat([mask_expanded, images.cpu()], dim=3)  # 水平拼接
            grid = make_grid(comparison, nrow=1, normalize=True, value_range=(-1, 1))
            save_image(grid, output_dir / f"{mask_name}_comparison.png")

    elif args.mask_folder:
        # 批量生成
        print(f"\n从文件夹批量生成: {args.mask_folder}")

        masks, mask_files = inference.load_masks_from_folder(
            args.mask_folder,
            channels=args.mask_channels
        )

        # 分批处理以避免显存不足
        batch_size = 4
        all_images = []
        total_time = 0.0
        total_samples = 0

        for i in range(0, len(masks), batch_size):
            batch_masks = masks[i:i+batch_size]
            batch_images = inference.generate(
                conditioning=batch_masks,
                num_inference_steps=args.num_inference_steps,
                controlnet_conditioning_scale=args.controlnet_scale,
                seed=args.seed if args.seed else None,
                num_images_per_mask=args.num_images
            )
            all_images.append(batch_images.cpu())

            # 累积时间统计
            total_time += inference.last_generation_stats["total_time"]
            total_samples += inference.last_generation_stats["num_samples"]

        all_images = torch.cat(all_images, dim=0)

        # 打印总体时间统计
        overall_stats = {
            "total_time": total_time,
            "num_samples": total_samples,
            "num_steps": args.num_inference_steps,
            "avg_time_per_image": total_time / total_samples if total_samples > 0 else 0,
            "avg_time_per_step": total_time / (total_samples * args.num_inference_steps) if total_samples > 0 else 0
        }
        print_timing_stats(overall_stats, title="批量生成时间统计")

        # 保存
        idx = 0
        for i, mask_file in enumerate(mask_files):
            mask_name = Path(mask_file).stem
            for j in range(args.num_images):
                img_path = output_dir / f"{mask_name}_gen_{j:04d}.png"
                save_image(all_images[idx], img_path, normalize=True, value_range=(-1, 1))
                idx += 1

        print(f"生成 {len(all_images)} 张图像，保存到: {output_dir}")

        # 保存总览网格
        if args.save_comparison:
            # 每行：掩码 + 对应的生成图像
            rows = []
            idx = 0
            for i in range(len(masks)):
                row = [masks[i]]
                for j in range(args.num_images):
                    row.append(all_images[idx])
                    idx += 1
                rows.append(torch.cat(row, dim=2))  # 水平拼接

            grid = make_grid(torch.stack(rows), nrow=1, normalize=True, value_range=(-1, 1))
            save_image(grid, output_dir / "all_comparison.png")

    else:
        print("错误: 请指定 --mask 或 --mask_folder")
        return

    print("\n推理完成!")


if __name__ == "__main__":
    main()
