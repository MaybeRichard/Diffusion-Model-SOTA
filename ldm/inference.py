"""
类别条件LDM推理脚本

功能：
- 加载训练好的模型
- 指定类别生成图像
- 支持批量生成
- 支持CFG强度调节
"""

import os
import argparse
import json
import time
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from diffusers import AutoencoderKL

from models import ClassEmbedder, create_unet, create_inference_scheduler


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


class LDMInference:
    """LDM推理类"""

    def __init__(
        self,
        checkpoint_path: str,
        vae_pretrained: str = "stabilityai/sd-vae-ft-mse",
        device: str = None
    ):
        """
        Args:
            checkpoint_path: 训练好的checkpoint路径
            vae_pretrained: 预训练VAE模型
            device: 推理设备
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载checkpoint
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # 获取配置
        self.config = checkpoint.get("config", {})
        self.class_names = checkpoint["class_names"]
        self.num_classes = checkpoint["num_classes"]
        self.image_size = self.config.get("image_size", 256)
        self.embed_dim = self.config.get("embed_dim", 768)

        print(f"类别列表: {self.class_names}")
        print(f"图像大小: {self.image_size}")

        # 创建类别到索引的映射
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

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
        self.unet.load_state_dict(checkpoint["unet"])
        self.unet.eval()
        self.unet = self.unet.to(self.device)

        # 创建并加载ClassEmbedder
        print("加载ClassEmbedder...")
        self.class_embedder = ClassEmbedder(
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            dropout_prob=0.0  # 推理时不需要dropout
        )
        self.class_embedder.load_state_dict(checkpoint["class_embedder"])
        self.class_embedder.eval()
        self.class_embedder = self.class_embedder.to(self.device)

        # 创建推理调度器
        self.scheduler = create_inference_scheduler()

        print("模型加载完成!")

    def get_class_idx(self, class_input) -> int:
        """
        获取类别索引

        Args:
            class_input: 类别名称（str）或索引（int）

        Returns:
            class_idx: 类别索引
        """
        if isinstance(class_input, int):
            if 0 <= class_input < self.num_classes:
                return class_input
            else:
                raise ValueError(f"类别索引 {class_input} 超出范围 [0, {self.num_classes-1}]")
        elif isinstance(class_input, str):
            if class_input in self.class_to_idx:
                return self.class_to_idx[class_input]
            else:
                raise ValueError(f"未知类别 '{class_input}'，可用类别: {self.class_names}")
        else:
            raise TypeError(f"class_input 必须是 str 或 int，而不是 {type(class_input)}")

    @torch.no_grad()
    def generate(
        self,
        class_label,
        num_samples: int = 1,
        num_inference_steps: int = 50,
        cfg_scale: float = 3.0,
        seed: int = None,
        batch_size: int = 16
    ) -> torch.Tensor:
        """
        生成指定类别的图像

        Args:
            class_label: 类别名称（str）或索引（int）
            num_samples: 生成数量
            num_inference_steps: 采样步数
            cfg_scale: Classifier-Free Guidance强度
            seed: 随机种子（None则随机）
            batch_size: 每批生成的样本数（防止OOM）

        Returns:
            images: 生成的图像 [N, 3, H, W]，范围[-1, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 获取类别索引
        class_idx = self.get_class_idx(class_label)
        class_name = self.class_names[class_idx]
        print(f"生成类别: {class_name} (idx={class_idx}), 数量: {num_samples}, 批大小: {batch_size}")

        latent_size = self.image_size // self.vae_scale_factor
        all_images = []

        # 计时
        total_start_time = time.time()
        batch_times = []

        # 分批生成
        num_batches = (num_samples + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx

            # 准备类别标签
            class_labels = torch.full(
                (current_batch_size,), class_idx,
                dtype=torch.long, device=self.device
            )

            # 获取条件和无条件嵌入
            cond_embeddings = self.class_embedder(class_labels, train=False)
            uncond_embeddings = self.class_embedder.get_null_embedding(current_batch_size, self.device)

            # 初始化随机噪声
            latents = torch.randn(
                current_batch_size, 4, latent_size, latent_size,
                device=self.device
            )

            # 设置推理调度器
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)

            # DDIM采样循环
            desc = f"采样中 (批次 {batch_idx+1}/{num_batches})"
            for t in tqdm(self.scheduler.timesteps, desc=desc):
                # 扩展时间步
                timestep = t.expand(current_batch_size)

                # CFG: 同时预测条件和无条件噪声
                latent_input = torch.cat([latents, latents], dim=0)
                timestep_input = torch.cat([timestep, timestep], dim=0)
                embedding_input = torch.cat([cond_embeddings, uncond_embeddings], dim=0)

                # 预测噪声
                noise_pred = self.unet(
                    latent_input,
                    timestep_input,
                    encoder_hidden_states=embedding_input
                ).sample

                # 分离条件和无条件预测
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)

                # 应用CFG
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

                # DDIM步进
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # 解码latent到图像
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents).sample

            # 移到CPU并收集结果，释放GPU显存
            all_images.append(images.cpu())
            del latents, images, cond_embeddings, uncond_embeddings
            torch.cuda.empty_cache()

            # 记录批次时间
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

        # 计算总体统计
        torch.cuda.synchronize() if self.device != "cpu" else None
        total_time = time.time() - total_start_time

        # 保存时间统计到实例变量
        self.last_generation_stats = {
            "total_time": total_time,
            "num_samples": num_samples,
            "num_steps": num_inference_steps,
            "avg_time_per_image": total_time / num_samples,
            "avg_time_per_step": total_time / (num_samples * num_inference_steps)
        }

        # 合并所有批次
        return torch.cat(all_images, dim=0)

    @torch.no_grad()
    def generate_all_classes(
        self,
        num_samples_per_class: int = 4,
        num_inference_steps: int = 50,
        cfg_scale: float = 3.0,
        seed: int = None,
        batch_size: int = 16
    ) -> dict:
        """
        为所有类别生成图像

        Returns:
            dict: {class_name: images_tensor}
        """
        results = {}

        for class_idx, class_name in enumerate(self.class_names):
            images = self.generate(
                class_label=class_idx,
                num_samples=num_samples_per_class,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                seed=seed,
                batch_size=batch_size
            )
            results[class_name] = images

        return results


def parse_args():
    parser = argparse.ArgumentParser(description="LDM推理脚本")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint路径")
    parser.add_argument("--output_dir", type=str, default="./generated",
                        help="输出目录")

    # 生成设置
    parser.add_argument("--class_label", type=str, default=None,
                        help="要生成的类别名称（不指定则生成所有类别）")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="每个类别生成的样本数量")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="采样步数")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="CFG强度")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="每批生成的样本数（防止OOM，默认16）")

    # 模型设置
    parser.add_argument("--vae_pretrained", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="预训练VAE模型")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备（cuda/cpu）")

    # 输出设置
    parser.add_argument("--save_grid", action="store_true",
                        help="是否保存网格图像")
    parser.add_argument("--save_individual", action="store_true", default=True,
                        help="是否保存单独图像")

    return parser.parse_args()


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化推理
    inference = LDMInference(
        checkpoint_path=args.checkpoint,
        vae_pretrained=args.vae_pretrained,
        device=args.device
    )

    # 打印可用类别
    print(f"\n可用类别: {inference.class_names}")

    if args.class_label is not None:
        # 生成指定类别
        images = inference.generate(
            class_label=args.class_label,
            num_samples=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            batch_size=args.batch_size
        )

        # 打印时间统计
        print_timing_stats(inference.last_generation_stats)

        class_idx = inference.get_class_idx(args.class_label)
        class_name = inference.class_names[class_idx]

        # 保存图像
        if args.save_individual:
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            for i, img in enumerate(images):
                img_path = class_dir / f"{class_name}_{i:04d}.png"
                save_image(img, img_path, normalize=True, value_range=(-1, 1))
            print(f"图像已保存到: {class_dir}")

        if args.save_grid:
            nrow = int(args.num_samples ** 0.5)
            grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
            grid_path = output_dir / f"{class_name}_grid.png"
            save_image(grid, grid_path)
            print(f"网格图像已保存到: {grid_path}")

    else:
        # 生成所有类别
        print("\n生成所有类别...")
        results = inference.generate_all_classes(
            num_samples_per_class=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            batch_size=args.batch_size
        )

        # 保存图像
        all_images = []
        for class_name, images in results.items():
            all_images.append(images)

            if args.save_individual:
                class_dir = output_dir / class_name
                class_dir.mkdir(exist_ok=True)
                for i, img in enumerate(images):
                    img_path = class_dir / f"{class_name}_{i:04d}.png"
                    save_image(img, img_path, normalize=True, value_range=(-1, 1))
                print(f"{class_name}: {len(images)} 张图像已保存")

        if args.save_grid:
            # 创建所有类别的大网格
            all_images = torch.cat(all_images, dim=0)
            grid = make_grid(all_images, nrow=args.num_samples, normalize=True, value_range=(-1, 1))
            grid_path = output_dir / "all_classes_grid.png"
            save_image(grid, grid_path)
            print(f"\n总网格图像已保存到: {grid_path}")

    print("\n推理完成!")


if __name__ == "__main__":
    main()
