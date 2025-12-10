"""
按类别生成图像的LDM推理脚本

功能：
- 为每个类别生成指定数量的图像
- 每个类别单独创建子文件夹
- 支持指定总数量（自动平均分配到各类别）
- 支持批量生成防止OOM
"""

import os
import argparse
import time
from pathlib import Path

import torch
from torchvision.utils import save_image
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
    print(f"  吞吐量:         {stats['throughput']:.2f} 张/秒")
    print(f"{'='*50}\n")


class LDMInferencePerClass:
    """按类别生成图像的LDM推理类"""

    def __init__(
        self,
        checkpoint_path: str,
        vae_pretrained: str = "stabilityai/sd-vae-ft-mse",
        device: str = None
    ):
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
        print(f"类别数量: {self.num_classes}")
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
        self.unet.load_state_dict(checkpoint["unet"])
        self.unet.eval()
        self.unet = self.unet.to(self.device)

        # 创建并加载ClassEmbedder
        print("加载ClassEmbedder...")
        self.class_embedder = ClassEmbedder(
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            dropout_prob=0.0
        )
        self.class_embedder.load_state_dict(checkpoint["class_embedder"])
        self.class_embedder.eval()
        self.class_embedder = self.class_embedder.to(self.device)

        # 创建推理调度器
        self.scheduler = create_inference_scheduler()

        print("模型加载完成!")

    @torch.no_grad()
    def generate_and_save_batch(
        self,
        class_idx: int,
        batch_size: int,
        num_inference_steps: int,
        cfg_scale: float,
        output_dir: Path,
        start_idx: int
    ) -> float:
        """
        生成一批图像并保存到磁盘

        Returns:
            batch_time: 该批次耗时
        """
        batch_start = time.time()

        latent_size = self.image_size // self.vae_scale_factor

        # 准备类别标签
        class_labels = torch.full(
            (batch_size,), class_idx,
            dtype=torch.long, device=self.device
        )

        # 获取条件和无条件嵌入
        cond_embeddings = self.class_embedder(class_labels, train=False)
        uncond_embeddings = self.class_embedder.get_null_embedding(batch_size, self.device)

        # 初始化随机噪声
        latents = torch.randn(
            batch_size, 4, latent_size, latent_size,
            device=self.device
        )

        # 设置推理调度器
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # DDIM采样循环
        for t in self.scheduler.timesteps:
            timestep = t.expand(batch_size)

            # CFG
            latent_input = torch.cat([latents, latents], dim=0)
            timestep_input = torch.cat([timestep, timestep], dim=0)
            embedding_input = torch.cat([cond_embeddings, uncond_embeddings], dim=0)

            noise_pred = self.unet(
                latent_input,
                timestep_input,
                encoder_hidden_states=embedding_input
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 解码
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample

        # 保存图像
        class_name = self.class_names[class_idx]
        for i, img in enumerate(images):
            img_idx = start_idx + i
            img_path = output_dir / f"{class_name}_{img_idx:05d}.png"
            save_image(img, img_path, normalize=True, value_range=(-1, 1))

        # 清理显存
        del latents, images, cond_embeddings, uncond_embeddings
        torch.cuda.empty_cache()

        return time.time() - batch_start

    def generate_for_class(
        self,
        class_idx: int,
        num_samples: int,
        output_dir: Path,
        num_inference_steps: int = 50,
        cfg_scale: float = 3.0,
        batch_size: int = 16
    ) -> dict:
        """
        为单个类别生成指定数量的图像

        Returns:
            stats: 时间统计字典
        """
        class_name = self.class_names[class_idx]
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n生成类别: {class_name} (idx={class_idx})")
        print(f"  数量: {num_samples}, 批大小: {batch_size}")
        print(f"  保存目录: {class_dir}")

        total_start = time.time()
        num_batches = (num_samples + batch_size - 1) // batch_size
        generated = 0

        pbar = tqdm(total=num_samples, desc=f"  {class_name}")

        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - generated)

            batch_time = self.generate_and_save_batch(
                class_idx=class_idx,
                batch_size=current_batch_size,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                output_dir=class_dir,
                start_idx=generated
            )

            generated += current_batch_size
            pbar.update(current_batch_size)

        pbar.close()

        if self.device != "cpu":
            torch.cuda.synchronize()

        total_time = time.time() - total_start

        return {
            "class_name": class_name,
            "num_samples": num_samples,
            "num_steps": num_inference_steps,
            "total_time": total_time,
            "avg_time_per_image": total_time / num_samples,
            "throughput": num_samples / total_time
        }

    def generate_all_classes(
        self,
        total_samples: int,
        output_dir: Path,
        num_inference_steps: int = 50,
        cfg_scale: float = 3.0,
        batch_size: int = 16,
        seed: int = None
    ) -> dict:
        """
        为所有类别生成图像

        Args:
            total_samples: 总样本数（平均分配到各类别）
            output_dir: 输出目录
            num_inference_steps: 采样步数
            cfg_scale: CFG强度
            batch_size: 批大小
            seed: 随机种子

        Returns:
            overall_stats: 总体统计
        """
        if seed is not None:
            torch.manual_seed(seed)

        samples_per_class = total_samples // self.num_classes
        remainder = total_samples % self.num_classes

        print(f"\n{'='*60}")
        print(f"按类别生成图像")
        print(f"{'='*60}")
        print(f"  总样本数: {total_samples}")
        print(f"  类别数: {self.num_classes}")
        print(f"  每类别样本数: {samples_per_class}" +
              (f" (前{remainder}个类别多1张)" if remainder > 0 else ""))
        print(f"  输出目录: {output_dir}")
        print(f"{'='*60}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_stats = []
        overall_start = time.time()

        for class_idx in range(self.num_classes):
            # 前remainder个类别多生成1张
            num_samples = samples_per_class + (1 if class_idx < remainder else 0)

            if num_samples == 0:
                continue

            stats = self.generate_for_class(
                class_idx=class_idx,
                num_samples=num_samples,
                output_dir=output_dir,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                batch_size=batch_size
            )
            all_stats.append(stats)

            # 打印该类别统计
            print(f"  完成: {stats['total_time']:.1f}s, "
                  f"{stats['throughput']:.2f} 张/秒")

        overall_time = time.time() - overall_start
        total_generated = sum(s['num_samples'] for s in all_stats)

        overall_stats = {
            "num_samples": total_generated,
            "num_classes": len(all_stats),
            "num_steps": num_inference_steps,
            "total_time": overall_time,
            "avg_time_per_image": overall_time / total_generated,
            "throughput": total_generated / overall_time,
            "per_class_stats": all_stats
        }

        return overall_stats


def parse_args():
    parser = argparse.ArgumentParser(description="按类别生成LDM图像")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint路径")
    parser.add_argument("--output_dir", type=str, default="./generated_per_class",
                        help="输出目录")

    # 生成设置
    parser.add_argument("--total_samples", type=int, default=8000,
                        help="总生成数量（平均分配到各类别）")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="采样步数")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="CFG强度")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="每批生成的样本数（防止OOM）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")

    # 模型设置
    parser.add_argument("--vae_pretrained", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="预训练VAE模型")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备（cuda/cpu）")

    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化推理
    inference = LDMInferencePerClass(
        checkpoint_path=args.checkpoint,
        vae_pretrained=args.vae_pretrained,
        device=args.device
    )

    # 生成所有类别
    stats = inference.generate_all_classes(
        total_samples=args.total_samples,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        seed=args.seed
    )

    # 打印总体统计
    print_timing_stats(stats, "总体采样统计")

    # 打印各类别统计
    print("各类别统计:")
    print("-" * 60)
    for s in stats['per_class_stats']:
        print(f"  {s['class_name']:20s}: {s['num_samples']:5d} 张, "
              f"{s['total_time']:6.1f}s, {s['throughput']:.2f} 张/秒")
    print("-" * 60)

    print(f"\n所有图像已保存到: {args.output_dir}")
    print("推理完成!")


if __name__ == "__main__":
    main()
