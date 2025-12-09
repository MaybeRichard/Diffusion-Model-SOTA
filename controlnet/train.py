"""
ControlNet训练脚本

特性：
- 使用预训练VAE（冻结）
- UNet可选冻结或微调
- 训练ControlNet分支
- 最优权重保存
- 训练过程可视化
- 支持断点续训
- 支持多卡并行
"""

import os
import sys
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.utils import save_image, make_grid

from dataset import get_dataloader
from models import ControlNetLDM, create_inference_scheduler


def print_model_summary(model, freeze_unet=True):
    """打印模型各组件参数量"""
    print(f"\n{'='*60}")
    print("ControlNet 模型参数量详情:")
    print(f"{'='*60}")

    # VAE 参数
    vae_total = sum(p.numel() for p in model.vae.parameters())
    print(f"  VAE:            {vae_total:>12,} ({vae_total/1e6:.2f}M) [冻结]")

    # UNet 参数
    unet_total = sum(p.numel() for p in model.unet.parameters())
    unet_trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    unet_status = "冻结" if freeze_unet else "可训练"
    print(f"  UNet:           {unet_total:>12,} ({unet_total/1e6:.2f}M) [{unet_status}]")

    # ControlNet 参数
    cn_total = sum(p.numel() for p in model.controlnet.parameters())
    cn_trainable = sum(p.numel() for p in model.controlnet.parameters() if p.requires_grad)
    print(f"  ControlNet:     {cn_total:>12,} ({cn_total/1e6:.2f}M) [可训练]")

    # DummyEncoder 参数
    enc_total = sum(p.numel() for p in model.dummy_encoder.parameters())
    print(f"  DummyEncoder:   {enc_total:>12,} ({enc_total/1e6:.4f}M) [可训练]")

    print(f"{'-'*60}")

    total = vae_total + unet_total + cn_total + enc_total
    trainable = cn_trainable + enc_total + (unet_trainable if not freeze_unet else 0)
    print(f"  总计:           {total:>12,} ({total/1e6:.2f}M)")
    print(f"  可训练:         {trainable:>12,} ({trainable/1e6:.2f}M)")
    print(f"{'='*60}\n")

    return total, trainable


def parse_args():
    parser = argparse.ArgumentParser(description="训练ControlNet")

    # 数据相关
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据集根目录")
    parser.add_argument("--images_folder", type=str, default="images",
                        help="图像文件夹名称")
    parser.add_argument("--masks_folder", type=str, default="masks",
                        help="掩码文件夹名称")
    parser.add_argument("--image_size", type=int, default=256,
                        help="图像大小")
    parser.add_argument("--mask_channels", type=int, default=1,
                        help="掩码通道数")
    parser.add_argument("--image_prefix", type=str, default=None,
                        help="图像文件名前缀（如 'Img_'，用于Img_XXX对应seg_XXX的情况）")
    parser.add_argument("--mask_prefix", type=str, default=None,
                        help="掩码文件名前缀（如 'seg_'）")

    # 模型相关
    parser.add_argument("--vae_pretrained", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="预训练VAE模型")
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="嵌入维度")
    parser.add_argument("--ldm_checkpoint", type=str, default=None,
                        help="预训练LDM checkpoint路径（强烈推荐！先训练LDM再训练ControlNet）")
    parser.add_argument("--freeze_unet", action="store_true", default=True,
                        help="是否冻结UNet")
    parser.add_argument("--no_freeze_unet", action="store_false", dest="freeze_unet",
                        help="不冻结UNet（微调）")

    # 训练相关
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0,
                        help="ControlNet条件强度")

    # 保存和日志
    parser.add_argument("--output_dir", type=str, default="./controlnet_checkpoints",
                        help="输出目录")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每隔多少epoch保存checkpoint")
    parser.add_argument("--sample_every", type=int, default=5,
                        help="每隔多少epoch生成样本")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="推理时的采样步数")
    parser.add_argument("--num_validation_samples", type=int, default=4,
                        help="验证时生成的样本数量")

    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--resume", type=str, default=None,
                        help="从checkpoint恢复训练")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="混合精度训练")

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args

        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        # 初始化accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="tensorboard",
            project_dir=str(self.output_dir)
        )

        # 设置随机种子
        set_seed(args.seed)

        # 初始化数据加载器
        self.dataloader, self.dataset = get_dataloader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            augment=True,
            images_folder=args.images_folder,
            masks_folder=args.masks_folder,
            mask_channels=args.mask_channels,
            image_prefix=args.image_prefix,
            mask_prefix=args.mask_prefix
        )

        # 初始化模型
        if args.ldm_checkpoint:
            # 从预训练LDM加载（推荐方式）
            if self.accelerator.is_main_process:
                print(f"\n从预训练LDM加载模型...")
            self.model = ControlNetLDM.from_ldm_checkpoint(
                ldm_checkpoint_path=args.ldm_checkpoint,
                vae_pretrained=args.vae_pretrained,
                conditioning_channels=3,
                freeze_unet=args.freeze_unet
            )
        else:
            # 从头创建（不推荐，除非数据集很大）
            if self.accelerator.is_main_process:
                print("\n警告: 未提供LDM checkpoint，UNet将随机初始化！")
                print("推荐流程: 先训练LDM，再用--ldm_checkpoint加载训练ControlNet")
            self.model = ControlNetLDM(
                vae_pretrained=args.vae_pretrained,
                embed_dim=args.embed_dim,
                conditioning_channels=3,
                freeze_unet=args.freeze_unet
            )

        # 打印模型参数量
        if self.accelerator.is_main_process:
            print_model_summary(self.model, freeze_unet=args.freeze_unet)

        # 初始化优化器
        self.optimizer = AdamW(
            self.model.get_trainable_parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # 初始化学习率调度器
        num_training_steps = len(self.dataloader) * args.num_epochs
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=args.learning_rate * 0.1
        )

        # 推理调度器
        self.inference_scheduler = create_inference_scheduler()

        # 保存验证掩码（用于生成样本时保持一致）
        self.validation_masks = None

        # 使用accelerator准备
        self.model, self.optimizer, self.dataloader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.dataloader, self.lr_scheduler
            )

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')

        # 从checkpoint恢复
        if args.resume:
            self.load_checkpoint(args.resume)

        # 保存配置
        if self.accelerator.is_main_process:
            config = vars(args)
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    def train(self):
        """主训练循环"""
        if self.accelerator.is_main_process:
            print(f"\n开始训练ControlNet...")
            print(f"总epoch数: {self.args.num_epochs}")
            print(f"批次大小: {self.args.batch_size}")
            print(f"梯度累积步数: {self.args.gradient_accumulation_steps}")
            print(f"有效批次大小: {self.args.batch_size * self.args.gradient_accumulation_steps}")
            print(f"UNet冻结: {self.args.freeze_unet}")

        # 初始化tensorboard
        self.accelerator.init_trackers("controlnet_training")

        # 获取固定的验证掩码
        self._prepare_validation_samples()

        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch(epoch)

            # 保存最优模型
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                if self.accelerator.is_main_process:
                    self.save_checkpoint("best")
                    print(f"  保存最优模型 (loss: {train_loss:.6f})")

            # 定期保存checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                if self.accelerator.is_main_process:
                    self.save_checkpoint(f"epoch_{epoch+1}")

            # 定期生成样本
            if (epoch + 1) % self.args.sample_every == 0:
                if self.accelerator.is_main_process:
                    self.generate_samples(epoch + 1)

        # 保存最终模型
        if self.accelerator.is_main_process:
            self.save_checkpoint("final")
            print("\n训练完成!")

        self.accelerator.end_training()

    def _prepare_validation_samples(self):
        """准备固定的验证样本"""
        # 从数据集中获取固定的掩码用于验证
        num_samples = min(self.args.num_validation_samples, len(self.dataset))
        masks = []
        images = []

        for i in range(num_samples):
            sample = self.dataset[i]
            masks.append(sample["conditioning"])
            images.append(sample["pixel_values"])

        self.validation_masks = torch.stack(masks)
        self.validation_images = torch.stack(images)

        if self.accelerator.is_main_process:
            # 保存验证掩码
            grid = make_grid(self.validation_masks, nrow=2, normalize=True)
            save_image(grid, self.samples_dir / "validation_masks.png")

            # 保存对应的真实图像
            grid = make_grid(self.validation_images, nrow=2, normalize=True, value_range=(-1, 1))
            save_image(grid, self.samples_dir / "validation_ground_truth.png")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        model = self.accelerator.unwrap_model(self.model)
        model.controlnet.train()
        model.dummy_encoder.train()
        if not self.args.freeze_unet:
            model.unet.train()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch+1}/{self.args.num_epochs}",
            disable=not self.accelerator.is_main_process
        )

        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                images = batch["pixel_values"]
                conditioning = batch["conditioning"]

                # 前向传播
                loss = self.model(
                    images,
                    conditioning,
                    controlnet_conditioning_scale=self.args.controlnet_conditioning_scale
                )

                # 反向传播
                self.accelerator.backward(loss)

                # 梯度裁剪
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        model.get_trainable_parameters(),
                        self.args.max_grad_norm
                    )

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
            })

            if self.global_step % 10 == 0:
                self.accelerator.log({
                    "train_loss": loss.item(),
                    "learning_rate": self.lr_scheduler.get_last_lr()[0]
                }, step=self.global_step)

        avg_loss = total_loss / num_batches

        if self.accelerator.is_main_process:
            print(f"  Epoch {epoch+1} 平均loss: {avg_loss:.6f}")
            self.accelerator.log({"epoch_loss": avg_loss}, step=epoch)

        return avg_loss

    @torch.no_grad()
    def generate_samples(self, epoch):
        """生成样本图像"""
        print(f"  生成样本图像...")

        model = self.accelerator.unwrap_model(self.model)
        model.eval()

        device = self.accelerator.device

        # 使用验证掩码生成
        conditioning = self.validation_masks.to(device)
        num_samples = conditioning.shape[0]

        # 生成图像
        images = self.sample(
            model=model,
            conditioning=conditioning,
            device=device
        )

        # 拼接掩码和生成图像
        # 掩码 | 生成图像
        comparison = torch.cat([conditioning, images], dim=0)
        grid = make_grid(comparison, nrow=num_samples, normalize=True, value_range=(-1, 1))

        save_path = self.samples_dir / f"samples_epoch_{epoch}.png"
        save_image(grid, save_path)
        print(f"  样本已保存到: {save_path}")

    @torch.no_grad()
    def sample(
        self,
        model,
        conditioning: torch.Tensor,
        device: torch.device,
        controlnet_conditioning_scale: float = None
    ) -> torch.Tensor:
        """
        使用ControlNet采样生成图像

        Args:
            model: ControlNetLDM模型
            conditioning: 控制条件（掩码）[N, 3, H, W]
            device: 设备
            controlnet_conditioning_scale: ControlNet强度

        Returns:
            images: 生成的图像 [N, 3, H, W]
        """
        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = self.args.controlnet_conditioning_scale

        num_samples = conditioning.shape[0]

        # 初始化随机噪声
        latent_size = self.args.image_size // model.vae_scale_factor
        latents = torch.randn(
            num_samples, 4, latent_size, latent_size,
            device=device
        )

        # 获取encoder hidden states
        encoder_hidden_states = model.dummy_encoder(num_samples).to(device)

        # 设置推理调度器
        self.inference_scheduler.set_timesteps(self.args.num_inference_steps, device=device)

        # 采样循环
        for t in self.inference_scheduler.timesteps:
            timestep = t.expand(num_samples)

            # ControlNet前向传播
            down_block_res_samples, mid_block_res_sample = model.controlnet(
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
            noise_pred = model.unet(
                latents,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ).sample

            # 调度器步进
            latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample

        # 解码latent到图像
        images = model.decode_latents(latents)

        return images

    def save_checkpoint(self, name):
        """保存checkpoint"""
        model = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            "controlnet": model.controlnet.state_dict(),
            "dummy_encoder": model.dummy_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": vars(self.args)
        }

        # 如果没有冻结UNet，也保存UNet权重
        if not self.args.freeze_unet:
            checkpoint["unet"] = model.unet.state_dict()

        save_path = self.output_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint已保存到: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        print(f"从 {checkpoint_path} 恢复训练...")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model = self.accelerator.unwrap_model(self.model)
        model.controlnet.load_state_dict(checkpoint["controlnet"])
        model.dummy_encoder.load_state_dict(checkpoint["dummy_encoder"])

        if "unet" in checkpoint and not self.args.freeze_unet:
            model.unet.load_state_dict(checkpoint["unet"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        print(f"  从epoch {self.current_epoch} 继续训练")
        print(f"  当前最优loss: {self.best_loss:.6f}")


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
