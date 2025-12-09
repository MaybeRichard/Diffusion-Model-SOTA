"""
类别条件LDM训练脚本

特性：
- 使用预训练VAE（冻结）
- 训练UNet和ClassEmbedder
- Classifier-Free Guidance支持
- 最优权重保存
- 训练过程可视化
- 支持断点续训
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
from models import ClassConditionedLDM, create_inference_scheduler


def count_parameters(model, name="Model"):
    """统计并打印模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n{'='*50}")
    print(f"{name} 参数统计:")
    print(f"  总参数量:     {total_params:>15,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数:   {trainable_params:>15,} ({trainable_params/1e6:.2f}M)")
    print(f"  冻结参数:     {frozen_params:>15,} ({frozen_params/1e6:.2f}M)")
    print(f"{'='*50}")

    return total_params, trainable_params


def print_model_summary(model):
    """打印模型各组件参数量"""
    print(f"\n{'='*60}")
    print("模型参数量详情:")
    print(f"{'='*60}")

    # VAE 参数
    vae_total = sum(p.numel() for p in model.vae.parameters())
    vae_trainable = sum(p.numel() for p in model.vae.parameters() if p.requires_grad)
    print(f"  VAE:            {vae_total:>12,} ({vae_total/1e6:.2f}M) [冻结]")

    # UNet 参数
    unet_total = sum(p.numel() for p in model.unet.parameters())
    unet_trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    print(f"  UNet:           {unet_total:>12,} ({unet_total/1e6:.2f}M) [可训练]")

    # ClassEmbedder 参数
    embed_total = sum(p.numel() for p in model.class_embedder.parameters())
    embed_trainable = sum(p.numel() for p in model.class_embedder.parameters() if p.requires_grad)
    print(f"  ClassEmbedder:  {embed_total:>12,} ({embed_total/1e6:.2f}M) [可训练]")

    print(f"{'-'*60}")

    total = vae_total + unet_total + embed_total
    trainable = unet_trainable + embed_trainable
    print(f"  总计:           {total:>12,} ({total/1e6:.2f}M)")
    print(f"  可训练:         {trainable:>12,} ({trainable/1e6:.2f}M)")
    print(f"{'='*60}\n")

    return total, trainable


def parse_args():
    parser = argparse.ArgumentParser(description="训练类别条件LDM")

    # 数据相关
    parser.add_argument("--data_root", type=str, default="./output_dir",
                        help="数据集根目录")
    parser.add_argument("--image_size", type=int, default=256,
                        help="图像大小")

    # 模型相关
    parser.add_argument("--vae_pretrained", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="预训练VAE模型")
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="类别嵌入维度")
    parser.add_argument("--cfg_dropout", type=float, default=0.1,
                        help="Classifier-Free Guidance dropout概率")

    # 训练相关
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")

    # 保存和日志
    parser.add_argument("--output_dir", type=str, default="./ldm_checkpoints",
                        help="输出目录")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每隔多少epoch保存checkpoint")
    parser.add_argument("--sample_every", type=int, default=5,
                        help="每隔多少epoch生成样本")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="推理时的采样步数")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="推理时的CFG强度")

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
            augment=True
        )
        self.num_classes = self.dataset.num_classes
        self.class_names = self.dataset.class_names

        # 打印类别信息
        if self.accelerator.is_main_process:
            print(f"\n类别列表: {self.class_names}")
            print(f"类别数量: {self.num_classes}")

        # 初始化模型
        self.model = ClassConditionedLDM(
            num_classes=self.num_classes,
            vae_pretrained=args.vae_pretrained,
            embed_dim=args.embed_dim,
            cfg_dropout=args.cfg_dropout
        )

        # 打印模型参数量
        if self.accelerator.is_main_process:
            print_model_summary(self.model)

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

        # 推理调度器（用于生成样本）
        self.inference_scheduler = create_inference_scheduler()

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
            config["class_names"] = self.class_names
            config["num_classes"] = self.num_classes
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    def train(self):
        """主训练循环"""
        if self.accelerator.is_main_process:
            print(f"\n开始训练...")
            print(f"总epoch数: {self.args.num_epochs}")
            print(f"批次大小: {self.args.batch_size}")
            print(f"梯度累积步数: {self.args.gradient_accumulation_steps}")
            print(f"有效批次大小: {self.args.batch_size * self.args.gradient_accumulation_steps}")

        # 初始化tensorboard
        self.accelerator.init_trackers("ldm_training")

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

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
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
                class_labels = batch["class_labels"]

                # 前向传播
                loss = self.model(images, class_labels)

                # 反向传播
                self.accelerator.backward(loss)

                # 梯度裁剪
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
            })

            # 记录到tensorboard
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
        num_samples_per_class = 2

        all_images = []

        for class_idx in range(self.num_classes):
            # 生成该类别的样本
            images = self.sample(
                model=model,
                num_samples=num_samples_per_class,
                class_label=class_idx,
                device=device
            )
            all_images.append(images)

        # 拼接所有图像
        all_images = torch.cat(all_images, dim=0)

        # 保存网格图像
        grid = make_grid(all_images, nrow=num_samples_per_class, normalize=True, value_range=(-1, 1))
        save_path = self.samples_dir / f"samples_epoch_{epoch}.png"
        save_image(grid, save_path)
        print(f"  样本已保存到: {save_path}")

        model.train()

    @torch.no_grad()
    def sample(self, model, num_samples, class_label, device, cfg_scale=None):
        """
        使用DDIM采样生成图像

        Args:
            model: LDM模型
            num_samples: 生成数量
            class_label: 类别标签（int）
            device: 设备
            cfg_scale: CFG强度，None则使用默认值
        """
        if cfg_scale is None:
            cfg_scale = self.args.cfg_scale

        # 准备类别标签
        class_labels = torch.full(
            (num_samples,), class_label,
            dtype=torch.long, device=device
        )

        # 获取条件和无条件嵌入
        cond_embeddings = model.class_embedder(class_labels, train=False)
        uncond_embeddings = model.class_embedder.get_null_embedding(num_samples, device)

        # 初始化随机噪声
        latent_size = self.args.image_size // model.vae_scale_factor
        latents = torch.randn(
            num_samples, 4, latent_size, latent_size,
            device=device
        )

        # 设置推理调度器
        self.inference_scheduler.set_timesteps(self.args.num_inference_steps, device=device)

        # DDIM采样循环
        for t in self.inference_scheduler.timesteps:
            # 扩展时间步
            timestep = t.expand(num_samples)

            # CFG: 同时预测条件和无条件噪声
            latent_input = torch.cat([latents, latents], dim=0)
            timestep_input = torch.cat([timestep, timestep], dim=0)
            embedding_input = torch.cat([cond_embeddings, uncond_embeddings], dim=0)

            # 预测噪声
            noise_pred = model.unet(
                latent_input,
                timestep_input,
                encoder_hidden_states=embedding_input
            ).sample

            # 分离条件和无条件预测
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)

            # 应用CFG
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            # DDIM步进
            latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample

        # 解码latent到图像
        images = model.decode_latents(latents)

        return images

    def save_checkpoint(self, name):
        """保存checkpoint"""
        model = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            "unet": model.unet.state_dict(),
            "class_embedder": model.class_embedder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "config": vars(self.args)
        }

        save_path = self.output_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint已保存到: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        print(f"从 {checkpoint_path} 恢复训练...")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model = self.accelerator.unwrap_model(self.model)
        model.unet.load_state_dict(checkpoint["unet"])
        model.class_embedder.load_state_dict(checkpoint["class_embedder"])

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
