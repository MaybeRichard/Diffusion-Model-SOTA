"""
ControlNet模型配置

基于Hugging Face Diffusers实现的ControlNet
使用掩码作为控制条件引导图像生成
"""

import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler
)
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from typing import Optional, Tuple


def create_vae(pretrained_model: str = "stabilityai/sd-vae-ft-mse") -> AutoencoderKL:
    """
    加载预训练VAE

    Args:
        pretrained_model: 预训练模型名称或路径

    Returns:
        vae: 预训练VAE模型（冻结）
    """
    vae = AutoencoderKL.from_pretrained(pretrained_model, torch_dtype=torch.float32)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def create_unet(
    in_channels: int = 4,
    out_channels: int = 4,
    cross_attention_dim: int = 768,
    block_out_channels: tuple = (128, 256, 512, 512),
    layers_per_block: int = 2,
    attention_head_dim: int = 8,
    sample_size: int = 32
) -> UNet2DConditionModel:
    """
    创建UNet模型

    注意：在ControlNet训练中，UNet通常是冻结的，
    只训练ControlNet分支
    """
    unet = UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
        ),
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim,
        norm_num_groups=32,
        act_fn="silu",
    )
    return unet


def create_controlnet(
    in_channels: int = 4,
    conditioning_channels: int = 3,  # 掩码输入通道数
    cross_attention_dim: int = 768,
    block_out_channels: tuple = (128, 256, 512, 512),
    layers_per_block: int = 2,
    attention_head_dim: int = 8,
) -> ControlNetModel:
    """
    创建ControlNet模型

    ControlNet复制UNet的encoder部分，
    将条件信息（掩码）注入到UNet的各层
    """
    controlnet = ControlNetModel(
        in_channels=in_channels,
        conditioning_channels=conditioning_channels,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ),
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim,
        norm_num_groups=32,
        act_fn="silu",
        conditioning_embedding_out_channels=(16, 32, 96, 256),
    )
    return controlnet


def create_controlnet_from_unet(
    unet: UNet2DConditionModel,
    conditioning_channels: int = 3
) -> ControlNetModel:
    """
    从已有UNet创建ControlNet（共享encoder权重初始化）
    """
    controlnet = ControlNetModel.from_unet(
        unet,
        conditioning_channels=conditioning_channels
    )
    return controlnet


def create_scheduler(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    prediction_type: str = "epsilon"
) -> DDPMScheduler:
    """创建DDPM噪声调度器"""
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
        clip_sample=False
    )
    return scheduler


def create_inference_scheduler(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear"
) -> UniPCMultistepScheduler:
    """创建UniPC调度器用于快速推理"""
    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        prediction_type="epsilon"
    )
    return scheduler


class DummyEncoder(nn.Module):
    """
    虚拟文本编码器

    由于我们不使用文本条件，创建一个固定的embedding
    作为cross-attention的输入
    """

    def __init__(self, embed_dim: int = 768, seq_len: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        # 可学习的固定embedding
        self.embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        """返回扩展后的embedding [batch_size, seq_len, embed_dim]"""
        return self.embedding.expand(batch_size, -1, -1)


class ControlNetLDM(nn.Module):
    """
    ControlNet Latent Diffusion Model

    组件：
    - VAE: 预训练的图像编码器/解码器（冻结）
    - UNet: 噪声预测网络（可选冻结或微调）
    - ControlNet: 条件控制网络（训练）
    - DummyEncoder: 固定的cross-attention输入

    推荐训练流程（针对OCT等专业领域）：
    1. 先使用ldm项目训练类别条件LDM
    2. 使用from_ldm_checkpoint()加载训练好的UNet权重
    3. 冻结UNet，只训练ControlNet
    """

    def __init__(
        self,
        vae_pretrained: str = "stabilityai/sd-vae-ft-mse",
        embed_dim: int = 768,
        conditioning_channels: int = 3,
        freeze_unet: bool = True,
        unet_config: dict = None,
        controlnet_config: dict = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.freeze_unet = freeze_unet

        # 加载预训练VAE（冻结）
        self.vae = create_vae(vae_pretrained)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # 创建UNet
        unet_config = unet_config or {}
        unet_config["cross_attention_dim"] = embed_dim
        self.unet = create_unet(**unet_config)

        # 冻结UNet
        if freeze_unet:
            self.unet.requires_grad_(False)
            self.unet.eval()

        # 创建ControlNet（从UNet初始化）
        self.controlnet = create_controlnet_from_unet(
            self.unet,
            conditioning_channels=conditioning_channels
        )

        # 创建虚拟编码器
        self.dummy_encoder = DummyEncoder(embed_dim=embed_dim)

        # 创建噪声调度器
        self.scheduler = create_scheduler()

    @classmethod
    def from_ldm_checkpoint(
        cls,
        ldm_checkpoint_path: str,
        vae_pretrained: str = "stabilityai/sd-vae-ft-mse",
        conditioning_channels: int = 3,
        freeze_unet: bool = True
    ):
        """
        从训练好的LDM checkpoint创建ControlNet模型

        这是OCT等专业领域图像的推荐方式：
        1. 先训练LDM学习领域图像的生成分布
        2. 加载LDM的UNet权重
        3. 从UNet初始化ControlNet
        4. 冻结UNet，只训练ControlNet

        Args:
            ldm_checkpoint_path: LDM checkpoint路径（来自ldm项目）
            vae_pretrained: VAE模型
            conditioning_channels: 掩码通道数
            freeze_unet: 是否冻结UNet

        Returns:
            model: 加载了预训练UNet的ControlNetLDM
        """
        print(f"从LDM checkpoint加载: {ldm_checkpoint_path}")

        # 加载LDM checkpoint
        checkpoint = torch.load(ldm_checkpoint_path, map_location="cpu", weights_only=False)

        # 获取配置
        ldm_config = checkpoint.get("config", {})
        embed_dim = ldm_config.get("embed_dim", 768)
        num_classes = checkpoint.get("num_classes", 8)
        class_names = checkpoint.get("class_names", [])

        print(f"  LDM配置: embed_dim={embed_dim}, num_classes={num_classes}")
        print(f"  类别: {class_names}")

        # 创建模型实例
        model = cls(
            vae_pretrained=vae_pretrained,
            embed_dim=embed_dim,
            conditioning_channels=conditioning_channels,
            freeze_unet=freeze_unet
        )

        # 加载UNet权重
        print("  加载UNet权重...")
        model.unet.load_state_dict(checkpoint["unet"])

        # 重新从加载权重后的UNet创建ControlNet（继承UNet的encoder权重）
        print("  从UNet初始化ControlNet...")
        model.controlnet = create_controlnet_from_unet(
            model.unet,
            conditioning_channels=conditioning_channels
        )

        # 冻结UNet
        if freeze_unet:
            model.unet.requires_grad_(False)
            model.unet.eval()
            print("  UNet已冻结")

        print("  加载完成!")
        return model

    def load_ldm_unet(self, ldm_checkpoint_path: str):
        """
        加载LDM checkpoint中的UNet权重（实例方法版本）

        Args:
            ldm_checkpoint_path: LDM checkpoint路径
        """
        print(f"加载LDM UNet权重: {ldm_checkpoint_path}")

        checkpoint = torch.load(ldm_checkpoint_path, map_location="cpu", weights_only=False)
        self.unet.load_state_dict(checkpoint["unet"])

        # 重新从UNet初始化ControlNet
        conditioning_channels = self.controlnet.config.conditioning_channels
        self.controlnet = create_controlnet_from_unet(
            self.unet,
            conditioning_channels=conditioning_channels
        )

        if self.freeze_unet:
            self.unet.requires_grad_(False)
            self.unet.eval()

        print("  UNet权重已加载，ControlNet已重新初始化")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """使用VAE编码图像到latent空间"""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """使用VAE解码latent到图像空间"""
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        return images

    def forward(
        self,
        images: torch.Tensor,
        conditioning: torch.Tensor,
        noise: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        controlnet_conditioning_scale: float = 1.0
    ) -> torch.Tensor:
        """
        训练前向传播

        Args:
            images: 目标图像 [B, 3, H, W]
            conditioning: 控制条件（掩码）[B, 3, H, W]
            noise: 可选的噪声
            timesteps: 可选的时间步
            controlnet_conditioning_scale: ControlNet强度

        Returns:
            loss: MSE损失
        """
        batch_size = images.shape[0]
        device = images.device

        # 编码图像到latent空间
        latents = self.encode_images(images)

        # 生成噪声
        if noise is None:
            noise = torch.randn_like(latents)

        # 采样随机时间步
        if timesteps is None:
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (batch_size,), device=device, dtype=torch.long
            )

        # 添加噪声
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # 获取dummy encoder hidden states
        encoder_hidden_states = self.dummy_encoder(batch_size).to(device)

        # ControlNet前向传播
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=conditioning,
            return_dict=False
        )

        # 应用controlnet_conditioning_scale
        down_block_res_samples = [
            sample * controlnet_conditioning_scale
            for sample in down_block_res_samples
        ]
        mid_block_res_sample = mid_block_res_sample * controlnet_conditioning_scale

        # UNet前向传播
        # 注意：即使freeze_unet=True，也不能使用torch.no_grad()
        # 因为需要让梯度流过ControlNet的输出(down_block_res_samples)
        # UNet参数已设置requires_grad=False，不会被更新
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        ).sample

        # 计算MSE损失
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return loss

    def get_trainable_parameters(self):
        """获取需要训练的参数"""
        params = list(self.controlnet.parameters()) + list(self.dummy_encoder.parameters())

        if not self.freeze_unet:
            params += list(self.unet.parameters())

        return params


if __name__ == "__main__":
    # 测试模型创建
    print("创建ControlNet模型...")
    model = ControlNetLDM(
        embed_dim=768,
        conditioning_channels=3,
        freeze_unet=True
    )

    print(f"VAE参数量: {sum(p.numel() for p in model.vae.parameters()) / 1e6:.2f}M (冻结)")
    print(f"UNet参数量: {sum(p.numel() for p in model.unet.parameters()) / 1e6:.2f}M (冻结)")
    print(f"ControlNet参数量: {sum(p.numel() for p in model.controlnet.parameters()) / 1e6:.2f}M")
    print(f"DummyEncoder参数量: {sum(p.numel() for p in model.dummy_encoder.parameters()) / 1e6:.4f}M")

    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"可训练参数总量: {trainable_params / 1e6:.2f}M")

    # 测试前向传播
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dummy_images = torch.randn(2, 3, 256, 256).to(device)
    dummy_masks = torch.randn(2, 3, 256, 256).to(device)

    loss = model(dummy_images, dummy_masks)
    print(f"测试loss: {loss.item():.4f}")
