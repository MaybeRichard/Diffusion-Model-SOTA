"""
类别条件LDM模型配置
使用Hugging Face Diffusers库，将类别标签注入UNet的cross-attention
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.models.embeddings import TimestepEmbedding


class ClassEmbedder(nn.Module):
    """
    类别嵌入模块
    将离散类别标签转换为连续嵌入向量，用于注入UNet的cross-attention
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 768,
        dropout_prob: float = 0.1
    ):
        """
        Args:
            num_classes: 类别数量
            embed_dim: 嵌入维度（需要与UNet的cross_attention_dim一致）
            dropout_prob: Classifier-Free Guidance的dropout概率
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob

        # 类别嵌入层（+1用于unconditional/null类别）
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)

        # null类别的索引
        self.null_class_idx = num_classes

    def forward(self, class_labels: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Args:
            class_labels: 类别标签 [batch_size]
            train: 是否为训练模式（用于CFG dropout）

        Returns:
            class_embeddings: [batch_size, 1, embed_dim] 用于cross-attention
        """
        batch_size = class_labels.shape[0]

        # 训练时随机dropout类别标签（用于Classifier-Free Guidance）
        if train and self.dropout_prob > 0:
            mask = torch.rand(batch_size, device=class_labels.device) < self.dropout_prob
            class_labels = torch.where(
                mask,
                torch.full_like(class_labels, self.null_class_idx),
                class_labels
            )

        # 获取嵌入 [batch_size, embed_dim]
        embeddings = self.embedding(class_labels)

        # 扩展为序列格式 [batch_size, 1, embed_dim] 用于cross-attention
        embeddings = embeddings.unsqueeze(1)

        return embeddings

    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """获取unconditional嵌入（用于CFG推理）"""
        null_labels = torch.full(
            (batch_size,), self.null_class_idx,
            dtype=torch.long, device=device
        )
        return self.embedding(null_labels).unsqueeze(1)


def create_vae(pretrained_model: str = "stabilityai/sd-vae-ft-mse") -> AutoencoderKL:
    """
    加载预训练VAE

    Args:
        pretrained_model: 预训练模型名称或路径

    Returns:
        vae: 预训练VAE模型
    """
    vae = AutoencoderKL.from_pretrained(pretrained_model, torch_dtype=torch.float32)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def create_unet(
    in_channels: int = 4,  # VAE latent通道数
    out_channels: int = 4,
    cross_attention_dim: int = 768,  # 类别嵌入维度
    block_out_channels: tuple = (128, 256, 512, 512),
    layers_per_block: int = 2,
    attention_head_dim: int = 8,
    sample_size: int = 32  # 256/8=32 for latent size
) -> UNet2DConditionModel:
    """
    创建用于类别条件生成的UNet

    关键配置：
    - cross_attention_dim: 用于接收类别嵌入
    - 使用较小的模型以适应医学图像数据集规模
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
) -> DDIMScheduler:
    """创建DDIM调度器用于快速推理"""
    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type="epsilon"
    )
    return scheduler


class ClassConditionedLDM(nn.Module):
    """
    类别条件Latent Diffusion Model包装类

    组件：
    - VAE: 预训练的图像编码器/解码器（冻结）
    - UNet: 噪声预测网络（训练）
    - ClassEmbedder: 类别嵌入模块（训练）
    """

    def __init__(
        self,
        num_classes: int,
        vae_pretrained: str = "stabilityai/sd-vae-ft-mse",
        embed_dim: int = 768,
        cfg_dropout: float = 0.1,
        unet_config: dict = None
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 加载预训练VAE（冻结）
        self.vae = create_vae(vae_pretrained)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # 创建UNet
        unet_config = unet_config or {}
        unet_config["cross_attention_dim"] = embed_dim
        self.unet = create_unet(**unet_config)

        # 创建类别嵌入器
        self.class_embedder = ClassEmbedder(
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout_prob=cfg_dropout
        )

        # 创建噪声调度器
        self.scheduler = create_scheduler()

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
        class_labels: torch.Tensor,
        noise: torch.Tensor = None,
        timesteps: torch.Tensor = None
    ) -> torch.Tensor:
        """
        训练前向传播

        Args:
            images: 原始图像 [B, 3, H, W]
            class_labels: 类别标签 [B]
            noise: 可选的噪声
            timesteps: 可选的时间步

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

        # 获取类别嵌入（训练模式，有CFG dropout）
        class_embeddings = self.class_embedder(class_labels, train=True)

        # 预测噪声
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=class_embeddings
        ).sample

        # 计算MSE损失
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return loss

    def get_trainable_parameters(self):
        """获取需要训练的参数（UNet + ClassEmbedder）"""
        return list(self.unet.parameters()) + list(self.class_embedder.parameters())


if __name__ == "__main__":
    # 测试模型创建
    print("创建模型...")
    model = ClassConditionedLDM(
        num_classes=8,
        embed_dim=768,
        cfg_dropout=0.1
    )

    print(f"VAE参数量: {sum(p.numel() for p in model.vae.parameters()) / 1e6:.2f}M (冻结)")
    print(f"UNet参数量: {sum(p.numel() for p in model.unet.parameters()) / 1e6:.2f}M")
    print(f"ClassEmbedder参数量: {sum(p.numel() for p in model.class_embedder.parameters()) / 1e6:.4f}M")

    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"可训练参数总量: {trainable_params / 1e6:.2f}M")

    # 测试前向传播
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dummy_images = torch.randn(2, 3, 256, 256).to(device)
    dummy_labels = torch.tensor([0, 1]).to(device)

    loss = model(dummy_images, dummy_labels)
    print(f"测试loss: {loss.item():.4f}")
