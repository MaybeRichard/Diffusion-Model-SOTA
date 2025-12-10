import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter
import math
import time
from PIL import Image
import argparse

from utils import *
from modules import UNet_conditional, EMA

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S"
)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3, progress=True):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)
            iterator = reversed(range(1, self.noise_steps))
            if progress:
                iterator = tqdm(iterator, position=0)
            
            for i in iterator:
                t = (torch.ones(n, device=self.device) * i).long()

                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_pred = model(x, t, None)
                    predicted_noise = uncond_pred + cfg_scale * (predicted_noise - uncond_pred)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def get_data_ddp(args, rank, world_size):
    """获取支持DDP的数据加载器"""
    from torchvision import transforms, datasets
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder(args.dataset_path, transform=transform)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, sampler


def save_checkpoint(args, epoch, model, ema_model, optimizer, scaler, rank=0):
    """
    保存完整的训练状态
    
    保存内容说明：
    - ckpt.pt: 模型权重，用于推理
    - ema_ckpt.pt: EMA模型权重，通常生成质量更稳定
    - optim.pt: 优化器状态（学习率、动量、Adam的m/v等）
    - scaler.pt: AMP的GradScaler状态
    - checkpoint.pt: 完整checkpoint，包含所有状态（推荐用于恢复训练）
    """
    if rank != 0:
        return
    
    save_dir = os.path.join("models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取模型权重（处理DDP包装）
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    # 分别保存各个组件（向后兼容）
    torch.save(model_state, os.path.join(save_dir, "ckpt.pt"))
    torch.save(ema_model.state_dict(), os.path.join(save_dir, "ema_ckpt.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optim.pt"))
    torch.save(scaler.state_dict(), os.path.join(save_dir, "scaler.pt"))
    
    # 保存完整checkpoint（推荐用于恢复训练）
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'ema_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'args': vars(args),  # 保存训练参数
    }
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pt"))
    
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(args, model, ema_model, optimizer, scaler, device, rank=0):
    """
    加载训练状态
    
    返回: start_epoch
    """
    save_dir = os.path.join("models", args.run_name)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    
    # 优先加载完整checkpoint
    if os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{device}")
        
        # 加载模型权重
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载EMA模型
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载scaler状态
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
        if rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")
            if 'args' in checkpoint:
                print(f"Original training args: {checkpoint['args']}")
        
        return start_epoch
    
    # 向后兼容：加载旧格式的分离文件
    elif os.path.exists(os.path.join(save_dir, "ckpt.pt")):
        if rank == 0:
            print("Loading from legacy checkpoint files...")
        
        map_location = f"cuda:{device}"
        
        # 加载模型
        model_state = torch.load(os.path.join(save_dir, "ckpt.pt"), map_location=map_location)
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        
        # 加载EMA模型
        if os.path.exists(os.path.join(save_dir, "ema_ckpt.pt")):
            ema_state = torch.load(os.path.join(save_dir, "ema_ckpt.pt"), map_location=map_location)
            ema_model.load_state_dict(ema_state)
        
        # 加载优化器
        if os.path.exists(os.path.join(save_dir, "optim.pt")):
            optim_state = torch.load(os.path.join(save_dir, "optim.pt"), map_location=map_location)
            optimizer.load_state_dict(optim_state)
        
        # 加载scaler
        if os.path.exists(os.path.join(save_dir, "scaler.pt")):
            scaler_state = torch.load(os.path.join(save_dir, "scaler.pt"), map_location=map_location)
            scaler.load_state_dict(scaler_state)
        
        # 尝试读取epoch
        start_epoch = 0
        if os.path.exists(os.path.join(save_dir, "epoch.txt")):
            with open(os.path.join(save_dir, "epoch.txt"), "r") as f:
                start_epoch = int(f.read().strip()) + 1
        
        if rank == 0:
            print(f"Resumed from legacy checkpoint, starting epoch {start_epoch}")
        
        return start_epoch
    
    else:
        if rank == 0:
            print("No checkpoint found, starting from scratch")
        return 0


def train_ddp(args):
    """DDP分布式训练"""
    # 初始化DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    # 设置随机种子
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    if rank == 0:
        setup_logging(args.run_name)
        print(f"Starting DDP training with {world_size} GPUs")
        print(f"Global batch size: {args.batch_size * world_size}")

    # 获取数据
    dataloader, sampler = get_data_ddp(args, rank, world_size)
    
    # ========== 关键：正确的模型创建顺序 ==========
    # 1. 创建模型
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    
    # 2. 创建EMA模型（在DDP包装之前）
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    # 3. DDP包装（必须添加 find_unused_parameters=True）
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    # ========== 关键部分结束 ==========
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    # AMP设置（使用新API避免警告）
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args, model, ema_model, optimizer, scaler, device, rank)
    
    # 只在rank 0创建tensorboard logger
    logger = None
    if rank == 0:
        logger = SummaryWriter(os.path.join("runs", args.run_name))
    
    l = len(dataloader)

    torch.backends.cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):
        # 设置sampler的epoch以确保每个epoch的shuffle不同
        sampler.set_epoch(epoch)
        
        if rank == 0:
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader

        model.train()
        epoch_loss = 0.0
        
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            if np.random.random() < 0.1:
                labels_in = None
            else:
                labels_in = labels

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                predicted_noise = model(x_t, t, labels_in)
                loss = mse(noise, predicted_noise)

            scaler.scale(loss).backward()

            if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # 更新EMA
            ema.step_ema(ema_model, model.module)

            epoch_loss += loss.item()
            
            if rank == 0:
                pbar.set_postfix(MSE=float(loss.item()))
                if logger is not None:
                    logger.add_scalar("MSE", float(loss.item()), global_step=epoch * l + i)

        # 计算平均loss并同步
        avg_loss = epoch_loss / len(dataloader)
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        
        if rank == 0:
            print(f"Epoch {epoch} average loss: {avg_loss_tensor.item():.6f}")

        # 保存和采样
        if epoch % args.save_interval == 0:
            # 保存checkpoint
            save_checkpoint(args, epoch, model, ema_model, optimizer, scaler, rank)
            
            # 采样（只在rank 0）
            if rank == 0:
                labels_vis = torch.arange(min(args.num_classes, 10)).long().to(device)
                
                model.eval()
                sampled_images = diffusion.sample(model.module, n=len(labels_vis), labels=labels_vis, 
                                                  cfg_scale=getattr(args, "cfg_scale", 3))
                ema_sampled_images = diffusion.sample(ema_model, n=len(labels_vis), labels=labels_vis, 
                                                       cfg_scale=getattr(args, "cfg_scale", 3))

                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
                save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))

        # 同步所有进程
        dist.barrier()

    if rank == 0 and logger is not None:
        logger.close()
    
    dist.destroy_process_group()


def train_single(args):
    """单卡训练"""
    setup_logging(args.run_name)

    device = args.device
    dataloader = get_data(args)

    model = UNet_conditional(num_classes=args.num_classes).to(device)
    
    # 创建EMA模型
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    use_amp = (device.startswith("cuda") and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        # 单卡模式下device index为0
        device_idx = 0 if device == "cuda" else int(device.split(":")[-1])
        start_epoch = load_checkpoint(args, model, ema_model, optimizer, scaler, device_idx, rank=0)

    torch.backends.cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            if np.random.random() < 0.1:
                labels_in = None
            else:
                labels_in = labels

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                predicted_noise = model(x_t, t, labels_in)
                loss = mse(noise, predicted_noise)

            scaler.scale(loss).backward()

            if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=float(loss.item()))
            logger.add_scalar("MSE", float(loss.item()), global_step=epoch * l + i)

        if epoch % args.save_interval == 0:
            # 保存checkpoint
            save_checkpoint(args, epoch, model, ema_model, optimizer, scaler, rank=0)
            
            # 采样
            labels_vis = torch.arange(min(args.num_classes, 10)).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels_vis), labels=labels_vis, 
                                              cfg_scale=getattr(args, "cfg_scale", 3))
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels_vis), labels=labels_vis, 
                                                   cfg_scale=getattr(args, "cfg_scale", 3))

            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))

    logger.close()


def get_model_path(args, rank=0):
    """
    获取模型路径，支持多种方式指定EMA模型
    
    优先级：
    1. 如果 --ckpt 直接指向 ema_ckpt.pt，直接使用
    2. 如果 --use-ema 且存在对应的 ema_ckpt.pt，使用EMA
    3. 否则使用 --ckpt 指定的模型
    """
    ckpt_path = args.ckpt
    
    # 如果已经指定了ema模型，直接返回
    if "ema" in os.path.basename(ckpt_path):
        if rank == 0:
            print(f"Using EMA model: {ckpt_path}")
        return ckpt_path
    
    # 如果指定了 --use-ema，尝试查找对应的ema模型
    if args.use_ema:
        # 尝试多种可能的EMA文件命名
        possible_ema_paths = [
            ckpt_path.replace("ckpt.pt", "ema_ckpt.pt"),
            ckpt_path.replace(".pt", "_ema.pt"),
            os.path.join(os.path.dirname(ckpt_path), "ema_ckpt.pt"),
        ]
        
        for ema_path in possible_ema_paths:
            if os.path.exists(ema_path):
                if rank == 0:
                    print(f"Using EMA model: {ema_path}")
                return ema_path
        
        if rank == 0:
            print(f"Warning: --use-ema specified but EMA model not found, using: {ckpt_path}")
    
    if rank == 0:
        print(f"Using model: {ckpt_path}")
    return ckpt_path


def sample_ddp(args):
    """多卡DDP采样"""
    assert torch.cuda.is_available(), "DDP采样需要GPU"
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = args.seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Starting DDP sampling with {world_size} GPUs")
        print(f"rank={rank}, seed={seed}, world_size={world_size}")

    # 加载模型
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    
    # 获取模型路径（支持EMA）
    ckpt_path = get_model_path(args, rank)
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{device}")
    model.load_state_dict(ckpt)
    model.eval()
    
    diffusion = Diffusion(img_size=args.image_size, device=device)

    if args.class_labels is not None:
        available_classes = args.class_labels
    else:
        available_classes = list(range(args.num_classes))

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        for cls in available_classes:
            class_folder = os.path.join(args.output_dir, f"class_{cls}")
            os.makedirs(class_folder, exist_ok=True)
        print(f"Saving samples to {args.output_dir}")
        print(f"Classes: {available_classes}")
    dist.barrier()

    n = args.batch_size
    global_batch_size = n * world_size
    total_samples = int(math.ceil(args.num_samples / global_batch_size) * global_batch_size)
    
    if rank == 0:
        print(f"Total images to generate: {total_samples}")
    
    samples_per_gpu = total_samples // world_size
    iterations = samples_per_gpu // n
    
    pbar = range(iterations)
    if rank == 0:
        pbar = tqdm(pbar, desc="Sampling")

    total_sampling_time = 0.0

    for iter_idx in pbar:
        global_start = (iter_idx * global_batch_size) + (rank * n)
        y_labels = [available_classes[(global_start + i) % len(available_classes)] for i in range(n)]
        y = torch.tensor(y_labels, device=device)

        start_time = time.time()
        samples = diffusion.sample(model, n, y, cfg_scale=args.cfg_scale, progress=False)
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        total_sampling_time += batch_time

        for i, (sample, label) in enumerate(zip(samples, y_labels)):
            global_idx = iter_idx * global_batch_size + rank * n + i
            img = sample.permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray(img)
            class_folder = os.path.join(args.output_dir, f"class_{label}")
            img.save(os.path.join(class_folder, f"{global_idx:06d}.png"))

    dist.barrier()

    total_time_tensor = torch.tensor([total_sampling_time], device=device)
    dist.reduce(total_time_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"\n{'='*50}")
        print("Samples per class:")
        total_saved = 0
        for cls in available_classes:
            class_folder = os.path.join(args.output_dir, f"class_{cls}")
            num_images = len([f for f in os.listdir(class_folder) if f.endswith('.png')])
            print(f"  Class {cls}: {num_images} images")
            total_saved += num_images
        
        avg_total_time = total_time_tensor.item() / world_size
        avg_time_per_image = avg_total_time / samples_per_gpu
        
        print(f"\n{'='*50}")
        print(f"Total samples saved: {total_saved}")
        print(f"Average sampling time per GPU: {avg_total_time:.2f}s")
        print(f"Average time per image: {avg_time_per_image:.2f}s")
        print(f"Throughput: {total_samples / avg_total_time:.2f} images/s")
        print(f"{'='*50}")

    dist.barrier()
    dist.destroy_process_group()


def sample_single(args):
    """单卡采样"""
    device = args.device
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    
    # 获取模型路径（支持EMA）
    ckpt_path = get_model_path(args, rank=0)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    
    diffusion = Diffusion(img_size=args.image_size, device=device)

    if args.class_labels is not None:
        available_classes = args.class_labels
    else:
        available_classes = list(range(args.num_classes))

    os.makedirs(args.output_dir, exist_ok=True)
    for cls in available_classes:
        class_folder = os.path.join(args.output_dir, f"class_{cls}")
        os.makedirs(class_folder, exist_ok=True)
    
    print(f"Saving samples to {args.output_dir}")

    n = args.num_samples
    y_labels = [available_classes[i % len(available_classes)] for i in range(n)]
    y = torch.tensor(y_labels, device=device)

    print(f"Generating {n} images...")

    start_time = time.time()
    x = diffusion.sample(model, n, y, cfg_scale=args.cfg_scale)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    for i in range(n):
        img = x[i].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        label = y_labels[i]
        class_folder = os.path.join(args.output_dir, f"class_{label}")
        img.save(os.path.join(class_folder, f"{i:06d}.png"))

    print(f"\n{'='*50}")
    print("Samples per class:")
    for cls in available_classes:
        class_folder = os.path.join(args.output_dir, f"class_{cls}")
        num_images = len([f for f in os.listdir(class_folder) if f.endswith('.png')])
        print(f"  Class {cls}: {num_images} images")

    avg_time = total_time / n
    print(f"\nTotal samples: {n}")
    print(f"Total sampling time: {total_time:.2f}s")
    print(f"Average time per image: {avg_time:.2f}s")
    print(f"{'='*50}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 模式选择
    parser.add_argument("--mode", type=str, 
                        choices=["train", "train_ddp", "sample", "sample_ddp"], 
                        default="sample",
                        help="运行模式: train/train_ddp/sample/sample_ddp")
    
    # 通用参数
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    
    # 训练参数
    parser.add_argument("--run-name", type=str, default="DDPM_conditional")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dataset-path", type=str, default="/data2/sichengli/Data/test/dataset/train")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="从checkpoint恢复训练")
    
    # 采样参数
    parser.add_argument("--ckpt", type=str, default="./models/DDPM_conditional/ckpt.pt")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--class-labels", type=int, nargs='+', default=None)
    parser.add_argument("--cfg-scale", type=float, default=0)
    parser.add_argument("--output-dir", type=str, default="samples")
    parser.add_argument("--use-ema", action="store_true", help="使用EMA模型进行采样")

    args = parser.parse_args()

    if args.mode == "train":
        train_single(args)
    elif args.mode == "train_ddp":
        train_ddp(args)
    elif args.mode == "sample_ddp":
        sample_ddp(args)
    else:
        sample_single(args)

'''
# 单卡恢复训练
python ddpm_conditional.py --mode train \
    --dataset-path /path/to/dataset \
    --resume

# 多卡DDP恢复训练
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 ddpm_conditional_ddp.py \
    --mode train_ddp \
    --dataset-path /data2/sichengli/Data/test/dataset/train \
    --resume

# EMA模型通常生成质量更稳定
# 多卡DDP采样，自动加载对应的EMA模型
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 ddpm_conditional_ddp.py \
    --mode sample_ddp \
    --ckpt ./models/DDPM_conditional/ckpt.pt \
    --use-ema \
    --num-samples 1000 \
    --batch-size 8
```

---

## 保存文件详解
```
models/DDPM_conditional/
├── ckpt.pt          # 模型权重 - 用于推理
├── ema_ckpt.pt      # EMA模型权重 - 推荐用于最终采样
├── optim.pt         # 优化器状态 - 包含Adam的动量等
├── scaler.pt        # AMP GradScaler状态
└── checkpoint.pt    # 完整checkpoint（推荐用于恢复训练）
'''