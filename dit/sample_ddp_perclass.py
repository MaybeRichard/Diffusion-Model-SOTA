# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from collections import defaultdict


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples (including subfolders).
    """
    samples = []
    count = 0
    # 遍历所有子文件夹收集图片
    for root, dirs, files in os.walk(sample_dir):
        for file in sorted(files):
            if file.endswith('.png') and count < num:
                sample_pil = Image.open(os.path.join(root, file))
                sample_np = np.asarray(sample_pil).astype(np.uint8)
                samples.append(sample_np)
                count += 1
    
    if count < num:
        print(f"Warning: Only found {count} images, expected {num}")
    
    samples = np.stack(samples)
    assert samples.shape == (len(samples), samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    
    # Determine class labels to use
    if args.class_labels is not None:
        available_classes = args.class_labels
    else:
        available_classes = list(range(args.num_classes))
    
    # 为每个类别创建子文件夹
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        for cls in available_classes:
            class_folder = os.path.join(sample_folder_dir, f"class_{cls}")
            os.makedirs(class_folder, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
        print(f"Created {len(available_classes)} class subfolders: {available_classes}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    
    # 每个类别的计数器（每个进程独立维护）
    class_counters = defaultdict(int)

    for iter_idx in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        # Evenly distribute across available classes
        global_start = (iter_idx * global_batch_size) + (rank * n)
        y_labels = [available_classes[(global_start + i) % len(available_classes)] for i in range(n)]
        y = torch.tensor(y_labels, device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files in class subfolders
        for i, (sample, label) in enumerate(zip(samples, y_labels)):
            # 使用全局索引确保文件名唯一
            global_idx = iter_idx * global_batch_size + rank * n + i
            class_folder = os.path.join(sample_folder_dir, f"class_{label}")
            Image.fromarray(sample).save(f"{class_folder}/{global_idx:06d}.png")

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    
    if rank == 0:
        # 统计每个类别的图片数量
        print("\nSamples per class:")
        for cls in available_classes:
            class_folder = os.path.join(sample_folder_dir, f"class_{cls}")
            num_images = len([f for f in os.listdir(class_folder) if f.endswith('.png')])
            print(f"  Class {cls}: {num_images} images")
        
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=8000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--class-labels", type=int, nargs='+', default=None,
                        help="List of class labels to generate. If not specified, evenly distributes across all classes.")
    args = parser.parse_args()
    main(args)


'''
  # 生成 50000 张图片，均匀分布到所有类别，按类别保存到子文件夹
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=7 sample_ddp_perclass.py --ckpt /data2/sichengli/Code/ldm/DiT/results/002-DiT-B-2/checkpoints/0160000.pt

  # 只在类别 0,1,2 之间均匀分布
  torchrun --nnodes=1 --nproc_per_node=4 sample_ddp.py --ckpt your_model.pt --class-labels 0 1 2

  # 输出文件夹结构示例:
  # samples/DiT-B-2-xxx/
  # ├── class_0/
  # │   ├── 000000.png
  # │   ├── 000008.png
  # │   └── ...
  # ├── class_1/
  # │   ├── 000001.png
  # │   └── ...
  # └── class_7/
  #     └── ...
'''