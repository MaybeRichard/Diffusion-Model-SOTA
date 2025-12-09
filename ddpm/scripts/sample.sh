#!/bin/bash
# DDPM Sampling Script
# Note: DDPM uses inline inference. Below are example Python scripts.

# ============ Unconditional Sampling ============
python -c "
import torch
from modules import UNet, count_parameters
from ddpm import Diffusion

device = 'cuda'
img_size = 64  # Must match training size

# Load model
model = UNet(img_size=img_size).to(device)
ckpt = torch.load('./models/DDPM_Uncondtional/ckpt.pt', weights_only=False)
model.load_state_dict(ckpt)

# Print model info
total, trainable = count_parameters(model)
print(f'Total parameters: {total:,}')
print(f'Trainable parameters: {trainable:,}')

# Sample
diffusion = Diffusion(img_size=img_size, device=device)
samples = diffusion.sample(model, n=8, log_time=True)
print(f'Generated {samples.shape[0]} images of shape {samples.shape[1:]}')

# Save
from utils import save_images
save_images(samples, 'generated_samples.jpg')
print('Saved to generated_samples.jpg')
"

# ============ Conditional Sampling ============
# python -c "
# import torch
# from modules import UNet_conditional, count_parameters
# from ddpm_conditional import Diffusion
#
# device = 'cuda'
# img_size = 64
# num_classes = 10
#
# # Load model
# model = UNet_conditional(num_classes=num_classes, img_size=img_size).to(device)
# ckpt = torch.load('./models/DDPM_conditional/ckpt.pt', weights_only=False)
# model.load_state_dict(ckpt)
#
# # Print model info
# total, trainable = count_parameters(model)
# print(f'Total parameters: {total:,}')
# print(f'Trainable parameters: {trainable:,}')
#
# # Sample with class labels
# diffusion = Diffusion(img_size=img_size, device=device)
# n = 8
# labels = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]).long().to(device)  # Class labels
# samples = diffusion.sample(model, n=n, labels=labels, cfg_scale=3, log_time=True)
# print(f'Generated {samples.shape[0]} images of shape {samples.shape[1:]}')
#
# # Save
# from utils import save_images
# save_images(samples, 'generated_conditional_samples.jpg')
# print('Saved to generated_conditional_samples.jpg')
# "

# ============ Sample specific class ============
# python -c "
# import torch
# from modules import UNet_conditional, count_parameters
# from ddpm_conditional import Diffusion
# from utils import save_images
#
# device = 'cuda'
# img_size = 64
# num_classes = 10
# target_class = 5  # Change this to your desired class
#
# model = UNet_conditional(num_classes=num_classes, img_size=img_size).to(device)
# ckpt = torch.load('./models/DDPM_conditional/ckpt.pt', weights_only=False)
# model.load_state_dict(ckpt)
#
# diffusion = Diffusion(img_size=img_size, device=device)
# n = 16
# labels = torch.Tensor([target_class] * n).long().to(device)
# samples = diffusion.sample(model, n=n, labels=labels, cfg_scale=3, log_time=True)
# save_images(samples, f'class_{target_class}_samples.jpg')
# "
