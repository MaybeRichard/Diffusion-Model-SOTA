#!/bin/bash
# DDPM Training Script

# ============ Unconditional DDPM ============
# Train unconditional DDPM on your dataset
python ddpm.py
# Note: Modify ddpm.py launch() function to set:
#   - args.dataset_path: path to your ImageFolder dataset
#   - args.image_size: image size (e.g., 64, 128, 256)
#   - args.batch_size: batch size
#   - args.epochs: number of training epochs

# ============ Conditional DDPM (with class labels) ============
# Train class-conditional DDPM
python ddpm_conditional.py
# Note: Modify ddpm_conditional.py launch() function to set:
#   - args.dataset_path: path to your ImageFolder dataset
#   - args.num_classes: number of classes
#   - args.image_size: image size
#   - args.batch_size: batch size
#   - args.epochs: number of training epochs

# ============ Custom Training Example ============
# You can also import and call train() directly:
#
# from ddpm import train
# import argparse
#
# args = argparse.Namespace(
#     run_name="my_experiment",
#     epochs=500,
#     batch_size=16,
#     image_size=64,
#     dataset_path="/path/to/dataset",
#     device="cuda",
#     lr=3e-4
# )
# train(args)

# ============ For Conditional Training ============
# from ddpm_conditional import train
# import argparse
#
# args = argparse.Namespace(
#     run_name="my_conditional_experiment",
#     epochs=300,
#     batch_size=16,
#     image_size=64,
#     num_classes=10,
#     dataset_path="/path/to/dataset",
#     device="cuda",
#     lr=3e-4
# )
# train(args)
