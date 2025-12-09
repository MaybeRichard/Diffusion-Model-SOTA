#!/bin/bash
# Evaluation Script for Diffusion Models
# Computes FID, Precision, and Recall

# ============ Configuration ============
# Modify these paths before running
REAL_IMAGES_DIR="/path/to/real/images"
GEN_IMAGES_DIR="/path/to/generated/images"

# ============ Run Evaluation ============
# Modify the paths in precision_recall.py:
#   REAL_IMAGES_DIR = "/path/to/real/images"
#   GEN_IMAGES_DIR = "/path/to/generated/images"
#   BATCH_SIZE = 32

python precision_recall.py

# ============ Alternative: Command-line FID only ============
# If you only need FID, you can use pytorch-fid directly:
# python -m pytorch_fid $REAL_IMAGES_DIR $GEN_IMAGES_DIR --batch-size 32

# ============ Expected Output ============
# ========================================
# FINAL EVALUATION REPORT
# Real Data: real_images
# Gen Data:  generated_images
# ========================================
# FID (InceptionV3):     XX.XXXX
# Precision (VGG16):     0.XXXX
# Recall (VGG16):        0.XXXX
# ========================================
