import os
import glob
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# 引入 pytorch_fid 的核心函数
try:
    from pytorch_fid import fid_score
except ImportError:
    print("错误: 未找到 pytorch_fid 库。请先运行 'pip install pytorch-fid'")
    exit(1)

# ==========================================
# 第一部分：VGG16 特征提取模型 (用于 P&R)
# ==========================================

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的 VGG16
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # 保留到 fc2 层 (classifier[4])
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:5])
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 第二部分：数据加载器 (用于 P&R)
# ==========================================

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*")))
        # 兼容更多格式
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'))]
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {folder_path}")
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # VGG 输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        # 强制转 RGB，适配 VGG 输入
        img = Image.open(img_path).convert('RGB') 
        img = self.transform(img)
        return img

def extract_features_vgg(folder_path, model, device, batch_size=32):
    dataset = ImageFolderDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    print(f"Extracting VGG features from: {folder_path}...")
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            feats = model(batch) # 输出 [B, 4096]
            all_features.append(feats.cpu())
            
    return torch.cat(all_features, dim=0)

# ==========================================
# 第三部分：Precision & Recall 计算核心
# ==========================================

def compute_distances(row_features, col_features):
    return torch.cdist(row_features, col_features)

def compute_pr(real_features, gen_features, k=3, batch_size=10000, device='cuda'):
    print(f"Computing Precision & Recall (k={k})...")
    
    real_features = real_features.to(device).to(torch.float32)
    gen_features = gen_features.to(device).to(torch.float32)

    def compute_manifold_radii(features):
        num_items = features.shape[0]
        kth_values = []
        for i in range(0, num_items, batch_size):
            batch = features[i : i + batch_size]
            dists = compute_distances(batch, features)
            kth_values.append(dists.kthvalue(k + 1, dim=1).values)
        return torch.cat(kth_values)

    def compute_overlap(probes, manifold_features, manifold_radii):
        num_probes = probes.shape[0]
        is_overlap = []
        for i in range(0, num_probes, batch_size):
            probe_batch = probes[i : i + batch_size]
            dists = compute_distances(probe_batch, manifold_features)
            in_manifold = (dists <= manifold_radii.unsqueeze(0)).any(dim=1)
            is_overlap.append(in_manifold)
        return torch.cat(is_overlap).float().mean().item()

    print("  Estimating Real Manifold radii...")
    real_radii = compute_manifold_radii(real_features)
    
    print("  Estimating Generated Manifold radii...")
    gen_radii = compute_manifold_radii(gen_features)

    print("  Calculating Precision...")
    precision = compute_overlap(gen_features, real_features, real_radii)

    print("  Calculating Recall...")
    recall = compute_overlap(real_features, gen_features, gen_radii)

    return precision, recall

# ==========================================
# 第四部分：FID 计算 (封装 pytorch_fid)
# ==========================================

def compute_fid_standard(real_path, gen_path, batch_size, device):
    """
    使用官方 pytorch_fid 库计算 Standard FID。
    这等价于运行: python -m pytorch_fid path1 path2
    """
    print("Computing FID (Standard InceptionV3)...")
    
    # 确保 device 是 pytorch_fid 接受的格式 (通常是 torch.device 对象)
    
    # dims=2048 是标准 FID 的设置
    # num_workers 可以根据 CPU 核心数调整
    fid_value = fid_score.calculate_fid_given_paths(
        paths=[real_path, gen_path],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=4
    )
    return fid_value

# ==========================================
# 主程序入口
# ==========================================

if __name__ == "__main__":
    # --- 配置区域 ---
    REAL_IMAGES_DIR = "/data2/sichengli/Code/RAE/real_sample"
    GEN_IMAGES_DIR = "/data2/sichengli/Code/RAE/samples_8class_all/DiTwDDTHead-0275000-cfg-1.50-bs16-ODE-50-euler-bf16"
    BATCH_SIZE = 32
    # ----------------
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    results = {}

    # -------------------------------------------------
    # 步骤 1: 计算 Precision 和 Recall (基于 VGG16)
    # -------------------------------------------------
    try:
        print(">>> Step 1: Calculating Precision & Recall (VGG16)")
        vgg_extractor = VGG16FeatureExtractor().to(device)
        
        # 提取 VGG 特征
        real_feats_vgg = extract_features_vgg(REAL_IMAGES_DIR, vgg_extractor, device, BATCH_SIZE)
        gen_feats_vgg = extract_features_vgg(GEN_IMAGES_DIR, vgg_extractor, device, BATCH_SIZE)
        
        # 计算 P&R
        precision, recall = compute_pr(real_feats_vgg, gen_feats_vgg, k=3, device=device)
        
        results['Precision'] = precision
        results['Recall'] = recall
        
        # 清理显存：VGG16 用完了，可以释放掉，为 InceptionV3 腾地方
        del vgg_extractor
        del real_feats_vgg
        del gen_feats_vgg
        torch.cuda.empty_cache()
        print("VGG16 resources released.\n")
        
    except Exception as e:
        print(f"Error computing P&R: {e}")

    # -------------------------------------------------
    # 步骤 2: 计算 FID (基于 InceptionV3)
    # -------------------------------------------------
    try:
        print(">>> Step 2: Calculating FID (InceptionV3)")
        # 直接调用 pytorch_fid 的 API
        fid = compute_fid_standard(REAL_IMAGES_DIR, GEN_IMAGES_DIR, BATCH_SIZE, device)
        
        results['FID'] = fid
        
    except Exception as e:
        print(f"Error computing FID: {e}")

# -------------------------------------------------
    # 输出最终报表
    # -------------------------------------------------
    print("\n" + "="*40)
    print(f"FINAL EVALUATION REPORT")
    print(f"Real Data: {os.path.basename(REAL_IMAGES_DIR)}")
    print(f"Gen Data:  {os.path.basename(GEN_IMAGES_DIR)}")
    print("="*40)
    
    if 'FID' in results:
        print(f"FID (InceptionV3):     {results['FID']:.4f}")
        
    if 'Precision' in results:
        print(f"Precision (VGG16):     {results['Precision']:.4f}")
        
    if 'Recall' in results:
        print(f"Recall (VGG16):        {results['Recall']:.4f}")
        
    print("="*40)