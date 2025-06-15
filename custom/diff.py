#!/usr/bin/env python3
# img_diff.py - Simplified version based on train.py implementation
import cv2
import numpy as np
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from kornia.metrics import psnr, ssim

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two images using multiple metrics and highlight edited regions."
    )
    p.add_argument("-b", "--before",  default="inference_results/timbrooks___instructpix2pix-clip-filtered/FLUX.1-Fill-dev_2.png",
                   help="Path to original image")
    p.add_argument("-a", "--after",  default="inference_results/timbrooks___instructpix2pix-clip-filtered/original_2.png",
                   help="Path to edited image")
    p.add_argument("-o", "--output", default=None,
                   help="Filename of saved result (default: auto-generated based on --after path)")
    p.add_argument("-t", "--thresh", type=int, default=15,
                   help="Pixel-value threshold for change detection (default: 15)")
    p.add_argument("-k", "--kernel", type=int, default=3,
                   help="Kernel size for morphological opening (default: 3)")
    p.add_argument("-m", "--method", default="edit_ratio",
                   choices=["edit_ratio", "psnr", "ssim", "l1", "l2"],
                   help="Difference calculation method (default: edit_ratio)")
    p.add_argument("--save_metrics", action="store_true",
                   help="Save metrics to JSON file")
    return p

def generate_output_path(after_path, suffix="_diff"):
    """根据after路径生成输出路径"""
    after_path = Path(after_path)
    stem = after_path.stem
    ext = after_path.suffix
    new_filename = f"{stem}{suffix}{ext}"
    return after_path.parent / new_filename

def load_image_as_tensor(image_path):
    """加载图像并转换为PyTorch张量，值域[-1, 1]"""
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return tensor

def diff_image(before, after, method='edit_ratio', thresh=15, kernel=3):
    """
    使用多种方法计算两张图像的差异
    
    Args:
        before: 编辑前图像，形状为 (B, C, H, W) 或 (C, H, W)，值域 [-1, 1]
        after: 编辑后图像，形状为 (B, C, H, W) 或 (C, H, W)，值域 [-1, 1]
        method: 差异计算方法，可选：
               'edit_ratio' - 原始的编辑区域占比方法
               'psnr' - 峰值信噪比 (越高越好)
               'ssim' - 结构相似性指数 (越高越好)
               'l1' - L1距离/MAE (越低越好)
               'l2' - L2距离/MSE (越低越好)
        thresh: 像素差异阈值 (仅用于edit_ratio方法)
        kernel: 形态学开运算核大小 (仅用于edit_ratio方法)
    
    Returns:
        如果method='all'，返回包含所有指标的字典和mask
        否则返回对应指标的张量，形状为 (B,) 或标量
    """
    # 确保输入是4维张量 (B, C, H, W)
    if before.dim() == 3:
        before = before.unsqueeze(0)
        after = after.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = before.shape[0]
    device = before.device
    
    # 将值域从[-1,1]转换到[0,1]用于PSNR和SSIM计算
    before_01 = (before + 1) / 2
    after_01 = (after + 1) / 2
    
    results = {}
    mask = None
    
    if method == 'psnr' or method == 'all':
        # 计算PSNR (Peak Signal-to-Noise Ratio)
        # 值越高表示图像质量越好，差异越小
        psnr_values = []
        for i in range(batch_size):
            psnr_val = psnr(after_01[i:i+1], before_01[i:i+1], max_val=1.0)
            psnr_values.append(psnr_val.item())
        results['psnr'] = torch.tensor(psnr_values, device=device)
    
    if method == 'ssim' or method == 'all':
        # 计算SSIM (Structural Similarity Index)
        # 值越高表示结构相似性越好，差异越小
        ssim_values = []
        for i in range(batch_size):
            ssim_val = torch.mean(ssim(after_01[i:i+1], before_01[i:i+1], window_size=5))
            ssim_values.append(ssim_val.item())
        results['ssim'] = torch.tensor(ssim_values, device=device)
    
    if method == 'l1' or method == 'all':
        # 计算L1距离 (Mean Absolute Error)
        # 值越低表示差异越小
        l1_values = F.l1_loss(after, before, reduction='none')
        l1_values = l1_values.view(batch_size, -1).mean(dim=1)
        results['l1'] = l1_values
    
    if method == 'l2' or method == 'all':
        # 计算L2距离 (Mean Squared Error)
        # 值越低表示差异越小
        l2_values = F.mse_loss(after, before, reduction='none')
        l2_values = l2_values.view(batch_size, -1).mean(dim=1)
        results['l2'] = l2_values
    
    
    if method == 'edit_ratio' or method == 'all':
        # 原始的编辑区域占比方法
        # 转换到 [0, 255] 范围并转为numpy
        before_np = ((before + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
        after_np = ((after + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
        
        ratios = []
        masks = []
        for i in range(batch_size):
            # 获取单张图像，转换为 HWC 格式
            img_before = before_np[i].transpose(1, 2, 0)  # CHW -> HWC
            img_after = after_np[i].transpose(1, 2, 0)    # CHW -> HWC
            
            # 计算每个通道的差异
            diff_b = np.abs(img_after[:,:,0].astype(np.int16) - img_before[:,:,0].astype(np.int16))
            diff_g = np.abs(img_after[:,:,1].astype(np.int16) - img_before[:,:,1].astype(np.int16))
            diff_r = np.abs(img_after[:,:,2].astype(np.int16) - img_before[:,:,2].astype(np.int16))
            
            # 合并三个通道的差异，取最大值
            color_diff_magnitude = np.maximum(np.maximum(diff_b, diff_g), diff_r).astype(np.uint8)
            
            # 二值化
            _, current_mask = cv2.threshold(color_diff_magnitude, thresh, 255, cv2.THRESH_BINARY)
            
            # 形态学开运算去除小杂点
            kernel_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
            current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel_elem, iterations=1)
            
            # 计算占比
            ratio = current_mask.sum() / 255 / current_mask.size
            ratios.append(ratio)
            masks.append(current_mask)
        
        results['edit_ratio'] = torch.tensor(ratios, device=device)
        # 返回第一个mask用于可视化
        mask = masks[0] if masks else None
    
    # 返回结果
    if method == 'all':
        if squeeze_output:
            return {k: v.squeeze() if v.dim() > 0 else v for k, v in results.items()}, mask
        return results, mask
    else:
        if method in results:
            result = results[method]
            if squeeze_output:
                return result.squeeze(), mask
            return result, mask
        else:
            print(f"Warning: 方法 '{method}' 不可用")
            return None, None

def main():
    args = build_parser().parse_args()
    before_path, after_path = Path(args.before), Path(args.after)
    
    # 生成输出路径
    if args.output is None:
        output_path = generate_output_path(after_path)
    else:
        output_path = Path(args.output)

    # ── 读取图像 ───────────────────────────────────
    try:
        before_tensor = load_image_as_tensor(before_path)
        after_tensor = load_image_as_tensor(after_path)
        
        # 调整张量大小以匹配
        h, w = before_tensor.shape[2], before_tensor.shape[3]
        after_tensor = F.interpolate(after_tensor, size=(h, w), mode='bilinear', align_corners=False)
        
    except Exception as e:
        print(f"Error: 无法加载图像: {e}")
        return

    # ── 计算所有指标 ─────────────────────────────────
    all_metrics, mask = diff_image(before_tensor, after_tensor, method='all', thresh=args.thresh, kernel=args.kernel)
    
    if all_metrics is None:
        print("Error: 无法计算指标")
        return
    
    # 转换为标量值用于显示
    display_metrics = {}
    for key, value in all_metrics.items():
        if isinstance(value, torch.Tensor):
            display_metrics[key] = value.item()
        else:
            display_metrics[key] = value

    # ── 输出结果 ──────────────────────────────────────
    print("\n" + "="*60)
    print("图像差异分析结果:")
    print("="*60)
    
    if 'psnr' in display_metrics:
        print(f"PSNR (峰值信噪比):     {display_metrics['psnr']:.2f} dB")
    if 'ssim' in display_metrics:
        print(f"SSIM (结构相似性):     {display_metrics['ssim']:.6f}")
    if 'l1' in display_metrics:
        print(f"L1距离 (平均绝对误差): {display_metrics['l1']:.6f}")
        print(f"L2距离 (均方误差):     {display_metrics['l2']:.6f}")
    if 'edit_ratio' in display_metrics:
        print(f"编辑区域占比:          {display_metrics['edit_ratio']:.4f} ({display_metrics['edit_ratio']*100:.2f}%)")
    print("="*60)
    
    # ── 保存指标到JSON ────────────────────────────────
    if args.save_metrics:
        metrics_path = generate_output_path(output_path, "_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(display_metrics, f, indent=2, ensure_ascii=False)
        print(f"指标已保存至: {metrics_path}")
    
    # ── 生成对比图像 ────────────────────────────────
    # 为了生成可视化，需要转换为OpenCV格式
    before_cv = ((before_tensor.squeeze(0).permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
    after_cv = ((after_tensor.squeeze(0).permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
    
    # 转换为BGR格式
    before_bgr = cv2.cvtColor(before_cv, cv2.COLOR_RGB2BGR)
    after_bgr = cv2.cvtColor(after_cv, cv2.COLOR_RGB2BGR)
    
    # 生成最终的对比图像：原图 | 编辑后 | 黑白mask
    if mask is not None:
        # 将单通道mask转换为三通道以便拼接
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        concat = cv2.hconcat([before_bgr, after_bgr, mask_bgr])
    else:
        concat = cv2.hconcat([before_bgr, after_bgr])
    
    cv2.imwrite(str(output_path), concat)
    print(f"对比图已保存至: {output_path}")
    
    # ── 根据指定方法返回特定指标 ──────────────────────
    if args.method != "all":
        if args.method in display_metrics:
            print(f"\n指定方法 '{args.method}' 的结果: {display_metrics[args.method]}")
        else:
            print(f"\nWarning: 方法 '{args.method}' 不可用")

if __name__ == "__main__":
    main()
