#!/usr/bin/env python3
"""
训练集编辑后指标统计脚本
用于统计训练集中图像编辑后的各种指标（PSNR、SSIM、L1、L2、edit_ratio）
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from kornia.metrics import psnr, ssim
from accelerate import Accelerator
from accelerate.utils import set_seed

from custom.custom_insp2p import CustomStableDiffusionInstructPix2PixPipeline
from dataset import get_hugging_dataset, get_filtered_dataset, collate_fn
from utils import denormalize


def calculate_edit_metrics(before, after, thresh=15, kernel=3):
    """
    计算两张图像编辑后的各种指标
    
    Args:
        before: 编辑前图像，形状为 (B, C, H, W) 或 (C, H, W)，值域 [-1, 1]
        after: 编辑后图像，形状为 (B, C, H, W) 或 (C, H, W)，值域 [-1, 1]
        thresh: 像素差异阈值
        kernel: 形态学开运算核大小
    
    Returns:
        dict: 包含各种指标的字典
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
    
    # 确保两个张量在同一设备上
    if before.device != after.device:
        after = after.to(before.device)
    
    # 将值域从[-1,1]转换到[0,1]用于PSNR和SSIM计算
    before_01 = (before + 1) / 2
    after_01 = (after + 1) / 2
    
    results = {}
    
    # 计算PSNR (Peak Signal-to-Noise Ratio)
    psnr_values = []
    for i in range(batch_size):
        # 为PSNR计算转换为float32
        before_01_float = before_01[i:i+1].float()
        after_01_float = after_01[i:i+1].float()
        psnr_val = psnr(after_01_float, before_01_float, max_val=1.0)
        psnr_values.append(psnr_val.item())
    results['psnr'] = torch.tensor(psnr_values, device=device)
    
    # 计算SSIM (Structural Similarity Index)
    ssim_values = []
    for i in range(batch_size):
        # 转换为float32以支持SSIM计算
        before_01_float = before_01[i:i+1].float()
        after_01_float = after_01[i:i+1].float()
        ssim_val = torch.mean(ssim(after_01_float, before_01_float, window_size=5))
        ssim_values.append(ssim_val.item())
    results['ssim'] = torch.tensor(ssim_values, device=device)
    
    # 计算L1距离 (Mean Absolute Error)
    l1_values = F.l1_loss(after, before, reduction='none')
    l1_values = l1_values.view(batch_size, -1).mean(dim=1)
    results['l1'] = l1_values
    
    # 计算L2距离 (Mean Squared Error)
    l2_values = F.mse_loss(after, before, reduction='none')
    l2_values = l2_values.view(batch_size, -1).mean(dim=1)
    results['l2'] = l2_values
    
    # 计算编辑区域占比
    before_np = ((before + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
    after_np = ((after + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
    
    ratios = []
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
        _, mask = cv2.threshold(color_diff_magnitude, thresh, 255, cv2.THRESH_BINARY)
        
        # 形态学开运算去除小杂点
        kernel_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_elem, iterations=1)
        
        # 计算占比
        ratio = mask.sum() / 255 / mask.size
        ratios.append(ratio)
    
    results['edit_ratio'] = torch.tensor(ratios, device=device)
    
    # 如果输入是3维的，压缩输出
    if squeeze_output:
        for key in results:
            results[key] = results[key].squeeze()
    
    return results


def initialize_pipeline(model_dir, weight_dtype, device):
    """初始化生成模型"""
    if "instruct-pix2pix" in model_dir or "magicbrush" in model_dir:
        pipe = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_dir, torch_dtype=weight_dtype, local_files_only=True
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_dir}")
    
    # 设置调度器
    if "magicbrush" in model_dir:
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # 设置为评估模式
    pipe.text_encoder.eval()
    pipe.unet.eval()
    pipe.vae.eval()
    
    return pipe

def generate_image(pipe, prompt, image, device, seed=42):
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    
    # 设置默认种子
    if seed is None:
        seed = getattr(args, 'generation_seed', 42)  # 使用args中的种子，默认为42
    
    # 创建生成器，每次生成图片都重置Generator，避免每调用一次随机函数，内部状态就前进一次。
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # 根据模型选择不同的pipe参数
    if "instruct-pix2pix-distill" in args.model_dir:
        generated_image = pipe(
            prompt, 
            image=image, 
            num_images_per_prompt=1, 
            num_inference_steps=4,  # 4改为2，测试效果
            guidance_scale=2.0,     # 使用较小的guidance_scale
            image_guidance_scale=1.0,  # 使用较小的image_guidance_scale
            generator=generator,
            output_type="pt"
        )
    elif "sd-turbo" in args.model_dir:
        generated_image = pipe(
            prompt, 
            image=image, 
            num_images_per_prompt=1, 
            num_inference_steps=2,
            guidance_scale=0.0, 
            strength=0.5,
            generator=generator,
            output_type="pt"
        ).images     # pipe返回值为StableDiffusionPipelineOutput 类型，需要取images，形状 (1, C, H, W)
        generated_image = 2 * generated_image - 1 # 将值域从[-1,1]转为[0,1]，防止图片泛白
    elif "magicbrush" in args.model_dir:
        generated_image = pipe(
            prompt, 
            image=image, 
            num_inference_steps=20, 
            image_guidance_scale=2.0, 
            guidance_scale=4, 
            generator=generator,
            output_type="pt",
            )
    else:
        generated_image = pipe(
            prompt, 
            image=image, 
            num_images_per_prompt=1, 
            num_inference_steps=20,
            guidance_scale=10, 
            image_guidance_scale=1.5, # 1.5原图保留太少（62.98%的编辑区域），2.0还可以（38.45%的编辑区域）
            generator=generator,
            output_type="pt",
        )
    return generated_image

def save_sample_images(original, edited, prompt, output_dir, sample_idx):
    """保存样例图像"""
    sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 保存原图
    save_image(denormalize(original[0].detach().cpu()), 
               os.path.join(sample_dir, "original.png"))
    
    # 保存编辑后的图像
    save_image(denormalize(edited[0].detach().cpu()), 
               os.path.join(sample_dir, "edited.png"))
    
    # 保存prompt
    with open(os.path.join(sample_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        if isinstance(prompt, list):
            f.write(str(prompt[0]))
        else:
            f.write(str(prompt))


def plot_metrics_distribution(metrics_data, output_dir):
    """绘制指标分布图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_names = ['psnr', 'ssim', 'l1', 'l2', 'edit_ratio']
    
    for i, metric_name in enumerate(metrics_names):
        if metric_name in metrics_data:
            data = metrics_data[metric_name]
            axes[i].hist(data, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric_name.upper()} Distribution')
            axes[i].set_xlabel(metric_name.upper())
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = np.mean(data)
            std_val = np.std(data)
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.3f}')
            axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', 
                          label=f'Mean+Std: {mean_val + std_val:.3f}')
            axes[i].axvline(mean_val - std_val, color='orange', linestyle='--', 
                          label=f'Mean-Std: {mean_val - std_val:.3f}')
            axes[i].legend()
    
    # 隐藏多余的子图
    for i in range(len(metrics_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    if args.seed is not None:
        set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if args.use_fp16 else torch.float32
    
    # 初始化生成模型
    print("正在初始化生成模型...")
    pipe = initialize_pipeline(args.model_dir, weight_dtype, device)
    
    # 加载数据集
    print("正在加载数据集...")
    if args.use_filtered_dataset:
        train_dataset, _ = get_filtered_dataset(
            args, args.image_size, None, args.train_size, 0
        )
    else:
        train_dataset, _ = get_hugging_dataset(
            args.train_data_dir, args.image_size, None, args.train_size, 0
        )
    
    # 使用训练集
    dataset = train_dataset
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集所有指标
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'l1': [],
        'l2': [],
        'edit_ratio': []
    }
    
    sample_count = 0
    
    print("开始处理数据集...")
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Processing")):
            image, prompt = data["image"], data["prompt"]
            image = image.to(device)
            # 生成编辑后的图像
            edited_image = generate_image(pipe, prompt, image, device, args.seed)
            
            # 计算指标
            metrics = calculate_edit_metrics(image, edited_image)
            
            # 收集指标
            for key in all_metrics.keys():
                if key in metrics:
                    if isinstance(metrics[key], torch.Tensor):
                        all_metrics[key].extend(metrics[key].cpu().numpy().tolist())
                    else:
                        all_metrics[key].append(metrics[key])
            
            # 保存前几个样例
            if sample_count < args.save_samples and args.save_samples > 0:
                save_sample_images(image, edited_image, prompt, args.output_dir, sample_count)
            
            sample_count += len(image)
            
    
    # 计算统计信息
    print("正在计算统计信息...")
    stats = {}
    for metric_name, values in all_metrics.items():
        if values:
            stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'count': len(values)
            }
    
    # 保存统计结果
    stats_file = os.path.join(args.output_dir, 'metrics_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 保存原始数据
    raw_data_file = os.path.join(args.output_dir, 'raw_metrics_data.json')
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    # 绘制分布图
    print("正在绘制分布图...")
    plot_metrics_distribution(all_metrics, args.output_dir)
    
    # 打印统计信息
    print("\n=== 训练集编辑后指标统计 ===")
    print(f"总处理样本数: {sample_count}")
    for metric_name, stat in stats.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  平均值: {stat['mean']:.4f}")
        print(f"  标准差: {stat['std']:.4f}")
        print(f"  最小值: {stat['min']:.4f}")
        print(f"  最大值: {stat['max']:.4f}")
        print(f"  中位数: {stat['median']:.4f}")
        print(f"  25%分位数: {stat['q25']:.4f}")
        print(f"  75%分位数: {stat['q75']:.4f}")
    
    print(f"\n结果已保存到: {args.output_dir}")
    print(f"统计信息: {stats_file}")
    print(f"原始数据: {raw_data_file}")
    print(f"分布图: {os.path.join(args.output_dir, 'metrics_distribution.png')}")
    
    if args.save_samples > 0:
        print(f"样例图像已保存到子目录中")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计训练集编辑后的指标")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练数据目录")
    parser.add_argument("--model_dir", type=str, required=True, help="生成模型目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--image_size", type=int, default=512, help="图像尺寸")
    parser.add_argument("--train_size", type=int, default=10000, help="训练集大小")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--save_samples", type=int, default=10, help="保存样例图像数量")
    parser.add_argument("--use_filtered_dataset", action="store_true", help="使用过滤后的数据集")
    parser.add_argument("--use_fp16", action="store_true", help="使用FP16精度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    main(args) 