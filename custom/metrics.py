import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms
from kornia.metrics import psnr, ssim
import json
import argparse
from itertools import combinations
import pandas as pd
from tqdm import tqdm

class ImageMetricsCalculator:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

    def _to_tensor(self, image):
        """统一转换为设备上的tensor"""
        if isinstance(image, PILImage.Image):
            tensor = transforms.ToTensor()(image) * 2 - 1
        else:
            tensor = image
        
        tensor = tensor.to(self.device)
        return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor

    def calculate_metrics(self, image1, image2):
        """计算两张图片之间的PSNR、SSIM、L1、L2指标"""
        img1_tensor = self._to_tensor(image1)
        img2_tensor = self._to_tensor(image2)
        
        # 转换到[0,1]用于PSNR和SSIM
        img1_01 = (img1_tensor + 1) / 2
        img2_01 = (img2_tensor + 1) / 2
        
        metrics = {
            'psnr': psnr(img2_01, img1_01, max_val=1.0).item(),
            'l1': F.l1_loss(img2_tensor, img1_tensor).item(),
            'l2': F.mse_loss(img2_tensor, img1_tensor).item(),
        }
        
        # SSIM计算
        ssim_val = ssim(img2_01, img1_tensor, window_size=5)
        metrics['ssim'] = torch.mean(ssim_val).item() if ssim_val.dim() > 0 else ssim_val.item()
        
        return metrics

    def load_image(self, image_path):
        """加载图片并调整大小"""
        try:
            image = PILImage.open(image_path).convert('RGB')
            # 调整图片大小到512x512以保持一致性
            image = image.resize((512, 512), PILImage.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            return None

    def get_image_files(self, directory):
        """获取目录下所有图片文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory, filename))
        
        return sorted(image_files)

    def calculate_pairwise_metrics(self, image_directory, output_file=None):
        """计算目录下所有图片两两之间的指标"""
        # 获取所有图片文件
        image_files = self.get_image_files(image_directory)
        
        if len(image_files) < 2:
            print(f"目录 {image_directory} 中的图片数量少于2张，无法进行两两比较")
            return None
        
        print(f"找到 {len(image_files)} 张图片")
        print("图片列表:")
        for i, img_path in enumerate(image_files):
            print(f"  {i+1}: {os.path.basename(img_path)}")
        
        # 预加载所有图片
        print("\n正在加载图片...")
        images = {}
        for img_path in tqdm(image_files, desc="加载图片"):
            img = self.load_image(img_path)
            if img is not None:
                images[img_path] = img
        
        print(f"成功加载 {len(images)} 张图片")
        
        # 计算所有图片对的指标
        results = []
        image_pairs = list(combinations(images.keys(), 2))
        total_pairs = len(image_pairs)
        
        print(f"\n开始计算 {total_pairs} 对图片的指标...")
        
        with torch.no_grad():
            for img1_path, img2_path in tqdm(image_pairs, desc="计算指标"):
                img1 = images[img1_path]
                img2 = images[img2_path]
                
                try:
                    metrics = self.calculate_metrics(img1, img2)
                    
                    result = {
                        'image1': os.path.basename(img1_path),
                        'image2': os.path.basename(img2_path),
                        'image1_path': img1_path,
                        'image2_path': img2_path,
                        'psnr': metrics['psnr'],
                        'ssim': metrics['ssim'],
                        'l1': metrics['l1'],
                        'l2': metrics['l2']
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"计算指标失败 {os.path.basename(img1_path)} vs {os.path.basename(img2_path)}: {e}")
        
        # 计算统计信息
        if results:
            df = pd.DataFrame(results)
            stats = {
                'total_pairs': len(results),
                'psnr_mean': df['psnr'].mean(),
                'psnr_std': df['psnr'].std(),
                'psnr_min': df['psnr'].min(),
                'psnr_max': df['psnr'].max(),
                'ssim_mean': df['ssim'].mean(),
                'ssim_std': df['ssim'].std(),
                'ssim_min': df['ssim'].min(),
                'ssim_max': df['ssim'].max(),
                'l1_mean': df['l1'].mean(),
                'l1_std': df['l1'].std(),
                'l1_min': df['l1'].min(),
                'l1_max': df['l1'].max(),
                'l2_mean': df['l2'].mean(),
                'l2_std': df['l2'].std(),
                'l2_min': df['l2'].min(),
                'l2_max': df['l2'].max(),
            }
            
            print("\n=" * 50)
            print("统计结果:")
            print(f"总图片对数: {stats['total_pairs']}")
            print(f"PSNR - 均值: {stats['psnr_mean']:.4f}, 标准差: {stats['psnr_std']:.4f}, 范围: [{stats['psnr_min']:.4f}, {stats['psnr_max']:.4f}]")
            print(f"SSIM - 均值: {stats['ssim_mean']:.4f}, 标准差: {stats['ssim_std']:.4f}, 范围: [{stats['ssim_min']:.4f}, {stats['ssim_max']:.4f}]")
            print(f"L1   - 均值: {stats['l1_mean']:.4f}, 标准差: {stats['l1_std']:.4f}, 范围: [{stats['l1_min']:.4f}, {stats['l1_max']:.4f}]")
            print(f"L2   - 均值: {stats['l2_mean']:.4f}, 标准差: {stats['l2_std']:.4f}, 范围: [{stats['l2_min']:.4f}, {stats['l2_max']:.4f}]")
            print("=" * 50)
            
            # 保存结果
            if output_file:
                # 保存详细结果
                output_data = {
                    'statistics': stats,
                    'pairwise_results': results
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # 同时保存CSV格式便于查看
                csv_file = output_file.replace('.json', '.csv')
                df.to_csv(csv_file, index=False, encoding='utf-8')
                
                print(f"\n结果已保存到:")
                print(f"  JSON格式: {output_file}")
                print(f"  CSV格式: {csv_file}")
            
            return results, stats
        
        else:
            print("没有成功计算任何图片对的指标")
            return None, None

def main():
    parser = argparse.ArgumentParser(description='计算图片文件夹中所有图片两两之间的PSNR、SSIM、L1、L2指标')
    parser.add_argument('--image_dir', type=str, required=True, help='包含图片的目录路径')
    parser.add_argument('--output_file', type=str, default=None, help='输出结果文件路径 (JSON格式)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='计算设备')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.image_dir):
        print(f"错误: 目录 {args.image_dir} 不存在")
        return
    
    if not os.path.isdir(args.image_dir):
        print(f"错误: {args.image_dir} 不是一个目录")
        return
    
    # 设置输出文件
    if args.output_file is None:
        dir_name = os.path.basename(args.image_dir.rstrip('/'))
        args.output_file = f"pairwise_metrics_{dir_name}.json"
    
    # 创建计算器并执行
    calculator = ImageMetricsCalculator(device=args.device)
    results, stats = calculator.calculate_pairwise_metrics(args.image_dir, args.output_file)
    
    if results:
        print(f"\n计算完成! 共计算了 {len(results)} 对图片的指标")
    else:
        print("\n计算失败或无有效结果")

if __name__ == "__main__":
    main()