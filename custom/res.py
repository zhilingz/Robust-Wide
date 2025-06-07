import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def calculate_residual(img1, img2):
    """
    计算两张图片的残差
    """
    # 确保两张图片尺寸相同
    if img1.shape != img2.shape:
        raise ValueError("两张图片的尺寸必须相同")
    
    # 计算残差并归一化到0-255范围
    residual = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    residual = np.clip(residual, 0, 255).astype(np.uint8)
    
    return residual

def load_and_process_images(folder_path):
    """
    加载文件夹中的四张图片并计算残差
    """
    # 定义图片文件名
    image_files = {
        'original': 'image.png',
        'watermarked': 'wm_image.png',
        'generated_before': 'generated_image_before_wm.png',
        'generated_after': 'generated_image.png'
    }
    
    # 加载图片
    images = {}
    for key, filename in image_files.items():
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到文件: {filepath}")
        
        # 使用PIL加载图片并转换为numpy数组
        img = Image.open(filepath)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        images[key] = np.array(img)
    
    # 计算残差
    residual_before = calculate_residual(images['original'], images['watermarked'])
    residual_after = calculate_residual(images['generated_before'], images['generated_after'])
    
    return images, residual_before, residual_after

def create_combined_image(images, residual_before, residual_after, output_path):
    """
    创建2行3列的组合图片
    """
    # 设置图片大小和布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：原图、残差、水印图
    axes[0, 0].imshow(images['original'])
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(residual_before, cmap='hot')
    axes[0, 1].set_title('Residual Image', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(images['watermarked'])
    axes[0, 2].set_title('Watermarked Image', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 第二行：编辑前生成图、残差、编辑后生成图
    axes[1, 0].imshow(images['generated_before'])
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(residual_after, cmap='hot')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(images['generated_after'])
    axes[1, 2].axis('off')
    
    # 添加行标签
    fig.text(0.02, 0.75, 'Before Edit', rotation=90, fontsize=14, fontweight='bold', 
             verticalalignment='center')
    fig.text(0.02, 0.25, 'After Edit', rotation=90, fontsize=14, fontweight='bold', 
             verticalalignment='center')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"组合图片已保存到: {output_path}")

def process_folder(folder_path, output_filename='combined_result.png'):
    """
    处理指定文件夹中的图片
    """
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        print(f"正在处理文件夹: {folder_path}")
        
        # 加载图片和计算残差
        images, residual_before, residual_after = load_and_process_images(folder_path)
        
        # 生成输出路径
        output_path = os.path.join(folder_path, output_filename)
        
        # 创建组合图片
        create_combined_image(images, residual_before, residual_after, output_path)
        
        print("处理完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 请替换为您的文件夹路径
    folder_path = input("请输入包含图片的文件夹路径: ").strip()
    
    # 处理图片
    process_folder(folder_path)
