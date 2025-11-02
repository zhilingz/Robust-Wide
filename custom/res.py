# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse

def calculate_residual(img1, img2):
    """
    计算两张图片的残差，并进行归一化（标准化到0-255范围，参考 residual_image = (residual_abs - residual_abs_min) / (residual_abs_max - residual_abs_min)）
    """
    # 确保两张图片尺寸一致
    if img1.shape != img2.shape:
        raise ValueError("Both images must have the same dimensions")
    
    # 计算绝对残差
    residual_abs = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    residual_abs_max = np.max(residual_abs)
    residual_abs_min = np.min(residual_abs)
    # 防止分母为0
    if residual_abs_max - residual_abs_min < 1e-8:
        residual_image = np.zeros_like(residual_abs)
    else:
        residual_image = (residual_abs - residual_abs_min) / (residual_abs_max - residual_abs_min)
    # 标准化到0-255并转为uint8
    residual_image = (residual_image * 255).clip(0, 255).astype(np.uint8)
    return residual_image

def load_and_process_images(folder_path):
    """
    Load four images from folder and calculate residuals
    """
    # Define image filenames
    image_files = {
        'original': 'image.png',
        'watermarked': 'wm_image.png',
        'generated_before': 'generated_image_before_wm.png',
        'generated_after': 'generated_image.png'
    }
    
    # Load images
    images = {}
    for key, filename in image_files.items():
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load image using PIL and convert to numpy array
        img = Image.open(filepath)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        images[key] = np.array(img)
    
    # Calculate residuals
    residual_before = calculate_residual(images['original'], images['watermarked'])
    residual_after = calculate_residual(images['generated_before'], images['generated_after'])
    
    return images, residual_before, residual_after

def create_combined_image(images, residual_before, residual_after, output_path):
    """
    Create 2x3 combined image layout
    """
    # Set image size and layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # First row: original, residual, watermarked
    axes[0, 0].imshow(images['original'])
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(residual_before, cmap='hot')
    axes[0, 1].set_title('Residual Image', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(images['watermarked'])
    axes[0, 2].set_title('Watermarked Image', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Second row: generated before, residual, generated after
    axes[1, 0].imshow(images['generated_before'])
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(residual_after, cmap='hot')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(images['generated_after'])
    axes[1, 2].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'Before Edit', rotation=90, fontsize=14, fontweight='bold', 
             verticalalignment='center')
    fig.text(0.02, 0.25, 'After Edit', rotation=90, fontsize=14, fontweight='bold', 
             verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)
    
    # Save image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined image saved to: {output_path}")

def process_folder(folder_path, output_filename='residual_comparison.png'):
    """
    Process images in specified folder
    """
    try:
        # Check if folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        print(f"Processing folder: {folder_path}")
        
        # Load images and calculate residuals
        images, residual_before, residual_after = load_and_process_images(folder_path)
        
        # Generate output path
        output_path = os.path.join(folder_path, output_filename)
        
        # Create combined image
        create_combined_image(images, residual_before, residual_after, output_path)
        
        print("Processing completed!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

def main():
    """
    Main function to handle command line arguments
    """
    parser = argparse.ArgumentParser(description='Calculate image residuals and generate combined image')
    parser.add_argument('-f', '--folder', type=str, required=True, 
                       help='Path to folder containing four images')
    parser.add_argument('-o', '--output', type=str, default='residual_comparison.png',
                       help='Output filename (default: residual_comparison.png)')
    
    args = parser.parse_args()
    
    # Process images
    process_folder(args.folder, args.output)

if __name__ == "__main__":
    main()
