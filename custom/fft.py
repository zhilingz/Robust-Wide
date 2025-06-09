import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse
import os
from scipy.fft import dctn

def mag_spectrum_fft(arr):
    """Calculate magnitude spectrum using FFT"""
    return 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(arr))) + 1)

def mag_spectrum_dct(arr):
    """Calculate magnitude spectrum using DCT"""
    # 使用2D DCT
    dct_result = dctn(arr, type=2, norm='ortho')
    return 20 * np.log(np.abs(dct_result) + 1)

def resize_to_same_size(images):
    """Resize all images to the same size (minimum dimensions)"""
    # Find minimum dimensions
    min_height = min(img.shape[0] for img in images)
    min_width = min(img.shape[1] for img in images)
    
    # Resize all images
    resized_images = []
    for img in images:
        if img.shape != (min_height, min_width):
            # Convert to PIL Image for resizing
            pil_img = Image.fromarray(img.astype(np.uint8))
            resized_pil = pil_img.resize((min_width, min_height), Image.Resampling.LANCZOS)
            resized_img = np.asarray(resized_pil, dtype=float)
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    
    return resized_images

def find_images_in_folder(folder_path):
    """Find images in the folder with fixed names"""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Fixed image names - 只需要原图和水印图
    image_names = ["image.png", "wm_image.png"]
    image_paths = []
    
    for name in image_names:
        image_path = folder / name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_paths.append(str(image_path))
    
    return image_paths[0], image_paths[1]  # original, watermarked

def main():
    parser = argparse.ArgumentParser(description="Calculate DCT and FFT for original and watermarked images")
    
    # Add folder parameter
    parser.add_argument("-f", "--folder", help="Folder containing images (if provided, individual image paths are ignored)")
    
    # Individual image parameters
    parser.add_argument("--image", default="image.png", help="Original image path (default: image.png)")
    parser.add_argument("--wm_image", default="wm_image.png", help="Watermarked image path (default: wm_image.png)")
    
    # Output parameter
    parser.add_argument("--output", help="Output image path (if not provided, will be auto-generated)")
    
    args = parser.parse_args()
    
    # Determine image paths and output path
    if args.folder:
        try:
            image_path, wm_image_path = find_images_in_folder(args.folder)
            if not args.output:
                output_path = str(Path(args.folder) / "dct_fft.png")
        except Exception as e:
            print(f"Error: {e}")
            return
    else:
        image_path = args.image
        wm_image_path = args.wm_image
        if not args.output:
            output_path = str(Path(wm_image_path).parent / (Path(wm_image_path).stem + "_dct_fft.png"))
    
    if args.output:
        output_path = args.output
    
    # Read images
    try:
        orig_rgb = Image.open(image_path).convert("RGB")
        wm_rgb = Image.open(wm_image_path).convert("RGB")
    except Exception as e:
        print(f"Error reading images: {e}")
        return
    
    # Convert to grayscale
    orig_gray = np.asarray(orig_rgb.convert("L"), dtype=float)
    wm_gray = np.asarray(wm_rgb.convert("L"), dtype=float)
    
    # Resize all images to the same size
    orig_gray, wm_gray = resize_to_same_size([orig_gray, wm_gray])
    
    # Calculate residual (difference between original and watermarked)
    residual_img = np.abs(orig_gray - wm_gray)
    
    # Calculate DCT and FFT of the residual
    residual_dct = mag_spectrum_dct(residual_img)
    residual_fft = mag_spectrum_fft(residual_img)
    
    # Create combined plot (2 rows: first row 3 images, second row 2 images)
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    
    fig.suptitle("Original, Residual, Watermarked Images and Residual DCT/FFT Analysis", fontsize=16, fontweight='bold')
    
    # First row: Original image, Residual, Watermarked image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_rgb)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis("off")
    
    ax2 = fig.add_subplot(gs[0, 1])
    im_residual = ax2.imshow(residual_img, cmap="gray")
    ax2.set_title("Residual (|Orig - WM|)", fontsize=12, fontweight='bold')
    ax2.axis("off")
    fig.colorbar(im_residual, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(wm_rgb)
    ax3.set_title("Watermarked Image", fontsize=12, fontweight='bold')
    ax3.axis("off")
    
    # Second row: DCT and FFT of residual (centered)
    ax4 = fig.add_subplot(gs[1, 0:2])  # Span first two columns
    im_dct = ax4.imshow(residual_dct, cmap="viridis")
    ax4.set_title("Residual DCT Spectrum", fontsize=12, fontweight='bold')
    ax4.axis("off")
    fig.colorbar(im_dct, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = fig.add_subplot(gs[1, 2])
    im_fft = ax5.imshow(residual_fft, cmap="viridis")
    ax5.set_title("Residual FFT Spectrum", fontsize=12, fontweight='bold')
    ax5.axis("off")
    fig.colorbar(im_fft, ax=ax5, fraction=0.046, pad=0.04)
    
    # Save the combined image
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"DCT和FFT分析图片生成成功，地址为：{output_path}")

if __name__ == "__main__":
    main()
