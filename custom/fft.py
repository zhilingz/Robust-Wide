import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse
import os

def mag_spectrum(arr):
    """Calculate magnitude spectrum"""
    return 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(arr))) + 1)

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
    """Find three images in the folder with fixed names"""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Fixed image names
    image_names = ["image.png", "wm_image.png", "generated_image.png"]
    image_paths = []
    
    for name in image_names:
        image_path = folder / name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_paths.append(str(image_path))
    
    return image_paths[0], image_paths[1], image_paths[2]  # original, watermarked, generated

def main():
    parser = argparse.ArgumentParser(description="Calculate magnitude spectra and residuals for three images")
    
    # Add folder parameter
    parser.add_argument("--folder", help="Folder containing three images (if provided, individual image paths are ignored)")
    
    # Individual image parameters
    parser.add_argument("--image", default="image.png", help="Original image path (default: image.png)")
    parser.add_argument("--wm_image", default="wm_image.png", help="Watermarked image path (default: wm_image.png)")
    parser.add_argument("--generated_image", default="generated_image.png", help="Generated image path (default: generated_image.png)")
    
    # Output parameter
    parser.add_argument("--output", help="Output image path (if not provided, will be auto-generated)")
    
    args = parser.parse_args()
    
    # Determine image paths and output path
    if args.folder:
        try:
            image_path, wm_image_path, gen_image_path = find_images_in_folder(args.folder)
            if not args.output:
                output_path = str(Path(args.folder) / "fft.png")
        except Exception as e:
            return
    else:
        image_path = args.image
        wm_image_path = args.wm_image
        gen_image_path = args.generated_image
        if not args.output:
            output_path = str(Path(gen_image_path).parent / (Path(gen_image_path).stem + "_fft.png"))
    
    if args.output:
        output_path = args.output
    
    # Read images
    try:
        orig_rgb = Image.open(image_path).convert("RGB")
        wm_rgb = Image.open(wm_image_path).convert("RGB")
        gen_rgb = Image.open(gen_image_path).convert("RGB")
    except Exception as e:
        return
    
    # Convert to grayscale
    orig_gray = np.asarray(orig_rgb.convert("L"), dtype=float)
    wm_gray = np.asarray(wm_rgb.convert("L"), dtype=float)
    gen_gray = np.asarray(gen_rgb.convert("L"), dtype=float)
    
    # Resize all images to the same size
    orig_gray, wm_gray, gen_gray = resize_to_same_size([orig_gray, wm_gray, gen_gray])
    
    # Calculate magnitude spectra
    mag_orig = mag_spectrum(orig_gray)
    mag_wm = mag_spectrum(wm_gray)
    mag_gen = mag_spectrum(gen_gray)
    
    # Calculate residuals
    residual_orig_wm = np.abs(mag_orig - mag_wm)
    residual_orig_gen = np.abs(mag_orig - mag_gen)
    residual_wm_gen = np.abs(mag_wm - mag_gen)
    
    # Create combined plot (3 rows x 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
    fig.suptitle("RGB Images, Magnitude Spectra and Residual Analysis", fontsize=18, fontweight='bold')
    
    # First row: RGB images
    rgb_images = [orig_rgb, wm_rgb, gen_rgb]
    rgb_titles = ["Original Image", "Watermarked Image", "Generated Image"]
    
    for i, (img, title) in enumerate(zip(rgb_images, rgb_titles)):
        axes[0, i].imshow(img)
        axes[0, i].set_title(title, fontsize=12, fontweight='bold')
        axes[0, i].axis("off")
    
    # Second row: Magnitude spectra
    spectra = [mag_orig, mag_wm, mag_gen]
    spec_titles = ["Original Spectrum", "Watermarked Spectrum", "Generated Spectrum"]
    
    for i, (spectrum, title) in enumerate(zip(spectra, spec_titles)):
        im = axes[1, i].imshow(spectrum, cmap="viridis")
        axes[1, i].set_title(title, fontsize=12, fontweight='bold')
        axes[1, i].axis("off")
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    # Third row: Residuals
    residuals = [residual_orig_wm, residual_orig_gen, residual_wm_gen]
    residual_titles = ["Original vs Watermarked", "Original vs Generated", "Watermarked vs Generated"]
    
    for i, (residual, title) in enumerate(zip(residuals, residual_titles)):
        im = axes[2, i].imshow(residual, cmap="viridis")
        axes[2, i].set_title(title, fontsize=12, fontweight='bold')
        axes[2, i].axis("off")
        fig.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    # Save the combined image
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"幅度谱图片生成成功，地址为：{output_path}")

if __name__ == "__main__":
    main()
