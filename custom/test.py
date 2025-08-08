#!/usr/bin/env python3
"""
Standalone test script for watermark model evaluation
Tests the watermark model on various distortion scenarios using the last 1200 samples from instruct-pix2pix dataset
"""

import os
import io
import argparse
import json
import logging
import numpy as np
import random
from functools import partial
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import kornia.filters as K
import kornia.enhance as E
from kornia.metrics import psnr, ssim

from datasets import load_dataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Import custom modules
import sys
sys.path.append('/public/zhangzhiling/code/Robust-Wide')

from model import WatermarkModel
from custom.custom_insp2p import CustomStableDiffusionInstructPix2PixPipeline
from dataset import preprocess_train, collate_fn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_wm_model(ckpt_dir, device="cuda"):
    """Load watermark model from checkpoint directory"""
    wm_model_config_path = os.path.join(ckpt_dir, "wm_model_config.yaml")
    if not os.path.exists(wm_model_config_path):
        raise FileNotFoundError(f"Model config not found: {wm_model_config_path}")
    
    wm_model_config = OmegaConf.load(wm_model_config_path)
    message_length = wm_model_config["wm_enc_config"]["message_length"]
    
    # Create model
    model = WatermarkModel(**wm_model_config, device=device, weight_dtype=torch.float32)
    
    # Load checkpoint
    model_ckpt_path = os.path.join(ckpt_dir, "wm_model.ckpt")
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")
    
    model_ckpt = torch.load(model_ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(model_ckpt)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded watermark model from {ckpt_dir}")
    logger.info(f"Message length: {message_length}")
    
    return model, message_length


def decoded_message_error_rate(message, decoded_message):
    """Calculate bit error rate for single message"""
    length = message.shape[0]
    message = message.gt(0.5)
    decoded_message = decoded_message.gt(0.5)
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    """Calculate bit error rate for batch of messages"""
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate


def denormalize(images):
    """Denormalize images from [-1,1] to [0,1]"""
    return (images / 2 + 0.5).clamp(0, 1)


def generate_image(pipe, prompt, wm_image, device, seed=42):
    """Generate edited image using instruct-pix2pix pipeline"""
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    generated_image = pipe(
        prompt,
        image=wm_image,
        num_images_per_prompt=1,
        num_inference_steps=20,
        guidance_scale=10,
        image_guidance_scale=1.5,
        generator=generator,
        output_type="pt",
    )
    
    return generated_image


def get_test_dataset(image_size=512, test_size=1200):
    """Load test dataset from the last 1200 samples of instruct-pix2pix dataset"""
    logger.info("Loading instruct-pix2pix dataset for testing")
    
    # Load instruct-pix2pix dataset
    dataset_path = "/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f"
    dataset = load_dataset(dataset_path)
    
    # Get total length
    total_samples = len(dataset["train"])
    logger.info(f"Total samples in instruct-pix2pix dataset: {total_samples}")
    
    # Check if we have enough samples
    if total_samples < test_size:
        raise ValueError(f"Dataset has only {total_samples} samples, but {test_size} requested")
    
    # Select last test_size samples
    test_start_idx = total_samples - test_size
    logger.info(f"Using test samples from index {test_start_idx} to {total_samples-1}")
    
    test_dataset = dataset["train"].select(range(test_start_idx, total_samples))
    
    # Apply preprocessing
    test_dataset = test_dataset.with_transform(partial(preprocess_train, image_size=image_size))
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return test_dataset


def apply_distortions(images, distortion_type):
    """Apply various distortions to images"""
    if distortion_type == "jpeg":
        # JPEG compression
        def apply_jpeg_compression(batch_images, quality=50):
            compressed_images = []
            buffer = io.BytesIO()
            
            for i in range(batch_images.shape[0]):
                img_tensor = batch_images[i].cpu()
                img_pil = transforms.ToPILImage()(img_tensor)
                
                buffer.seek(0)
                img_pil.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                
                img_compressed = Image.open(buffer)
                compressed_img_tensor = transforms.ToTensor()(img_compressed)
                compressed_images.append(compressed_img_tensor)
            
            buffer.close()
            return torch.stack(compressed_images).to(batch_images.device)
        
        return apply_jpeg_compression(images, quality=50)
    
    elif distortion_type == "median_blur":
        return K.median_blur(images, kernel_size=5)
    elif distortion_type == "gaussian_blur":
        return transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(images)
    elif distortion_type == "gaussian_noise":
        noise = torch.randn_like(images) * 0.05
        return torch.clamp(images + noise, -1, 1)
    elif distortion_type == "sharpness":
        return E.sharpness(images, 2.0)
    elif distortion_type == "brightness":
        return E.adjust_brightness(images, 0.8)
    elif distortion_type == "contrast":
        return E.adjust_contrast(images, 1.5)
    elif distortion_type == "saturation":
        return E.adjust_saturation(images, 1.5)
    elif distortion_type == "hue":
        return E.adjust_hue(images, 0.1)
    elif distortion_type == "noise_denoise":
        noisy = images + torch.randn_like(images) * 0.1
        return K.gaussian_blur2d(noisy, kernel_size=(5, 5), sigma=(1.5, 1.5))
    elif distortion_type == "random_crop":
        batch, c, h, w = images.shape
        crop_size = int(min(h, w) * 0.8)
        cropped = transforms.RandomCrop(crop_size)(images)
        return F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
    elif distortion_type == "random_rotation":
        angles = random.uniform(-30, 30)
        return transforms.functional.rotate(images, angles)
    else:
        return images


def test_watermark_model(args):
    """
    Test watermark model on multiple distortion scenarios:
    1. No distortion
    2. Image editing distortion (using generative model)
    3. Common distortions (various image processing operations)
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load watermark model
    wm_model, message_length = load_wm_model(args.checkpoint_dir, device)
    
    # Load test dataset
    test_dataset = get_test_dataset(args.image_size, args.test_size)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Load instruct-pix2pix pipeline for editing distortion test
    logger.info("Loading instruct-pix2pix pipeline")
    test_pipe = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
        "/public/zhangzhiling/models/timbrooks/instruct-pix2pix",
        torch_dtype=torch.float32,
        local_files_only=True
    ).to(device)
    test_pipe.text_encoder.eval()
    test_pipe.unet.eval()
    test_pipe.vae.eval()
    
    # Define distortion types
    distortion_types = [
        "jpeg", "median_blur", "gaussian_blur", "gaussian_noise", 
        "sharpness", "brightness", "contrast", "saturation", "hue",
        "noise_denoise", "random_crop", "random_rotation", 
    ]
    
    # Initialize result collectors
    results = {
        "no_distortion": 0,
        "edit_distortion": 0,
        "common_distortions": 0
    }
    
    for dist_type in distortion_types:
        results[f"{dist_type}"] = 0
    
    psnr_values = []
    ssim_values = []
    
    dataset_size = len(test_dataloader)
    logger.info(f"Starting evaluation on {dataset_size} batches")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            if batch_idx % 50 == 0:
                logger.info(f"Processing batch {batch_idx}/{dataset_size}")
            
            # Generate random message
            message = torch.randint(0, 2, (args.batch_size, message_length)).to(
                device=device, dtype=torch.float32
            )
            image, prompt = data["image"], data["prompt"]
            image = image.to(device)
            
            # Generate watermarked image
            wm_image = wm_model.encoder(image, message)
            
            # Calculate PSNR and SSIM
            psnr_value = psnr(denormalize(wm_image.detach()), denormalize(image), 1)
            ssim_value = torch.mean(ssim(denormalize(wm_image.detach()), denormalize(image), window_size=5))
            
            psnr_values.append(psnr_value.item())
            ssim_values.append(ssim_value.item())
            
            # ============ Scenario 1: No distortion ============
            decoded_message_no_distortion = wm_model.decoder(wm_image.to(dtype=torch.float32))
            error_rate_no_distortion = decoded_message_error_rate_batch(
                message, decoded_message_no_distortion
            )
            results["no_distortion"] += error_rate_no_distortion
            
            # ============ Scenario 2: Image editing distortion ============
            try:
                generated_image = generate_image(test_pipe, prompt, wm_image, device)
                decoded_message_after_edit = wm_model.decoder(generated_image.to(dtype=torch.float32))
                
                error_rate_after_edit = decoded_message_error_rate_batch(
                    message, decoded_message_after_edit
                )
                results["edit_distortion"] += error_rate_after_edit
            except Exception as e:
                logger.warning(f"Failed to generate edited image for batch {batch_idx}: {e}")
                results["edit_distortion"] += 1.0  # Assume worst case
            
            # ============ Scenario 3: Common distortions ============
            common_distortion_total = 0
            
            for dist_type in distortion_types:
                try:
                    distorted_image = apply_distortions(wm_image, dist_type)
                    decoded_message_distorted = wm_model.decoder(distorted_image.to(dtype=torch.float32))
                    
                    error_rate_distorted = decoded_message_error_rate_batch(
                        message, decoded_message_distorted
                    )
                    
                    results[dist_type] += error_rate_distorted
                    common_distortion_total += error_rate_distorted
                except Exception as e:
                    logger.warning(f"Failed to apply {dist_type} distortion for batch {batch_idx}: {e}")
                    results[dist_type] += 1.0  # Assume worst case
                    common_distortion_total += 1.0
            
            results["common_distortions"] += common_distortion_total / len(distortion_types)
    
    # Calculate final results
    final_results = {}
    
    # Main scenarios
    scenarios = ["no_distortion", "edit_distortion", "common_distortions"]
    for scenario in scenarios:
        final_results[f"{scenario}_BER"] = results[scenario] / dataset_size
    
    # Individual distortion types
    for dist_type in distortion_types:
        final_results[f"{dist_type}_BER"] = results[dist_type] / dataset_size
    
    # Image quality metrics
    final_results["psnr"] = float(np.mean(psnr_values))
    final_results["ssim"] = float(np.mean(ssim_values))
    
    # Clean up
    del test_pipe
    torch.cuda.empty_cache()
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Test watermark model on various distortion scenarios")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing watermark model checkpoint and config")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for testing (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for testing")
    parser.add_argument("--test_size", type=int, default=1200,
                        help="Number of test samples to use")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size for processing")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file to save results (JSON format)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    logger.info("Starting watermark model evaluation")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Image size: {args.image_size}")
    
    # Run evaluation
    results = test_watermark_model(args)
    
    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Image Quality Metrics:")
    logger.info(f"  PSNR: {results['psnr']:.4f}")
    logger.info(f"  SSIM: {results['ssim']:.4f}")
    
    logger.info(f"\nMain Scenarios:")
    logger.info(f"  No Distortion BER: {results['no_distortion_BER']:.4f}")
    logger.info(f"  Edit Distortion BER: {results['edit_distortion_BER']:.4f}")
    logger.info(f"  Common Distortions BER: {results['common_distortions_BER']:.4f}")
    
    logger.info(f"\nIndividual Distortion Types:")
    distortion_types = [
        "jpeg", "median_blur", "gaussian_blur", "gaussian_noise", 
        "sharpness", "brightness", "contrast", "saturation", "hue",
        "noise_denoise", "random_crop", "random_rotation"
    ]
    
    for dist_type in distortion_types:
        logger.info(f"  {dist_type} BER: {results[f'{dist_type}_BER']:.4f}")
    
    # Save results to file if specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
