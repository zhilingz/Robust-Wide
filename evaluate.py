import os
import argparse
import json
import logging
import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf
from kornia.metrics import psnr, ssim
import lpips
from model import WatermarkModel
from datasets import load_dataset
from diffusers import StableDiffusionInstructPix2PixPipeline
from custom.filter import setup_logging
from utils import (
    denormalize,
    decoded_message_error_rate,
)
from inference import generate_wm_image, load_wm_model

def load_image(imgname, target_size=256) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB') if isinstance(imgname, str) else imgname
    tform = transforms.Compose(
        [
            transforms.Resize(target_size, antialias=True),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return tform(pil_img)[None, ...]


@torch.no_grad()
def main(
    ckpt_dir,
    eval_img_dir,
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    test_size: int,
    device: str = "cuda:0",
    logger: logging.Logger = None,
):
    insp2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "/public/zhangzhiling/models/timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None
    ).to(device)
    wm_model, message_length = load_wm_model(ckpt_dir=ckpt_dir)
    wm_model = wm_model.to(device)
    
    # 初始化LPIPS模型
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
    
    full_dataset = load_dataset(eval_img_dir)
    total_len = len(full_dataset["train"])
    if total_len < test_size:
        raise ValueError(
            f"instructpix2pix 数据集样本数量({total_len})不足以提供测试集({test_size})样本"
        )
    test_start_idx = total_len - test_size
    logger.info("从 instructpix2pix 数据集末尾选择测试集，范围: %d - %d", test_start_idx, total_len - 1)
    eval_dataset = full_dataset["train"].select(range(test_start_idx, total_len))
    psnr_list, ssim_list, ber_list, lpips_list = [], [], [], []

    for data_dict in tqdm(eval_dataset):
        image_path, instruction = data_dict['original_image'], data_dict['edit_prompt']
        
        orig_image = load_image(image_path, 512).to(device)
        message = torch.randint(0, 2, size=(1, message_length)).float().to(device)
        wm_image = wm_model.encoder(orig_image, message)
        edited_wm_image = insp2p_pipe(
            instruction,
            image=wm_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
        ).images[0]
        edited_wm_image = load_image(edited_wm_image, edited_wm_image.size[0]).to(device)
        watermark = wm_model.decoder(edited_wm_image)
        
        psnr_value = psnr(denormalize(wm_image), denormalize(orig_image), 1)
        ssim_value = torch.mean(ssim(denormalize(wm_image), denormalize(orig_image), window_size=5))
        lpips_value = lpips_fn(orig_image, wm_image)
        ber = decoded_message_error_rate(message[0], watermark[0])
        
        psnr_list.append(psnr_value.item())
        ssim_list.append(ssim_value.item())
        lpips_list.append(lpips_value.item())
        ber_list.append(ber)
    
        logger.info(
            f"psnr: {np.mean(psnr_list)}, ssim: {np.mean(ssim_list)}, lpips: {np.mean(lpips_list)}, ber: {np.mean(ber_list)}"
        )

    logger.info(f"Final metrics:")
    logger.info(f"num_samples: {len(psnr_list)}")
    logger.info(f"mean±std:    ")
    logger.info(f"ber:  {float(np.mean(ber_list)):.3g}±{float(np.std(ber_list)):.2g}")
    logger.info(f"psnr: {float(np.mean(psnr_list)):.3g}±{float(np.std(psnr_list)):.2g}")
    logger.info(f"ssim: {float(np.mean(ssim_list)):.3g}±{float(np.std(ssim_list)):.2g}")
    logger.info(f"lpips:{float(np.mean(lpips_list)):.3g}±{float(np.std(lpips_list)):.2g}")
    generate_wm_image(wm_model, message_length, './examples/Gadot.png', logger.handlers[0].baseFilename, 512, device="cuda:0",)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--eval_img_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--guidance_scale', type=float, default=10.0)
    parser.add_argument('--image_guidance_scale', type=float, default=1.5)
    parser.add_argument('--test_size', type=int, default=1000)
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    params = vars(args)
    logger.info(json.dumps(params, ensure_ascii=False))
    main(
        ckpt_dir=args.ckpt_dir,
        eval_img_dir=args.eval_img_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        test_size=args.test_size,
        device='cuda:0',
        logger=logger,
    )
