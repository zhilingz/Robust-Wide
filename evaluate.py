import os
import argparse
import json
import logging
import torch
import random
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
from kornia.metrics import psnr, ssim
import lpips
from datasets import load_dataset
from diffusers import StableDiffusionInstructPix2PixPipeline
from custom.filter import setup_logging
from utils import (
    denormalize,
    decoded_message_error_rate,
    tensor_to_pil,
    normalize,
)
from inference import load_wm_model

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
    output_dir,
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    test_size: int,
    watermark_strength: float = 1.0,
    device: str = "cuda:0",
    logger: logging.Logger = None,
):
    insp2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "/public/zhangzhiling/models/timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None
    ).to(device)
    
    # 自动查找最新的checkpoint
    ckpt_files = [d for d in os.listdir(ckpt_dir) if d.startswith("step")]
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=lambda x: int(x.replace("step", "")))
        ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
        logger.info(f"自动选择最新的checkpoint: {ckpt_path}")
    else:
        # 如果没有step*目录，则假定ckpt_dir本身就是checkpoint目录
        ckpt_path = ckpt_dir
        logger.info(f"未找到 'step*' 目录, 直接使用ckpt_dir: {ckpt_path}")

    wm_model, message_length = load_wm_model(ckpt_dir=ckpt_path)
    wm_model = wm_model.to(device)
    
    # 初始化LPIPS模型
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)

    # 固定随机种子以确保结果可复现
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    message = torch.randint(0, 2, size=(1, message_length)).float().to(device)

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

    # 创建保存目录
    images_save_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(images_save_dir, exist_ok=True)

    for idx, data_dict in enumerate(eval_dataset):
        image_path, instruction = data_dict['original_image'], data_dict['edit_prompt']
        
        orig_image = load_image(image_path, 512).to(device)
        
        # 使用水印强度系数控制水印添加程度
        if watermark_strength == 0.0:
            # 强度为0时不添加水印
            wm_image = orig_image
        else:
            # 获取完整水印图像
            full_wm_image = wm_model.encoder(orig_image, message)
            # 根据强度系数在原图和水印图之间插值
            wm_image = orig_image + watermark_strength * (full_wm_image - orig_image)
        edited_wm_image = insp2p_pipe(
            instruction,
            image=wm_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
        ).images[0]
        edited_wm_image = load_image(edited_wm_image, edited_wm_image.size[0]).to(device)
        watermark = wm_model.decoder(edited_wm_image)
        
        # 保存前3张图片的原图、水印图和残差图，将原图、水印图和残差图保存在一张图上
        if idx < 3:
            # 计算水印残差并可视化
            residual = wm_image - orig_image
            residual_abs = torch.abs(residual)
            residual_abs_max = torch.max(residual_abs).item()
            residual_abs_min = torch.min(residual_abs).item()
            if residual_abs_max - residual_abs_min < 1e-8:
                residual_image = torch.zeros_like(residual_abs)
            else:
                residual_image = (residual_abs - residual_abs_min) / (residual_abs_max - residual_abs_min)
            residual_image = normalize(residual_image)

            # 转为PIL并横向拼接：原图 | 水印图 | 残差图
            pil_orig = tensor_to_pil(orig_image[0].detach().cpu())[0]
            pil_wm = tensor_to_pil(wm_image[0].detach().cpu())[0]
            pil_res = tensor_to_pil(residual_image[0].detach().cpu())[0]

            w, h = pil_orig.size
            triplet = Image.new('RGB', (w * 3, h))
            triplet.paste(pil_orig, (0, 0))
            triplet.paste(pil_wm, (w, 0))
            triplet.paste(pil_res, (2 * w, 0))

            triplet_path = os.path.join(images_save_dir, f"sample_{idx:03d}.png")
            triplet.save(triplet_path)
        
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
    logger.info(
        f"mean±std:\n"
        f"ber:  {float(np.mean(ber_list)):.3g}±{float(np.std(ber_list)):.2g}\n"
        f"psnr: {float(np.mean(psnr_list)):.3g}±{float(np.std(psnr_list)):.2g}\n"
        f"ssim: {float(np.mean(ssim_list)):.3g}±{float(np.std(ssim_list)):.2g}\n"
        f"lpips:{float(np.mean(lpips_list)):.3g}±{float(np.std(lpips_list)):.2g}\n"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--eval_img_dir', type=str, default='/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--edit_strength', type=str, default='middle', choices=['small', 'middle', 'large'], help='编辑力度 small/middle/large')
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--watermark_strength', type=float, default=1.0, help='水印强度系数，1.0为原始水印强度，0.0为不添加水印')
    args = parser.parse_args()

    # 如果未指定output_dir，则自动生成
    if args.output_dir is None:
        args.output_dir = os.path.join(args.ckpt_dir, "evaluate", args.edit_strength)

    # 根据编辑力度设置参数
    if args.edit_strength == 'small':
        num_inference_steps = 10
        guidance_scale = 3.0
        image_guidance_scale = 1.5
    elif args.edit_strength == 'middle':
        num_inference_steps = 10
        guidance_scale = 10.0
        image_guidance_scale = 1.5
    else:  # large
        num_inference_steps = 10
        guidance_scale = 10.0
        image_guidance_scale = 1.0

    logger = setup_logging(args.output_dir)
    params = vars(args)
    # 将自动设置的参数也加入日志
    params['num_inference_steps'] = num_inference_steps
    params['guidance_scale'] = guidance_scale
    params['image_guidance_scale'] = image_guidance_scale
    logger.info(json.dumps(params, ensure_ascii=False))
    main(
        ckpt_dir=args.ckpt_dir,
        eval_img_dir=args.eval_img_dir,
        output_dir=args.output_dir,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        test_size=args.test_size,
        watermark_strength=args.watermark_strength,
        device='cuda:0',
        logger=logger,
    )
