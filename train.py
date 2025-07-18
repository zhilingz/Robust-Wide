import os
import io
import argparse
import datetime, pytz
import json
import logging
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

# 测试失真
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import random
import kornia.filters as K
import kornia.enhance as E
import kornia.augmentation as A

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from kornia.metrics import psnr, ssim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from custom.custom_insp2p import CustomStableDiffusionInstructPix2PixPipeline

from dataset import get_hugging_dataset, get_filtered_dataset, collate_fn
from model import WatermarkModel
from utils import (
    decoded_message_error_rate_batch,
    denormalize,
)


logger = logging.getLogger(__name__)

def diff_image(before, after, method='edit_ratio', thresh=15, kernel=3):
    """
    使用多种方法计算两张图像的差异
    
    Args:
        before: 编辑前图像，形状为 (B, C, H, W) 或 (C, H, W)，值域 [-1, 1]
        after: 编辑后图像，形状为 (B, C, H, W) 或 (C, H, W)，值域 [-1, 1]
        method: 差异计算方法，可选：
               'edit_ratio' - 原始的编辑区域占比方法，阈值为0.3
               'psnr' - 峰值信噪比 (越高越好)，阈值为15
               'ssim' - 结构相似性指数 (越高越好)，阈值为0.75
               'l1' - L1距离/MAE (越低越好)，阈值为0.2
               'l2' - L2距离/MSE (越低越好)，阈值为0.2
        thresh: 像素差异阈值 (仅用于edit_ratio方法)
        kernel: 形态学开运算核大小 (仅用于edit_ratio方法)
    
    Returns:
        返回对应指标的张量，形状为 (B,) 或标量
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
    skip_flag = False

    # 定义各指标的阈值
    if method == 'all':
        thresholds = {
            'psnr': 20.0,        # PSNR阈值，单位dB
            'ssim': 0.80,        # SSIM阈值
            'l1': 0.1,           # L1距离阈值
            'l2': 0.03,          # L2距离阈值
            'edit_ratio': 0.20   # 编辑区域占比阈值
        }
    else:
        thresholds = {
            'psnr': args.filter_threshold,
            'ssim': args.filter_threshold,
            'l1': args.filter_threshold,
            'l2': args.filter_threshold,
            'edit_ratio': args.filter_threshold
        }
        
    if method == 'psnr' or method == 'all':
        # 计算PSNR (Peak Signal-to-Noise Ratio)
        # 值越高表示图像质量越好
        psnr_values = []
        for i in range(batch_size):
            psnr_val = psnr(after_01[i:i+1], before_01[i:i+1], max_val=1.0)
            psnr_values.append(psnr_val.item())
        results['psnr'] = torch.tensor(psnr_values, device=device)
        if results['psnr'] < thresholds['psnr']:
            skip_flag = True
            return skip_flag
    
    if method == 'ssim' or method == 'all':
        # 计算SSIM (Structural Similarity Index)
        # 值越高表示结构相似性越好
        ssim_values = []
        for i in range(batch_size):
            ssim_val = torch.mean(ssim(after_01[i:i+1], before_01[i:i+1], window_size=5))
            ssim_values.append(ssim_val.item())
        results['ssim'] = torch.tensor(ssim_values, device=device)
        if results['ssim'] < thresholds['ssim']:
            skip_flag = True
            return skip_flag
    
    if method == 'l1' or method == 'all':
        # 计算L1距离 (Mean Absolute Error)
        # 值越低表示差异越小
        l1_values = F.l1_loss(after, before, reduction='none')
        l1_values = l1_values.view(batch_size, -1).mean(dim=1)
        results['l1'] = l1_values
        if results['l1'] > thresholds['l1']:
            skip_flag = True
            return skip_flag
        
    if method == 'l2' or method == 'all':
        # 计算L2距离 (Mean Squared Error)
        # 值越低表示差异越小
        l2_values = F.mse_loss(after, before, reduction='none')
        l2_values = l2_values.view(batch_size, -1).mean(dim=1)
        results['l2'] = l2_values
        if results['l2'] > thresholds['l2']:
            skip_flag = True
            return skip_flag
        
    if method == 'edit_ratio' or method == 'all':
        # 原始的编辑区域占比方法
        # 转换到 [0, 255] 范围并转为numpy
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
        if results['edit_ratio'] > thresholds['edit_ratio']:
            skip_flag = True
            return skip_flag
    
    return skip_flag
    # 返回结果
    # result = results[method]
    # if squeeze_output:
    #     return result.squeeze()
    # return result, skip_flag

def initialize_pipeline(args, weight_dtype, device):
    # 训练：不同模型不同加载方式
    if "instruct-pix2pix" in args.model_dir or "magicbrush" in args.model_dir:
        pipe = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.model_dir, torch_dtype=weight_dtype, local_files_only=True
        ).to(device)
    elif "sd-turbo" in args.model_dir:
        from custom.custom_i2i import CustomStableDiffusionImg2ImgPipeline
        pipe = CustomStableDiffusionImg2ImgPipeline.from_pretrained(
            args.model_dir, torch_dtype=weight_dtype, local_files_only=True
        ).to(device)
    elif "sd-x2-latent-upscaler" in args.model_dir:
        from custom.custom_sd import CustomStableDiffusionPipeline, CustomStableDiffusionLatentUpscalePipeline
        pipe = CustomStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=weight_dtype, local_files_only=True
        ).to(device)
        upscaler = CustomStableDiffusionLatentUpscalePipeline.from_pretrained(
            args.model_dir, torch_dtype=weight_dtype, local_files_only=True
        ).to(device)
        pipe = [pipe, upscaler]
    elif "FLUX" in args.model_dir:
        from custom.custom_flux import CustomFluxFillPipeline
        pipe = CustomFluxFillPipeline.from_pretrained(
            args.model_dir, torch_dtype=weight_dtype, local_files_only=True
        )
        pipe.load_lora_weights(
            "RiverZ/normal-lora", local_files_only=True, 
            weight_name="pytorch_lora_weights.safetensors",
            lora_scale=1.0)
        pipe = pipe.to(device)
    else:
        raise ValueError("model not supported")

    # 不同模型不同scheduler
    if "magicbrush" in args.model_dir:
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif "instruct-pix2pix-distill" in args.model_dir:
        from diffusers import LCMScheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        # Adapt the InstructPix2Pix model using the LoRA parameters
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

    # 冻结参数
    if "sd-x2-latent-upscaler" in args.model_dir:
        for p in pipe:
            p.freeze_params()
            p.text_encoder.train()
            p.unet.train()
            p.vae.train()
    else:
        pipe.freeze_params()
        pipe.text_encoder.train()
        pipe.unet.train()
        pipe.vae.train()

    return pipe

def generate_image(args, pipe, prompt, wm_image, accelerator, is_test=False, device=None, seed=None):
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    
    # 设置默认种子
    if seed is None:
        seed = getattr(args, 'generation_seed', 42)  # 使用args中的种子，默认为42
    
    # 创建生成器，每次生成图片都重置Generator，避免每调用一次随机函数，内部状态就前进一次。
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    if is_test:
        # 这里的参数你可以根据需求调整
        generated_image = pipe(
            prompt,
            image=wm_image,
            num_images_per_prompt=1,
            num_inference_steps=20,
            guidance_scale=10,
            image_guidance_scale=1.5,
            generator=generator,
            last_grad_steps=args.last_grad_steps,
            output_type="pt",
        )
    else:
        # 根据模型选择不同的pipe参数
        if "instruct-pix2pix-distill" in args.model_dir:
            generated_image = pipe(
                prompt, 
                image=wm_image, 
                num_images_per_prompt=1, 
                num_inference_steps=4,  # 4改为2，测试效果
                guidance_scale=2.0,     # 使用较小的guidance_scale
                image_guidance_scale=1.0,  # 使用较小的image_guidance_scale
                generator=generator,
                last_grad_steps=args.last_grad_steps,
                output_type="pt"
            )
        elif "sd-turbo" in args.model_dir:
            generated_image = pipe(
                prompt, 
                image=wm_image, 
                num_images_per_prompt=1, 
                num_inference_steps=2,
                guidance_scale=0.0, 
                strength=0.5,
                generator=generator,
                last_grad_steps=args.last_grad_steps,
                output_type="pt"
            ).images     # pipe返回值为StableDiffusionPipelineOutput 类型，需要取images，形状 (1, C, H, W)
            generated_image = 2 * generated_image - 1 # 将值域从[-1,1]转为[0,1]，防止图片泛白
        elif "sd-x2-latent-upscaler" in args.model_dir:
            # 将水印图像编码到潜在空间
            # 移除: wm_image = wm_image.to(dtype=torch.float16) 
            # 因为它可能创建了一个新的 float16 张量，而原始的 float32 张量可能在计算图中仍然被引用。
            # 通常建议让 Accelerator 自动处理混合精度类型。
            # 需要存储两个模型的梯度，batchsize设为1
            with accelerator.autocast():
                latent_dist = pipe[0].vae.encode(wm_image).latent_dist 
                low_res_latents = latent_dist.mean 
                low_res_latents = low_res_latents * pipe[0].vae.config.scaling_factor 
            generated_image = pipe[1](
                prompt=prompt,
                image=low_res_latents,
                num_inference_steps=20,
                guidance_scale=0,
                generator=generator,
                last_grad_steps=args.last_grad_steps,
                output_type="pt"
            )
            # 方法二：将生成的图像调整到512x512大小
            generated_image = F.interpolate(generated_image, size=(512, 512), mode='bilinear', align_corners=False)
            generated_image = 2 * generated_image - 1  
        elif "magicbrush" in args.model_dir:
            generated_image = pipe(
                prompt, 
                image=wm_image, 
                num_inference_steps=20, 
                image_guidance_scale=2.0, 
                guidance_scale=4, 
                generator=generator,
                last_grad_steps=args.last_grad_steps,
                output_type="pt",
                )
        elif "FLUX" in args.model_dir:
            # 将 tensor 转成 PIL 图像
            pil_img = to_pil_image(denormalize(wm_image[0].cpu()))
            width, height = pil_img.size
            # 构造拼接图像和 mask
            combined = Image.new("RGB", (width*2, height))
            combined.paste(pil_img, (0, 0))
            combined.paste(pil_img, (width, 0))
            mask_array = np.zeros((height, width*2), dtype=np.uint8)
            mask_array[:, width:] = 255
            mask = Image.fromarray(mask_array)
            # prompt
            prompt = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {prompt}'
            # 运行 FluxFillPipeline
            result = pipe(
                prompt=prompt,
                image=combined,
                mask_image=mask,
                height=height,
                width=width*2,
                guidance_scale=50,
                num_inference_steps=28,
                generator=generator,
                output_type="pt"
            ).images[0]
            # 裁剪右半部分并转回 tensor
            cropped = result.crop((width, 0, width*2, height))
            generated_image = transforms.ToTensor()(cropped).unsqueeze(0).to(device)
        else:
            generated_image = pipe(
                prompt, 
                image=wm_image, 
                num_images_per_prompt=1, 
                num_inference_steps=20,
                guidance_scale=10, 
                image_guidance_scale=1.5, # 1.5原图保留太少（62.98%的编辑区域），2.0还可以（38.45%的编辑区域）
                generator=generator,
                last_grad_steps=args.last_grad_steps,
                output_type="pt",
            )
    return generated_image

def calculate_edit_analysis(before, after, thresh=15, kernel=3):
    """
    计算编辑区域的mask和占比
    """
    # 转换为numpy进行opencv操作
    before_np = ((before.detach().cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()[0].transpose(1, 2, 0)
    after_np = ((after.detach().cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()[0].transpose(1, 2, 0)
    
    # 计算颜色差异
    diff = np.abs(after_np.astype(np.int16) - before_np.astype(np.int16))
    color_diff = np.max(diff, axis=2).astype(np.uint8)
    
    # 二值化和形态学处理
    _, mask = cv2.threshold(color_diff, thresh, 255, cv2.THRESH_BINARY)
    kernel_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_elem, iterations=1)
    
    ratio = mask.sum() / 255 / mask.size
    return mask, ratio

def calculate_metrics(before, after):
    """计算图像对比指标"""
    # 统一转换为tensor
    def to_tensor(image):
        if isinstance(image, Image.Image):
            tensor = transforms.ToTensor()(image) * 2 - 1
        else:
            tensor = image
        return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
    
    before = to_tensor(before)
    after = to_tensor(after)
    
    # 转换到[0,1]用于PSNR和SSIM
    before_01 = (before + 1) / 2
    after_01 = (after + 1) / 2
    
    # 计算编辑比例和mask
    mask, edit_ratio = calculate_edit_analysis(before, after)
    
    # 计算SSIM
    ssim_val = ssim(after_01, before_01, window_size=5)
    ssim_score = torch.mean(ssim_val).item() if ssim_val.dim() > 0 else ssim_val.item()
    
    metrics = {
        'psnr': psnr(after_01, before_01, max_val=1.0).item(),
        'ssim': ssim_score,
        'l1': F.l1_loss(after, before).item(),
        'l2': F.mse_loss(after, before).item(),
        'edit_ratio': edit_ratio
    }
    
    return metrics, mask

def tensor_to_bgr(tensor):
    """将tensor转换为BGR格式的numpy数组"""
    np_img = ((tensor.detach().cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()[0]
    if np_img.shape[0] == 3:  # CHW -> HWC
        np_img = np_img.transpose(1, 2, 0)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

def create_metrics_overlay(metrics, width, height):
    """创建包含指标信息的图像"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if not metrics:
        return img
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    texts = [
        f"PSNR: {metrics['psnr']:.2f}",
        f"SSIM: {metrics['ssim']:.3f}",
        f"L1: {metrics['l1']:.4f}",
        f"L2: {metrics['l2']:.4f}",
        f"Edit: {metrics['edit_ratio']:.3f}"
    ]
    
    for i, text in enumerate(texts):
        y_pos = 30 + i * 25
        cv2.putText(img, text, (10, y_pos), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img

def create_comparison_image(pipe, accelerator, original_img, wm_img, prompt, output_dir, step):
    """
    创建并保存包含8张图片的2x4对比图像
    第一行：原图、原图的编辑图像A、A相比原图的编辑区域、A相比原图的残差
    第二行：原图的水印图像、水印图像的编辑图像B、B相比原图的编辑区域、B相比原图的残差
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成编辑图像
    with torch.no_grad():   
        pipe.text_encoder.eval()
        pipe.unet.eval()
        pipe.vae.eval()
        
        generated_image_A = generate_image(args, pipe, prompt, original_img, accelerator, device=accelerator.device)
        generated_image_B = generate_image(args, pipe, prompt, wm_img, accelerator, device=accelerator.device)
        
        pipe.text_encoder.train()
        pipe.unet.train()
        pipe.vae.train()
    
    # 计算指标和mask
    metrics_A, mask_A = calculate_metrics(original_img, generated_image_A)
    metrics_B, mask_B = calculate_metrics(original_img, generated_image_B)
    
    # 转换所有图像
    images = [original_img, generated_image_A, generated_image_B, wm_img]
    bgr_images = [tensor_to_bgr(img) for img in images]
    original_bgr, gen_A_bgr, gen_B_bgr, wm_bgr = bgr_images
    
    h, w = original_bgr.shape[:2]
    
    # 处理mask和残差
    mask_A_bgr = cv2.cvtColor(mask_A, cv2.COLOR_GRAY2BGR) if mask_A is not None else np.zeros((h, w, 3), dtype=np.uint8)
    mask_B_bgr = cv2.cvtColor(mask_B, cv2.COLOR_GRAY2BGR) if mask_B is not None else np.zeros((h, w, 3), dtype=np.uint8)
    
    diff_A = cv2.absdiff(original_bgr, gen_A_bgr)
    diff_B = cv2.absdiff(original_bgr, gen_B_bgr)
    
    # 创建指标图像
    metrics_A_img = create_metrics_overlay(metrics_A, w, h)
    metrics_B_img = create_metrics_overlay(metrics_B, w, h)
    
    # 创建网格
    row1 = cv2.hconcat([original_bgr, gen_A_bgr, mask_A_bgr, diff_A, metrics_A_img])
    row2 = cv2.hconcat([wm_bgr, gen_B_bgr, mask_B_bgr, diff_B, metrics_B_img])
    comparison_grid = cv2.vconcat([row1, row2])
    
    # 添加标签
    labels = ["Original", "Edit", "Mask", "Diff", "Metrics"]
    label_bg = np.zeros((30, comparison_grid.shape[1], 3), dtype=np.uint8)
    
    for i, label in enumerate(labels):
        x_pos = i * w + 10
        cv2.putText(label_bg, label, (x_pos, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    final_image = cv2.vconcat([label_bg, comparison_grid])
    
    # 添加prompt文本
    if prompt:
        prompt_text = prompt[0] if isinstance(prompt, list) else str(prompt)
        prompt_text = prompt_text.strip()
        
        # 创建文本背景
        (text_width, text_height), baseline = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_bg_height = text_height + baseline + 30
        text_bg = np.zeros((text_bg_height, final_image.shape[1], 3), dtype=np.uint8)
        
        cv2.putText(text_bg, prompt_text, (15, text_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        final_image = cv2.vconcat([final_image, text_bg])
    
    # 保存图像
    output_path = os.path.join(output_dir, f"comparison_8grid_{step}.png")
    cv2.imwrite(output_path, final_image)
    
    return output_path, metrics_A, metrics_B

def setup_logging(args, logger):
    # 获取数据集名称和模型名称
    if args.enable_offline_filter:
        data_name = args.train_data_dir.split(os.sep)[-2]
        model_name = args.model_dir.split(os.sep)[-1]
    else:
        data_name = args.train_data_dir.split(os.sep)[-4]
        model_name = args.model_dir.split(os.sep)[-1]

    # 获取当前时间并构建输出目录
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%dT%H-%M-%S")
    
    # 获取SLURM任务ID
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if slurm_job_id:
        output_with_time_dir = os.path.join(args.output_dir, f"{now}_{data_name}_{model_name}_{slurm_job_id}")
    else:
        output_with_time_dir = os.path.join(args.output_dir, f"{now}_{data_name}_{model_name}")
    
    os.makedirs(output_with_time_dir, exist_ok=True)

    # 自定义时区格式化器类
    class ShangHaiTimeFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            # 获取UTC时间戳并转换为datetime
            dt = datetime.datetime.fromtimestamp(record.created, tz=pytz.UTC)
            # 将UTC时间转换为上海时间
            dt = dt.astimezone(pytz.timezone('Asia/Shanghai'))
            
            # 格式化时间
            if datefmt:
                s = dt.strftime(datefmt)
            else:
                s = dt.strftime("%Y-%m-%d %H:%M:%S")
            return s
        
    # 设置根日志级别
    logger.setLevel(logging.INFO)

    # 创建文件处理器并应用自定义格式化器
    formatter = ShangHaiTimeFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%m/%d/%Y %H:%M:%S")
    fhlr = logging.FileHandler(os.path.join(output_with_time_dir, "log.txt"))
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)
    
    logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))
    # —— 打印 sbatch/SLURM 环境信息 —— 
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_ON_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_JOB_TIME_LIMIT"
    ]
    slurm_info = {var: os.environ.get(var) for var in slurm_vars}
    logger.info(f"Slurm info: {slurm_info}")

    return output_with_time_dir

def test_model(args, wm_model, test_dataloader, device, accelerator):
    """
    测试水印模型在多种场景下的性能：
    1. 无失真场景
    2. 图像编辑失真场景(使用生成模型)
    3. 通用失真场景(包括多种图像处理操作)
    
    输出每种场景的比特错误率(BER)以及原图和水印图的SSIM和PSNR
    """
    # 重新加载 test_pipe
    from custom.custom_insp2p import CustomStableDiffusionInstructPix2PixPipeline
    test_pipe = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
        "/public/zhangzhiling/models/timbrooks/instruct-pix2pix",
        torch_dtype=wm_model.weight_dtype,
        local_files_only=True
    ).to(device)
    test_pipe.text_encoder.eval()
    test_pipe.unet.eval()
    test_pipe.vae.eval()
    
    # 定义通用失真变换
    def apply_distortions(images, distortion_type):
        if distortion_type == "jpeg":
            # JPEG压缩
            def apply_jpeg_compression(batch_images, quality=50):
                """
                对批量图像应用JPEG压缩（高效版本）
                
                参数:
                batch_images: 形状为 [batch_size, channels, height, width] 的张量
                quality: JPEG压缩质量 (0-100)
                """
                compressed_images = []
                
                # 创建内存缓冲区
                buffer = io.BytesIO()
                
                # 遍历batch中的每个图像
                for i in range(batch_images.shape[0]):
                    # 提取单张图像并转换为PIL格式
                    img_tensor = batch_images[i].cpu()  # [channels, height, width]
                    img_pil = transforms.ToPILImage()(img_tensor)
                    
                    # 使用内存缓冲区进行JPEG压缩，避免文件IO
                    buffer.seek(0)
                    img_pil.save(buffer, format='JPEG', quality=quality)
                    buffer.seek(0)
                    
                    # 从缓冲区加载压缩后的图像
                    img_compressed = Image.open(buffer)
                    
                    # 转回张量并添加到列表
                    compressed_img_tensor = transforms.ToTensor()(img_compressed)
                    compressed_images.append(compressed_img_tensor)
                
                # 清理缓冲区
                buffer.close()
                
                # 将处理后的图像重新组合为batch
                return torch.stack(compressed_images).to(batch_images.device)
            return apply_jpeg_compression(wm_image, quality=50)
        elif distortion_type == "median_blur":
            # 中值模糊
            return K.median_blur(images, kernel_size=5)
        elif distortion_type == "gaussian_blur":
            # 高斯模糊
            return transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(images)
        elif distortion_type == "gaussian_noise":
            # 高斯噪声
            noise = torch.randn_like(images) * 0.05
            return torch.clamp(images + noise, -1, 1)
        elif distortion_type == "sharpness":
            # 锐化
            return E.sharpness(images, 2.0)  # 增强锐度
        elif distortion_type == "brightness":
            # 亮度调整
            return E.adjust_brightness(images, 0.8)  # 降低亮度
        elif distortion_type == "contrast":
            # 对比度调整
            return E.adjust_contrast(images, 1.5)  # 增加对比度
        elif distortion_type == "saturation":
            # 饱和度调整
            return E.adjust_saturation(images, 1.5)  # 增加饱和度
        elif distortion_type == "hue":
            # 色调调整
            return E.adjust_hue(images, 0.1)  # 调整色调
        elif distortion_type == "noise_denoise":
            # 添加噪声后去噪
            noisy = images + torch.randn_like(images) * 0.1
            return K.gaussian_blur2d(noisy, kernel_size=(5, 5), sigma=(1.5, 1.5))
        elif distortion_type == "random_crop":
            # 随机裁剪并调整回原始大小
            batch, c, h, w = images.shape
            crop_size = int(min(h, w) * 0.8)  # 裁剪80%的区域
            cropped = transforms.RandomCrop(crop_size)(images)
            return F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
        elif distortion_type == "random_rotation":
            # 随机旋转
            angles = random.uniform(-30, 30)  # 旋转角度在-30到30度之间
            return transforms.functional.rotate(images, angles)
        else:
            return images
    
    # 测试模型
    wm_model.eval()
    
    # 定义不同的测试场景
    scenarios = ["no_distortion", "edit_distortion", "common_distortions"]
    
    # 扩展失真类型
    distortion_types = [
        "jpeg", "median_blur", "gaussian_blur", "gaussian_noise", 
        "sharpness", "brightness", "contrast", "saturation", "hue",
        "noise_denoise", "random_crop", "random_rotation", 
    ]
    
    # 为每个场景创建比特错误率收集器
    results = {
        "no_distortion": 0,
        "edit_distortion": 0,
        "common_distortions": 0
    }
    
    # 为通用失真场景创建每种失真类型的比特错误率收集器
    for dist_type in distortion_types:
        results[f"{dist_type}"] = 0
    
    # 添加SSIM和PSNR收集器
    psnr_values = []
    ssim_values = []
    
    # 计算数据集大小用于平均
    dataset_size = len(test_dataloader)

    with torch.no_grad():
        for data in test_dataloader:
            # 生成随机消息
            message = torch.randint(0, 2, (args.batch_size, args.message_length)).to(
                device=device, dtype=torch.float32
            )
            image, prompt = data["image"], data["prompt"]
            
            wm_image = wm_model.encoder(image, message)
            
            # 计算原图和水印图的PSNR和SSIM
            psnr_value = psnr(denormalize(wm_image.detach()), denormalize(image), 1)
            ssim_value = torch.mean(ssim(denormalize(wm_image.detach()), denormalize(image), window_size=5))
            
            psnr_values.append(psnr_value.item())
            ssim_values.append(ssim_value.item())
            
            # ============ 场景1: 无失真 ============ #
            decoded_message_no_distortion = wm_model.decoder(wm_image.to(dtype=torch.float32))
            error_rate_no_distortion = decoded_message_error_rate_batch(
                message, decoded_message_no_distortion
            )
            
            results["no_distortion"] += error_rate_no_distortion
            
            # ============ 场景2: 图像编辑失真 ============ #
            generated_image = generate_image(args, test_pipe, prompt, wm_image, accelerator, is_test=True, device=device)
            decoded_message_after_edit = wm_model.decoder(generated_image.to(dtype=torch.float32))
            
            error_rate_after_edit = decoded_message_error_rate_batch(
                message, decoded_message_after_edit
            )
            
            results["edit_distortion"] += error_rate_after_edit
            
            # ============ 场景3: 通用失真 ============ #
            common_distortion_total = 0
            
            for dist_type in distortion_types:
                distorted_image = apply_distortions(wm_image, dist_type)
                
                decoded_message_distorted = wm_model.decoder(distorted_image.to(dtype=torch.float32))
                
                error_rate_distorted = decoded_message_error_rate_batch(
                    message, decoded_message_distorted
                )
                
                results[dist_type] += error_rate_distorted
                common_distortion_total += error_rate_distorted
            
            # 更新通用失真的平均错误率
            results["common_distortions"] += common_distortion_total / len(distortion_types)

    log_dict = {}
    
    # 处理主要场景的平均错误率
    for scenario in scenarios:
        log_dict[f"{scenario}_BER"] = results[scenario] / dataset_size
    
    # 处理通用失真中每种失真类型的平均错误率
    for dist_type in distortion_types:
        log_dict[f"{dist_type}_BER"] = results[dist_type] / dataset_size
    
    # 添加SSIM和PSNR的平均值
    log_dict["psnr"] = float(np.mean(psnr_values))
    log_dict["ssim"] = float(np.mean(ssim_values))
    
    logger.info(log_dict)
    
    # 释放 test_pipe 显存
    del test_pipe  
    torch.cuda.empty_cache()
    
    wm_model.train()
    return log_dict

def save_all(g_model, save_dir, accelerator, args, wm_model_config, pipe, image, wm_image, prompt, output_with_time_dir):
    """
    保存模型、配置文件、图片和执行相关脚本
    """
    # 保存模型和配置
    unwrapped_model = accelerator.unwrap_model(g_model)
    accelerator.save(unwrapped_model.state_dict(), os.path.join(save_dir, "wm_model.ckpt"))
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    OmegaConf.save(wm_model_config, os.path.join(save_dir, "wm_model_config.yaml"))
    
    # 保存编辑图片
    save_image(denormalize(image[0].detach().cpu()), os.path.join(save_dir, "image.png"))
    save_image(denormalize(wm_image[0].detach().cpu()), os.path.join(save_dir, "wm_image.png"))
    
    with torch.no_grad():   
        pipe.text_encoder.eval()
        pipe.unet.eval()
        pipe.vae.eval()
        generated_image_before_wm = generate_image(args, pipe, prompt, image, accelerator, device=accelerator.device)
        generated_image = generate_image(args, pipe, prompt, wm_image, accelerator, device=accelerator.device)
        pipe.text_encoder.train()
        pipe.unet.train()
        pipe.vae.train()

    if isinstance(generated_image, torch.Tensor):
        save_image(denormalize(generated_image[0].detach().cpu()), os.path.join(save_dir, "generated_image.png"))
        save_image(denormalize(generated_image_before_wm[0].detach().cpu()), os.path.join(save_dir, "generated_image_before_wm.png"))
    else:
        save_image(denormalize(generated_image[0][0].detach().cpu()), os.path.join(save_dir, "generated_image.png"))
        save_image(denormalize(generated_image_before_wm[0][0].detach().cpu()), os.path.join(save_dir, "generated_image_before_wm.png"))
    
    # 保存 prompt
    with open(os.path.join(save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        if isinstance(prompt, list):
            f.write(str(prompt[0]))
        else:
            f.write(str(prompt))

    # 调用脚本，生成inference、频谱图、编辑区域图、log图
    image_file = './examples/Gadot.png'
    cmds = [
        f'python inference.py --ckpt_dir "{save_dir}" --image_file "{image_file}" --output_dir "{save_dir}"',
        f'python custom/fft.py --folder "{save_dir}"',
        f'python custom/diff.py --before "{save_dir}/wm_image.png" --after "{save_dir}/generated_image.png" --output "{save_dir}/diff.png"',
        # f'sbatch custom/lp.sh "{output_with_time_dir}/log.txt"',
        f'python custom/log2plt.py -l "{output_with_time_dir}/log.txt"',
        f'python custom/res.py --folder "{save_dir}"'
    ]
    for cmd in cmds:
        os.system(cmd)

def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    logger.info("Using GPU:", os.environ['CUDA_VISIBLE_DEVICES'])

    # 配置日志记录并获取输出目录
    output_with_time_dir = setup_logging(args, logger)
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    wm_model_config = OmegaConf.load(args.wm_model_config)
    args.message_length = wm_model_config["wm_enc_config"]["message_length"]
    wm_model = WatermarkModel(
        **wm_model_config,
        device=device,
        weight_dtype=weight_dtype,
    )
    wm_model.train()

    params_to_optimize = list(p for p in wm_model.parameters() if p.requires_grad)

    pipe = initialize_pipeline(args, weight_dtype, device)
    logger.info("pipe",pipe)
    
    if args.enable_offline_filter:
        # 离线筛选数据集
        train_dataset, test_dataset = get_filtered_dataset(
            args, 
            logger,
            args.image_size, 
            accelerator, 
            args.train_size, 
            args.test_size
        )
        logger.info("使用离线筛选数据集进行训练")
    else:
        # 正常加载数据集
        train_dataset, test_dataset = get_hugging_dataset(
            logger,
            args.train_data_dir, 
            args.image_size, 
            accelerator, 
            args.train_size, 
            args.test_size
        )
        logger.info("使用完整数据集进行训练")

    # 如果开启实时筛选，batch_size必须为1
    if args.enable_online_filter:
        if args.batch_size != 1:
            logger.warning(f"开启实时筛选时，batch_size必须为1，当前设置为{args.batch_size}，已自动调整为1")
            args.batch_size = 1

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    opt = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate,)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    wm_model, opt, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        wm_model, opt, train_dataloader, test_dataloader, lr_scheduler
    )

    step = 0
    global_step = 0
    finished_flag = False
    
    if args.enable_output_images:
        metrics_A_list = []
        metrics_B_list = []

    # 只在开启筛选时初始化统计变量
    if args.enable_online_filter:
        diff_values_all = []
        processed_samples = 0  # 处理的样本总数
        filtered_samples = 0   # 通过筛选的样本数
        skipped_samples = 0    # 跳过的样本数
        
        # 创建保存筛选样本的目录
        filtered_samples_dir = os.path.join(output_with_time_dir, "filtered_samples")
        os.makedirs(filtered_samples_dir, exist_ok=True)
    
    while True:
        for data in train_dataloader:
            step += 1
            
            with accelerator.accumulate(wm_model):
                message = torch.randint(0, 2, (args.batch_size, args.message_length)).to(
                    device=device, dtype=torch.float32
                )
                image, prompt = data["image"], data["prompt"]

                # 开启实时筛选
                if args.enable_online_filter:
                    processed_samples += 1
                    # 1) 先在 no_grad 环境里做粗筛
                    with torch.no_grad():
                        generated_preview = generate_image(args, pipe, prompt, image, accelerator, device=device)
                        skip_flag = diff_image(image, generated_preview, method=args.filter_method)
                        if skip_flag:
                            # 编辑区域过大，跳过这个样本
                            skipped_samples += 1
                            continue
                        else:
                            # 样本通过筛选
                            filtered_samples += 1
                            
                            # 保存前100个通过筛选的样本
                            if filtered_samples <= 100:
                                # 保存原图
                                save_image(
                                    denormalize(image[0].detach().cpu()),
                                    os.path.join(filtered_samples_dir, f"{filtered_samples}_image.png")
                                )
                                # 保存生成图
                                save_image(
                                    denormalize(generated_preview[0].detach().cpu()),
                                    os.path.join(filtered_samples_dir, f"{filtered_samples}_generated_preview.png")
                                )
                                # 保存prompt
                                with open(os.path.join(filtered_samples_dir, f"{filtered_samples}_prompt.txt"), "w", encoding="utf-8") as f:
                                    if isinstance(prompt, list):
                                        f.write(str(prompt[0]))
                                    else:
                                        f.write(str(prompt))
                
                # 2) 真正要训练的样本再跑一次 full forward（带梯度）
                wm_image = wm_model.encoder(image, message)

                if "sd-x2-latent-upscaler" in args.model_dir:
                    image_latents = pipe[1].vae.encode(image.to(dtype=weight_dtype)).latent_dist.mode()
                    wm_image_latents = pipe[1].vae.encode(wm_image.to(dtype=weight_dtype)).latent_dist.mode()
                else:
                    image_latents = pipe.vae.encode(image.to(dtype=weight_dtype)).latent_dist.mode()
                    wm_image_latents = pipe.vae.encode(wm_image.to(dtype=weight_dtype)).latent_dist.mode()

                decoded_message_before_edit = wm_model.decoder(wm_image.to(dtype=torch.float32))
                
                generated_image = generate_image(args, pipe, prompt, wm_image, accelerator, device=device)
        
                decoded_message_after_edit = wm_model.decoder(generated_image.to(dtype=torch.float32))
                
                # Calculate losses, decoder_weight默认0.1，enc_latent_weight 默认0.001
                enc_pixel_loss = F.mse_loss(image.float(), wm_image.float())
                enc_latent_loss = F.mse_loss(image_latents.float(), wm_image_latents.float())
                dec_loss_before_edit = F.mse_loss(message, decoded_message_before_edit)
                dec_loss_after_edit = F.mse_loss(message, decoded_message_after_edit)
                enc_loss = enc_pixel_loss + args.enc_latent_weight * enc_latent_loss
                dec_loss = dec_loss_before_edit + args.decoder_weight * dec_loss_after_edit
                # 先让encoder收敛
                # if global_step < 1000:
                #     loss = enc_pixel_loss + args.enc_latent_weight * enc_latent_loss
                #     loss += dec_loss_before_edit + 0.000001 * dec_loss_after_edit
                # else:
                #     loss = enc_pixel_loss + args.enc_latent_weight * enc_latent_loss
                #     loss += dec_loss_before_edit + args.decoder_weight * dec_loss_after_edit
                loss = enc_loss + dec_loss


                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    opt.step()
                    lr_scheduler.step()
                    opt.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.log_steps == 0:
                        psnr_value = psnr(denormalize(wm_image.detach()), denormalize(image), 1)
                        ssim_value = torch.mean(ssim(denormalize(wm_image.detach()), denormalize(image), window_size=5))
                        error_rate_after_edit = decoded_message_error_rate_batch(
                            message, decoded_message_after_edit
                        )
                        error_rate_before_edit = decoded_message_error_rate_batch(
                            message, decoded_message_before_edit
                        )
                        
                        log_dict = {
                            "step": step,
                            "global_step": global_step,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "enc_pixel_loss": enc_pixel_loss.detach().item(),
                            "enc_latent_loss": enc_latent_loss.detach().item(),
                            "dec_loss_before_edit": dec_loss_before_edit.detach().item(),
                            "dec_loss_after_edit": dec_loss_after_edit.detach().item(),
                            "psnr": psnr_value.item(),
                            "ssim": ssim_value.item(),
                            "error_rate_before_edit": error_rate_before_edit,
                            "error_rate_after_edit": error_rate_after_edit,
                        }
                        
                        # 根据是否开启筛选添加不同的编辑区域信息
                        if args.enable_online_filter:
                            # 计算筛选统计信息
                            filter_rate = filtered_samples / processed_samples if processed_samples > 0 else 0
                            skip_rate = skipped_samples / processed_samples if processed_samples > 0 else 0
                            
                            log_dict.update({
                                "processed_samples": processed_samples,
                                "filtered_samples": filtered_samples,
                                "skipped_samples": skipped_samples,
                                "filter_rate": filter_rate,
                                "skip_rate": skip_rate,
                            })
                        
                        logger.info(log_dict)

                    if global_step % args.save_steps == 0:
                        test_model(args, wm_model, test_dataloader, device, accelerator)
                        
                        # 保存模型
                        save_step_dir = os.path.join(output_with_time_dir, f"step{global_step}")
                        os.makedirs(save_step_dir, exist_ok=True)
                        save_all(wm_model, save_step_dir, accelerator, args, wm_model_config, pipe, image, wm_image, prompt, output_with_time_dir)
                    
                    if args.enable_output_images and global_step < 4000 and global_step % 10 == 0:
                        _, metrics_A, metrics_B = create_comparison_image(pipe, accelerator, image, wm_image, prompt, output_with_time_dir+'/output_images', global_step)
                        metrics_A_list.append(metrics_A)
                        metrics_B_list.append(metrics_B)
            
            if global_step >= args.max_train_steps:
                finished_flag = True
                break

        if finished_flag:
            break

    if accelerator.is_main_process:
        # 创建简化版的save_all用于最终保存
        def save_final(g_model, save_dir):
            unwrapped_model = accelerator.unwrap_model(g_model)
            accelerator.save(unwrapped_model.state_dict(), os.path.join(save_dir, "wm_model.ckpt"))
            with open(os.path.join(save_dir, "train_config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            OmegaConf.save(wm_model_config, os.path.join(save_dir, "wm_model_config.yaml"))
        
        save_final(wm_model, output_with_time_dir)
        test_model(args, wm_model, test_dataloader, device, accelerator)
        
        # 最终统计（只在开启筛选时）
        if args.enable_online_filter and diff_values_all:
            final_avg_diff_value = np.mean(diff_values_all)
            final_filter_rate = filtered_samples / processed_samples if processed_samples > 0 else 0
            final_skip_rate = skipped_samples / processed_samples if processed_samples > 0 else 0
            
            # 保存最终统计信息
            final_stats_file = os.path.join(output_with_time_dir, "final_statistics.json")
            stats = {
                "processed_samples": processed_samples,
                "filtered_samples": filtered_samples,
                "skipped_samples": skipped_samples,
                "filter_rate": final_filter_rate,
                "skip_rate": final_skip_rate,
                "avg_diff_value": final_avg_diff_value,
                "filter_threshold": args.filter_threshold,
                "realtime_filter_enabled": True
            }
            
            with open(final_stats_file, "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

        # 保存metrics_A和metrics_B
        if args.enable_output_images:
            # 计算metrics_A的各项指标均值
            metrics_A_psnr = np.mean([m['psnr'] for m in metrics_A_list])
            metrics_A_ssim = np.mean([m['ssim'] for m in metrics_A_list])
            metrics_A_l1 = np.mean([m['l1'] for m in metrics_A_list])
            metrics_A_l2 = np.mean([m['l2'] for m in metrics_A_list])
            metrics_A_edit_ratio = np.mean([m['edit_ratio'] for m in metrics_A_list])
            
            # 计算metrics_B的各项指标均值
            metrics_B_psnr = np.mean([m['psnr'] for m in metrics_B_list])
            metrics_B_ssim = np.mean([m['ssim'] for m in metrics_B_list])
            metrics_B_l1 = np.mean([m['l1'] for m in metrics_B_list])
            metrics_B_l2 = np.mean([m['l2'] for m in metrics_B_list])
            metrics_B_edit_ratio = np.mean([m['edit_ratio'] for m in metrics_B_list])
            
            logger.info(f"metrics_A - PSNR: {metrics_A_psnr:.4f}, SSIM: {metrics_A_ssim:.4f}, L1: {metrics_A_l1:.4f}, L2: {metrics_A_l2:.4f}, Edit Ratio: {metrics_A_edit_ratio:.4f}")
            logger.info(f"metrics_B - PSNR: {metrics_B_psnr:.4f}, SSIM: {metrics_B_ssim:.4f}, L1: {metrics_B_l1:.4f}, L2: {metrics_B_l2:.4f}, Edit Ratio: {metrics_B_edit_ratio:.4f}")
            
    accelerator.end_training()

    # 只在主进程执行，结束训练后自动调用绘图脚本
    if accelerator.is_main_process:
        log_file_path = os.path.join(output_with_time_dir, "log.txt")
        logger.info(f"训练结束，开始生成绘图，日志路径: {log_file_path}")
        os.system(f"python custom/log2plt.py -l {log_file_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_size", type=int, default=20000)
    parser.add_argument("--test_size", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--message_length", type=int, default=256)
    parser.add_argument("--max_train_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--decoder_weight", type=float, default=1.5)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--wm_model_config", type=str, default=None)
    parser.add_argument("--last_grad_steps", type=int, default=3)
    parser.add_argument("--enc_latent_weight", type=float, default=None)
    parser.add_argument("--filter_threshold", type=float, default=0.3, help="Threshold for filtering out samples")
    parser.add_argument("--enable_online_filter", action="store_true", default=False, help="Enable real-time filtering")
    parser.add_argument("--filter_method", type=str, default="edit_ratio", help="Method for filtering out samples")
    parser.add_argument("--enable_offline_filter", action="store_true", default=False, help="Enable offline filtering")
    parser.add_argument("--enable_output_images", action="store_true", default=False, help="Enable output images")
    args = parser.parse_args()
    main(args)