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

from dataset import get_hugging_dataset, collate_fn
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
                image_guidance_scale=2.0, # 1.5原图保留太少（62.98%的编辑区域），2.0还可以（38.45%的编辑区域）
                generator=generator,
                last_grad_steps=args.last_grad_steps,
                output_type="pt",
            )
    return generated_image

def setup_logging(args, logger):
    # 获取数据集名称和模型名称
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
    
    logger.info("save models!")

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

    print("Using GPU:", os.environ['CUDA_VISIBLE_DEVICES'])

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
    print("pipe",pipe)

    # 正常加载数据集，不使用筛选
    train_dataset, test_dataset = get_hugging_dataset(
        args.train_data_dir, 
        args.image_size, 
        accelerator, 
        args.train_size, 
        args.test_size
    )
    logger.info("使用完整数据集进行训练")

    # 如果开启实时筛选，batch_size必须为1
    if args.enable_realtime_filter:
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
    
    # 只在开启筛选时初始化统计变量
    if args.enable_realtime_filter:
        diff_values_all = []
        processed_samples = 0  # 处理的样本总数
        filtered_samples = 0   # 通过筛选的样本数
        skipped_samples = 0    # 跳过的样本数
    
    while True:
        for data in train_dataloader:
            step += 1
            
            with accelerator.accumulate(wm_model):
                message = torch.randint(0, 2, (args.batch_size, args.message_length)).to(
                    device=device, dtype=torch.float32
                )
                image, prompt = data["image"], data["prompt"]

                # 开启筛选
                if args.enable_realtime_filter:
                    processed_samples += 1
                    # 1) 先在 no_grad 环境里做粗筛
                    with torch.no_grad():
                        generated_preview =  generate_image(args, pipe, prompt, image, accelerator, device=device)
                        skip_flag = diff_image(image, generated_preview, method=args.filter_method)
                        # diff_values_all.append(diff_value)
                        if skip_flag:
                            # 编辑区域过大，跳过这个样本
                            skipped_samples += 1
                            # logger.info(f"skipped_samples, edit_ratio: {skipped_samples},{edit_ratio}")
                            continue # 直接跳，连计算图都没建
                        else:
                            # 样本通过筛选
                            filtered_samples += 1
                            # logger.info(f"filtered_samples, edit_ratio: {filtered_samples},{edit_ratio}")
                
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

                # # 线性调整 enc_loss 系数
                # enc_loss_coeff = 0.1 + 0.9 * min(global_step, args.max_train_steps) / args.max_train_steps
                # loss = enc_loss_coeff * enc_loss + dec_loss
                
                # Curriculum-Style Weight Scheduling to Accelerate Convergence
                # if args.enable_realtime_filter:
                #     if global_step < 1000:
                #         w_pix, w_lat, w_dec_bf, w_dec_af = 1, 0.001, 1., 0.001
                #     else:
                #         w = min(1., (global_step-1000)/6000)
                #         w_pix, w_lat, w_dec_bf, w_dec_af = 1., 0.001, 1, 0.1*w
                #     loss = (
                #         w_pix * enc_pixel_loss +
                #         w_lat * enc_latent_loss +
                #         w_dec_bf * dec_loss_before_edit +
                #         w_dec_af * dec_loss_after_edit
                #     )
                # else:
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
                        if args.enable_realtime_filter:
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
        if args.enable_realtime_filter and diff_values_all:
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
                "realtime_filter_enabled": True,
                "diff_values_distribution": {
                    "min": float(np.min(diff_values_all)),
                    "max": float(np.max(diff_values_all)),
                    "mean": float(np.mean(diff_values_all)),
                    "std": float(np.std(diff_values_all)),
                    "median": float(np.median(diff_values_all)),
                }
            }
            
            with open(final_stats_file, "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

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
    parser.add_argument("--enable_realtime_filter", action="store_true", default=False, help="Enable real-time filtering")
    parser.add_argument("--filter_method", type=str, default="edit_ratio", help="Method for filtering out samples")
    args = parser.parse_args()
    main(args)