"""
全面水印测评脚本
包括三个场景的测试：
1. 场景1: 无失真 - 测试在没有任何失真的情况下水印的恢复能力
2. 场景2: 图像编辑失真 - 测试经过生成模型编辑后水印的鲁棒性
3. 场景3: 通用失真 - 测试经过各种图像处理操作后水印的鲁棒性
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import io
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.datasets import CocoDetection

import kornia as K
from kornia.metrics import psnr, ssim
import lpips

from datasets import load_dataset
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL

import sys
from pathlib import Path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from utils import (
    denormalize,
    decoded_message_error_rate,
    tensor_to_pil,
    normalize,
)


def compute_residual(img1, img2):
    """
    计算两个图像之间的残差并归一化
    
    Args:
        img1: 第一个图像张量
        img2: 第二个图像张量
    
    Returns:
        归一化后的残差图像张量
    """
    residual = img1 - img2
    residual_abs = torch.abs(residual)
    residual_abs_max = torch.max(residual_abs).item()
    residual_abs_min = torch.min(residual_abs).item()
    
    if residual_abs_max - residual_abs_min < 1e-8:
        residual_image = torch.zeros_like(residual_abs)
    else:
        residual_image = (residual_abs - residual_abs_min) / (residual_abs_max - residual_abs_min)
    
    return normalize(residual_image)

def save_distortion_visualization(orig_image, wm_image, distorted_image, save_path):
    """
    保存失真可视化图片
    
    图片布局（2行3列）：
    第一行：原图        | 水印图      | 原图和水印图的残差
    第二行：空白(白色)  | 失真图      | 失真图和水印图的残差
    
    Args:
        orig_image: 原始图像张量 [1, C, H, W]
        wm_image: 水印图像张量 [1, C, H, W]
        distorted_image: 失真图像张量 [1, C, H, W]
        save_path: 保存路径
    """
    # 计算残差
    residual_orig_wm = compute_residual(wm_image, orig_image)
    residual_dist_wm = compute_residual(distorted_image, wm_image)
    
    # 转为PIL图像
    pil_orig = tensor_to_pil(orig_image[0].detach().cpu())[0]
    pil_wm = tensor_to_pil(wm_image[0].detach().cpu())[0]
    pil_res_orig_wm = tensor_to_pil(residual_orig_wm[0].detach().cpu())[0]
    pil_distorted = tensor_to_pil(distorted_image[0].detach().cpu())[0]
    pil_res_dist_wm = tensor_to_pil(residual_dist_wm[0].detach().cpu())[0]
    
    # 创建2行3列的图像布局
    w, h = pil_orig.size
    combined = Image.new('RGB', (w * 3, h * 2), color=(255, 255, 255))
    
    # 第一行：原图 | 水印图 | 原图和水印图的残差
    combined.paste(pil_orig, (0, 0))
    combined.paste(pil_wm, (w, 0))
    combined.paste(pil_res_orig_wm, (2 * w, 0))
    
    # 第二行：空白 | 失真图 | 失真图和水印图的残差
    combined.paste(pil_distorted, (w, h))
    combined.paste(pil_res_dist_wm, (2 * w, h))
    
    # 保存
    combined.save(save_path)

def load_wm_model(wm_model_name, device='cuda:0'):
    """
    加载水印模型
    Args:
        wm_model_name: 水印模型名称
    Returns:
        model: 包装后的水印模型（统一接口）
        message_length: 水印消息长度
        ckpt_path: 检查点路径
        image_size: 模型要求的输入图像大小
    """
    if wm_model_name == "Robust-Wide":
        from model.Robust_Wide import RobustWideModel
        from omegaconf import OmegaConf
        ckpt_dir = "/public/zhangzhiling/code/Robust-Wide/train_results/old/2025-03-05T19-50-18_timbrooks___instructpix2pix-clip-filtered"
        ckpt_files = [d for d in os.listdir(ckpt_dir) if d.startswith("step")]
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda x: int(x.replace("step", "")))
            ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
            print(f"自动选择最新的checkpoint: {ckpt_path}")
        else:
            ckpt_path = ckpt_dir
            print(f"使用指定的checkpoint目录: {ckpt_path}")
        wm_model_config_path = os.path.join(ckpt_path, "wm_model_config.yaml")
        wm_model_config = OmegaConf.load(wm_model_config_path)
        message_length = wm_model_config["wm_enc_config"]["message_length"]
        image_size = 512  # Robust-Wide使用512的图像大小
        
        model = RobustWideModel(**wm_model_config)
        model_ckpt = torch.load(
            os.path.join(ckpt_path, "wm_model.ckpt"), 
            map_location='cpu', 
            weights_only=True
        )
        model.load_state_dict(model_ckpt)
    elif wm_model_name == "MaskMark":
        from model.MaskWM.models.Mask_Model import MaskWMModel
        from omegaconf import OmegaConf
        name = "D_32bits_vae_ft"  # D_64bits D_32bits_vae_ft D_32bits
        ckpt_path = f'model/MaskWM/checkpoints/{name}.pth'
        model_config = OmegaConf.load(f"model/MaskWM/configs/model/{name}.yaml")
        model = MaskWMModel(**model_config)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
        message_length = model_config["wm_enc_config"]["message_length"]
        image_size = model_config["wm_enc_config"]["image_size"]  # 从配置文件读取图像大小
    elif wm_model_name == "TrustMark":
        # 使用新的TrustMark WatermarkModel类，统一接口
        from model.TrustMark import TrustMarkModel
        model = TrustMarkModel(
            use_ECC=False,
            verbose=True, 
            model_type='Q', 
            encoding_type=None, 
            loadBBoxDetector=False,
            device=device
        )
        message_length = model.get_message_length()
        ckpt_path = f'TrustMarkModel'
        image_size = 512  # TrustMark使用512的图像大小
    elif wm_model_name == "VINE":
        from model.VINE import VINEModel
        print("加载VINE水印模型...")
        message_length = 100  # VINE 默认消息长度
        model = VINEModel(message_length=message_length, device=device)
        ckpt_path = f'Shilin-LU/VINE-R-Enc'
        image_size = 512  # VINE使用512的图像大小
    elif wm_model_name == "WAM":
        from model.WAM import WAMModel
        print("加载WAM水印模型...")
        message_length = 64  # WAM支持64位消息
        model = WAMModel(message_length=message_length, device=device)
        ckpt_path = f'/public/zhangzhiling/code/Robust-Wide/evaluation/model/WAM/checkpoints/wam_mit.pth'
        image_size = 512  # WAM使用512的图像大小
    else:
        raise ValueError(f"未知的水印模型: {wm_model_name}")
    
    # 设置为评估模式
    model.eval()
    
    print(f"模型输入图像大小: {image_size}x{image_size}")
    
    return model, message_length, ckpt_path, image_size

def load_image(imgname, target_size=512) -> torch.Tensor:
    """加载并预处理图像"""
    pil_img = Image.open(imgname).convert('RGB') if isinstance(imgname, str) else imgname
    tform = transforms.Compose([
        transforms.Resize(target_size, antialias=True),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return tform(pil_img)[None, ...]

def apply_jpeg_compression(batch_images, quality=50):
    """
    对批量图像应用JPEG压缩
    
    Args:
        batch_images: 形状为 [batch_size, channels, height, width] 的张量
        quality: JPEG压缩质量 (0-100)
    """
    compressed_images = []
    buffer = io.BytesIO()
    
    for i in range(batch_images.shape[0]):
        img_tensor = denormalize(batch_images[i]).cpu()
        img_pil = transforms.ToPILImage()(img_tensor)
        
        buffer.seek(0)
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        img_compressed = Image.open(buffer).copy()
        compressed_img_tensor = transforms.ToTensor()(img_compressed)
        compressed_images.append(compressed_img_tensor)
    
    buffer.close()
    result = torch.stack(compressed_images).to(batch_images.device)
    return normalize(result)

def apply_distortions(images, distortion_type):
    """
    应用各种图像失真
    
    Args:
        images: 输入图像张量，范围[-1, 1]
        distortion_type: 失真类型
    """
    if distortion_type == "jpeg":
        # JPEG压缩 quality=60
        return apply_jpeg_compression(images, quality=60)
    
    elif distortion_type == "gaussian_filter":
        # 高斯滤波 kernel_size=7 (使用7而不是1，因为kornia需要奇数且>=3), sigma=3
        img_01 = denormalize(images)
        blurred = K.filters.gaussian_blur2d(img_01, kernel_size=(7, 7), sigma=(3.0, 3.0))
        return normalize(blurred)
    
    elif distortion_type == "gaussian_noise":
        # 高斯噪声 mean=0, std=0.05
        noise = torch.randn_like(images) * 0.05
        return torch.clamp(images + noise, -1, 1)
    
    elif distortion_type == "median_filter":
        # 中值滤波 kernel_size=3
        img_01 = denormalize(images)
        blurred = K.filters.median_blur(img_01, kernel_size=(3, 3))
        return normalize(blurred)
    
    elif distortion_type == "salt_pepper_noise":
        # 椒盐噪声 noise_ratio=0.05
        img_01 = denormalize(images)
        prob = 0.05  # 噪声概率
        
        # 生成随机mask
        rnd = torch.rand_like(img_01)
        
        # 盐噪声（白色）
        salt_mask = rnd < prob / 2
        # 椒噪声（黑色）
        pepper_mask = (rnd >= prob / 2) & (rnd < prob)
        
        noisy = img_01.clone()
        noisy[salt_mask] = 1.0
        noisy[pepper_mask] = 0.0
        
        return normalize(noisy)
    
    elif distortion_type == "resize":
        # 缩放 scaling_factor=0.5
        batch, c, h, w = images.shape
        img_01 = denormalize(images)
        small = F.interpolate(img_01, size=(h//2, w//2), mode='bilinear', align_corners=False)
        restored = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
        return normalize(restored)
    
    elif distortion_type == "brightness":
        # 亮度调整 range (0.7, 1.3)
        factor = random.uniform(0.7, 1.3)
        img_01 = denormalize(images)
        adjusted = K.enhance.adjust_brightness(img_01, factor)
        return normalize(adjusted)
    
    elif distortion_type == "contrast":
        # 对比度调整 range (0.7, 1.3)
        factor = random.uniform(0.7, 1.3)
        img_01 = denormalize(images)
        adjusted = K.enhance.adjust_contrast(img_01, factor)
        return normalize(adjusted)
    
    elif distortion_type == "hue":
        # 色调调整 range (-0.1, 0.1)
        factor = random.uniform(-0.1, 0.1)
        img_01 = denormalize(images)
        adjusted = K.enhance.adjust_hue(img_01, factor)
        return normalize(adjusted)
    
    elif distortion_type == "saturation":
        # 饱和度调整 range (0.7, 1.3)
        factor = random.uniform(0.7, 1.3)
        img_01 = denormalize(images)
        adjusted = K.enhance.adjust_saturation(img_01, factor)
        return normalize(adjusted)
    
    elif distortion_type == "rotation":
        # 旋转 angle from [-30°, 30°]
        angle = random.uniform(-30, 30)
        img_01 = denormalize(images)
        rotated = transforms.functional.rotate(img_01, angle)
        return normalize(rotated)
    
    elif distortion_type == "perspective":
        # 透视变换 distortion_scale from [0.1, 0.3]
        distortion_scale = random.uniform(0.1, 0.3)
        img_01 = denormalize(images)
        batch, c, h, w = img_01.shape
        
        # 使用kornia的perspective变换
        # 生成随机的透视变换参数
        perspective_transformed = K.augmentation.RandomPerspective(
            distortion_scale=distortion_scale, 
            p=1.0
        )(img_01)
        return normalize(perspective_transformed)
    
    elif distortion_type == "horizontal_flip":
        # 水平翻转
        img_01 = denormalize(images)
        flipped = torch.flip(img_01, dims=[-1])
        return normalize(flipped)
    
    else:
        return images

@torch.no_grad()
def comprehensive_evaluation(
    wm_model_name,
    output_dir,
    test_size=100,
    device="cuda:0",
    use_coco=False,
    args_dict=None,
):
    """
    全面水印测评函数
    
    Args:
        output_dir: 输出目录
        test_size: 测试样本数量
        device: 设备
        use_coco: 是否使用MS-COCO 2014数据集（如果为True，则不测试图像编辑失真）
        args_dict: 命令行参数字典（可选），用于保存到结果中
    """
    # 加载水印模型
    print("加载水印模型...")
    wm_model, message_length, ckpt_path, image_size = load_wm_model(wm_model_name, device)
    wm_model = wm_model.to(device)
    
    # 加载图像编辑模型 (Instruct-Pix2Pix) - 仅在不使用COCO时加载
    edit_pipe = None
    if not use_coco:
        print("加载Instruct-Pix2Pix模型...")
        edit_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "/public/zhangzhiling/models/timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None
        ).to(device)
        edit_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(edit_pipe.scheduler.config)
        edit_pipe.set_progress_bar_config(disable=True)  # 禁用进度条
        try:
            unet_dev = next(edit_pipe.unet.parameters()).device
            vae_dev = next(edit_pipe.vae.parameters()).device
            te_dev = next(edit_pipe.text_encoder.parameters()).device
            print(f"Instruct-Pix2Pix 组件设备: UNet={unet_dev}, VAE={vae_dev}, TextEncoder={te_dev}")
        except Exception as e:
            print(f"检查 Instruct-Pix2Pix 设备失败: {e}")
    else:
        print("使用MS-COCO数据集，跳过Instruct-Pix2Pix模型加载")
    
    # 初始化LPIPS模型
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
    
    # 固定随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    # 加载评估数据集
    if use_coco:
        # 使用MS-COCO 2014数据集
        print("使用MS-COCO 2014 validation集")
        coco_img_dir = "/public/zhangzhiling/code/Robust-Wide/evaluation/model/MaskWM/data/val2014"
        coco_ann_file = "/public/zhangzhiling/code/Robust-Wide/evaluation/model/MaskWM/data/annotations/instances_val2014.json"
        
        # 创建简单的COCO数据集包装器，只返回图像
        class SimpleCocoDataset(CocoDetection):
            def __getitem__(self, index):
                img_id = self.ids[index]
                img = self._load_image(img_id)
                return {'image': img, 'id': img_id}
        
        eval_dataset = SimpleCocoDataset(root=coco_img_dir, annFile=coco_ann_file)
        # 限制数据集大小
        eval_dataset.ids = eval_dataset.ids[:test_size]
        actual_size = len(eval_dataset)
        print(f"成功加载MS-COCO validation集 {actual_size} 个样本")
    else:
        # 使用InstructPix2Pix数据集
        # 直接加载最后的 test_size 个样本，避免加载整个数据集
        eval_img_dir = "/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f"
        eval_dataset = load_dataset(eval_img_dir, split=f"train[-{test_size}:]")
        actual_size = len(eval_dataset)
        print(f"成功加载{eval_img_dir}数据集最后 {actual_size} 个样本")
    
    # 定义失真类型
    # Valuemetric (值度量失真)
    valuemetric_distortions = [
        "jpeg",
        "gaussian_filter",
        "gaussian_noise",
        "median_filter",
        "salt_pepper_noise",
        "resize",
        "brightness",
        "contrast",
        "hue",
        "saturation",
    ]
    
    # Geometric (几何变换失真)
    geometric_distortions = [
        "rotation",
        "perspective",
        "horizontal_flip",
    ]
    
    # 所有失真类型
    distortion_types = valuemetric_distortions + geometric_distortions 
    
    # 初始化结果收集器
    results = {
        "no_distortion_ber": [],
        "vae_compression_ber": [],
        "edit_distortion_ber": [],
        "psnr": [],
        "ssim": [],
        "lpips": [],
    }
    
    # 为每种失真类型创建收集器
    for dist_type in distortion_types:
        results[f"{dist_type}_ber"] = []
    
    # 创建保存目录
    images_save_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(images_save_dir, exist_ok=True)
    
    print("开始全面测评...")
    print(f"场景1: 无失真")
    if not use_coco:
        print(f"场景2: 图像编辑失真 (Instruct-Pix2Pix)")
    else:
        print(f"场景2: 跳过（使用MS-COCO数据集）")
    print(f"场景3: 通用失真 ({len(distortion_types)} 种失真类型)")
    # 耗时统计
    t_encode_list = []
    t_edit_list = []
    t_common_list = []
    per_dist_time = {k: [] for k in distortion_types}
    
    for idx, data_dict in enumerate(tqdm(eval_dataset, desc="评估进度")):
        # 根据数据集类型获取图像路径
        if use_coco:
            # MS-COCO数据集格式：{'image': PIL.Image, 'id': int, ...}
            if isinstance(data_dict.get('image'), Image.Image):
                image_path = data_dict['image']
            else:
                # 如果是路径字符串
                image_path = data_dict['image']
            instruction = None  # COCO没有编辑指令
        else:
            # InstructPix2Pix数据集格式
            image_path = data_dict['original_image']
            instruction = data_dict['edit_prompt']
        
        # 生成随机水印消息
        message = torch.randint(0, 2, size=(1, message_length)).float().to(device)
        
        # 加载原始图像
        orig_image = load_image(image_path, image_size).to(device)
        
        # 生成水印图像
        t0 = time.perf_counter()
        wm_image = wm_model.encoder(orig_image, message)
        t_encode_list.append(time.perf_counter() - t0)
        
        # 计算图像质量指标
        psnr_value = psnr(denormalize(wm_image), denormalize(orig_image), 1)
        ssim_value = torch.mean(ssim(denormalize(wm_image), denormalize(orig_image), window_size=5))
        lpips_value = lpips_fn(orig_image, wm_image)
        
        results["psnr"].append(psnr_value.item())
        results["ssim"].append(ssim_value.item())
        results["lpips"].append(lpips_value.item())
        
        # ============ 场景1: 无失真 ============
        decoded_message_no_dist = wm_model.decoder(wm_image)
        ber_no_dist = decoded_message_error_rate(message[0], decoded_message_no_dist[0])
        results["no_distortion_ber"].append(ber_no_dist)
        
        # ============ 场景2: 图像编辑失真 ============
        if not use_coco:
            # 2.1 使用VAE压缩（编码-解码）
            # 使用已加载的edit_pipe中的VAE进行压缩
            with torch.no_grad():
                # 将图像编码到latent space
                latents = edit_pipe.vae.encode(wm_image.to(dtype=torch.float16)).latent_dist.sample()
                latents = latents * edit_pipe.vae.config.scaling_factor
                
                # 解码回图像空间
                vae_compressed_image = edit_pipe.vae.decode(latents / edit_pipe.vae.config.scaling_factor).sample
                vae_compressed_image = vae_compressed_image.to(dtype=torch.float32)
            
            # 解码VAE压缩后的水印
            decoded_message_vae = wm_model.decoder(vae_compressed_image)
            ber_vae = decoded_message_error_rate(message[0], decoded_message_vae[0])
            results["vae_compression_ber"].append(ber_vae)
            
            # 保存VAE压缩的可视化（仅第一张图片）
            if idx == 0:
                vae_path = os.path.join(images_save_dir, f"distortion_vae_compression.png")
                save_distortion_visualization(orig_image, wm_image, vae_compressed_image, vae_path)
            
            # 2.2 使用Instruct-Pix2Pix进行图像编辑
            # 转换为PIL图像用于编辑
            wm_image_pil = tensor_to_pil(wm_image[0].cpu())[0]
            
            # 使用Instruct-Pix2Pix进行图像编辑，中等编辑力度
            t1 = time.perf_counter()
            edited_image_pil = edit_pipe(
                instruction,
                image=wm_image_pil,
                num_inference_steps=10,
                guidance_scale=10.0,
                image_guidance_scale=1.5,
            ).images[0]
            t_edit_list.append(time.perf_counter() - t1)
            
            # 转回张量，使用模型期望的图像大小
            edited_image = load_image(edited_image_pil, image_size).to(device)
            
            # 解码水印
            decoded_message_edit = wm_model.decoder(edited_image)
            ber_edit = decoded_message_error_rate(message[0], decoded_message_edit[0])
            results["edit_distortion_ber"].append(ber_edit)

            # 保存编辑失真的可视化（仅第一张图片）
            if idx == 0:
                edit_path = os.path.join(images_save_dir, f"distortion_edit.png")
                save_distortion_visualization(orig_image, wm_image, edited_image, edit_path)
        
        # ============ 场景3: 通用失真 ============
        t2 = time.perf_counter()
        for dist_type in distortion_types:
            td0 = time.perf_counter()
            distorted_image = apply_distortions(wm_image, dist_type)
            decoded_message_dist = wm_model.decoder(distorted_image)
            ber_dist = decoded_message_error_rate(message[0], decoded_message_dist[0])
            results[f"{dist_type}_ber"].append(ber_dist)
            per_dist_time[dist_type].append(time.perf_counter() - td0)
            
            # 对第一张图片，保存每种失真的可视化结果
            if idx == 0:
                distortion_path = os.path.join(images_save_dir, f"distortion_{dist_type}.png")
                save_distortion_visualization(orig_image, wm_image, distorted_image, distortion_path)
        t_common_list.append(time.perf_counter() - t2)
        

    
    # 计算并输出统计结果
    print("\n" + "="*80)
    print("全面测评结果统计")
    print("="*80)
    
    # 图像质量指标
    print(f"\n【图像质量指标】")
    print(f"PSNR:  {np.mean(results['psnr']):.3f} ± {np.std(results['psnr']):.3f} dB")
    print(f"SSIM:  {np.mean(results['ssim']):.4f} ± {np.std(results['ssim']):.4f}")
    print(f"LPIPS: {np.mean(results['lpips']):.4f} ± {np.std(results['lpips']):.4f}")
    
    # 场景1: 无失真
    print(f"\n【场景1: 无失真】")
    print(f"BER: {np.mean(results['no_distortion_ber']):.6f} ± {np.std(results['no_distortion_ber']):.6f}")
    
    # 场景2: 图像编辑失真
    print(f"\n【场景2: 图像编辑失真】")
    if not use_coco:
        print(f"  VAE压缩 BER: {np.mean(results['vae_compression_ber']):.6f} ± {np.std(results['vae_compression_ber']):.6f}")
        print(f"  Instruct-Pix2Pix编辑 BER: {np.mean(results['edit_distortion_ber']):.6f} ± {np.std(results['edit_distortion_ber']):.6f}")
    else:
        print(f"跳过（使用MS-COCO数据集）")
    
    # 场景3: 通用失真
    print(f"\n【场景3: 通用失真】")
    
    # 3.1 Valuemetric失真
    print(f"\n  3.1 Valuemetric (值度量失真):")
    valuemetric_bers = []
    for dist_type in valuemetric_distortions:
        ber_list = results[f"{dist_type}_ber"]
        mean_ber = np.mean(ber_list)
        std_ber = np.std(ber_list)
        valuemetric_bers.append(mean_ber)
        print(f"    {dist_type:20s}: {mean_ber:.6f} ± {std_ber:.6f}")
    print(f"    {'Valuemetric平均':20s}: {np.mean(valuemetric_bers):.6f} ± {np.std(valuemetric_bers):.6f}")
    
    # 3.2 Geometric失真
    print(f"\n  3.2 Geometric (几何变换失真):")
    geometric_bers = []
    for dist_type in geometric_distortions:
        ber_list = results[f"{dist_type}_ber"]
        mean_ber = np.mean(ber_list)
        std_ber = np.std(ber_list)
        geometric_bers.append(mean_ber)
        print(f"    {dist_type:20s}: {mean_ber:.6f} ± {std_ber:.6f}")
    print(f"    {'Geometric平均':20s}: {np.mean(geometric_bers):.6f} ± {np.std(geometric_bers):.6f}")
    
    # 总体平均
    common_dist_bers = valuemetric_bers + geometric_bers
    print(f"\n  通用失真总体平均BER: {np.mean(common_dist_bers):.6f} ± {np.std(common_dist_bers):.6f}")
    # 打印耗时
    if len(t_encode_list) > 0:
        print(f"\n【耗时统计】")
        print(f"编码(encoder) 总耗时: {np.sum(t_encode_list):.3f}s")
        if len(t_edit_list) > 0:
            print(f"编辑失真(Instruct-Pix2Pix) 总耗时: {np.sum(t_edit_list):.3f}s")
        print(f"通用失真(整体) 总耗时: {np.sum(t_common_list):.3f}s")
        # 各通用失真单项耗时
        for k, v in per_dist_time.items():
            if len(v) > 0:
                print(f" - {k:20s}: {np.sum(v):.3f}s")
    
    # 保存详细结果到JSON（格式与logger输出一致）
    # 汇总耗时统计
    timing_summary = {}
    if len(t_encode_list) > 0:
        timing_summary["encoder_total_s"] = float(np.sum(t_encode_list))
    if len(t_edit_list) > 0:
        timing_summary["edit_instructpix2pix_total_s"] = float(np.sum(t_edit_list))
    if len(t_common_list) > 0:
        timing_summary["common_distortions_overall_total_s"] = float(np.sum(t_common_list))
    # 各通用失真单项耗时
    per_dist_timing = {}
    for k, v in per_dist_time.items():
        if len(v) > 0:
            per_dist_timing[k] = float(np.sum(v))
    if len(per_dist_timing) > 0:
        timing_summary["per_distortion_total_s"] = per_dist_timing

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt_path": ckpt_path,
        "test_size": test_size,
        "message_length": message_length,
        "dataset": "MS-COCO 2014" if use_coco else "InstructPix2Pix",
        "image_quality": {
            "psnr": f"{np.mean(results['psnr']):.3f} ± {np.std(results['psnr']):.3f} dB",
            "ssim": f"{np.mean(results['ssim']):.4f} ± {np.std(results['ssim']):.4f}",
            "lpips": f"{np.mean(results['lpips']):.4f} ± {np.std(results['lpips']):.4f}",
        },
        "scenario_1_no_distortion": {
            "ber": f"{np.mean(results['no_distortion_ber']):.6f} ± {np.std(results['no_distortion_ber']):.6f}",
        },
        "scenario_2_edit_distortion": {
            "vae_compression_ber": f"{np.mean(results['vae_compression_ber']):.6f} ± {np.std(results['vae_compression_ber']):.6f}" if not use_coco else "skipped (using MS-COCO)",
            "instruct_pix2pix_ber": f"{np.mean(results['edit_distortion_ber']):.6f} ± {np.std(results['edit_distortion_ber']):.6f}" if not use_coco else "skipped (using MS-COCO)",
        },
        "scenario_3_common_distortions": {
            "average_ber": f"{np.mean(common_dist_bers):.6f} ± {np.std(common_dist_bers):.6f}",
            "valuemetric_average_ber": f"{np.mean(valuemetric_bers):.6f} ± {np.std(valuemetric_bers):.6f}",
            "geometric_average_ber": f"{np.mean(geometric_bers):.6f} ± {np.std(geometric_bers):.6f}",
            "valuemetric_distortions": {},
            "geometric_distortions": {}
        }
    }
    # 合并耗时统计
    if len(timing_summary) > 0:
        summary["timing"] = timing_summary
    
    # 添加Valuemetric失真详情
    for dist_type in valuemetric_distortions:
        mean_ber = np.mean(results[f"{dist_type}_ber"])
        std_ber = np.std(results[f"{dist_type}_ber"])
        summary["scenario_3_common_distortions"]["valuemetric_distortions"][dist_type] = f"{mean_ber:.6f} ± {std_ber:.6f}"
    
    # 添加Geometric失真详情
    for dist_type in geometric_distortions:
        mean_ber = np.mean(results[f"{dist_type}_ber"])
        std_ber = np.std(results[f"{dist_type}_ber"])
        summary["scenario_3_common_distortions"]["geometric_distortions"][dist_type] = f"{mean_ber:.6f} ± {std_ber:.6f}"
    
    # 保存结果（文件名包含当前时间）
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json_path = os.path.join(output_dir, f"{timestamp_str}.json")
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存至: {results_json_path}")
    print("="*80)
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全面水印测评脚本")
    
    parser.add_argument(
        '--wm_model_name', 
        type=str, 
        required=True,
        help='水印模型名字'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str,
        default=None,
        help='输出目录（默认为 checkpoint_dir/comprehensive_eval）'
    )
    
    parser.add_argument(
        '--test_size', 
        type=int,
        default=100,
        help='测试样本数量'
    )
    
    parser.add_argument(
        '--use_coco',
        action='store_true',
        help='使用MS-COCO 2014测试集（此时不测试图像编辑失真）'
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.ckpt_dir, "comprehensive_eval")
    
    # 记录参数
    print("="*80)
    print("全面水印测评")
    print("="*80)
    print(f"参数配置:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("="*80)
    
    # 执行测评
    comprehensive_evaluation(
        wm_model_name=args.wm_model_name,
        output_dir=args.output_dir,
        test_size=args.test_size,
        use_coco=args.use_coco,
        args_dict=vars(args),  # 将所有命令行参数传递进去
    )    
    print("\n全面测评完成！")
