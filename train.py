import os
import io
import argparse
import datetime, pytz
import json
import logging
import lpips

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision.utils import save_image
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

def generate_image(args, pipe, prompt, wm_image, accelerator, is_test=False):
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    if is_test:
        # 这里的参数你可以根据需求调整
        generated_image = pipe(
            prompt,
            image=wm_image,
            num_images_per_prompt=1,
            num_inference_steps=20,
            guidance_scale=10,
            image_guidance_scale=1.5,
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
                last_grad_steps=args.last_grad_steps,
                output_type="pt"
            ).images     # pipe返回值为StableDiffusionPipelineOutput 类型，需要取images，形状 (1, C, H, W)
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
                generator=torch.manual_seed(33),
                last_grad_steps=args.last_grad_steps,
                output_type="pt"
            )
            # 方法二：将生成的图像调整到512x512大小
            generated_image = F.interpolate(generated_image, size=(512, 512), mode='bilinear', align_corners=False)
        elif "magicbrush" in args.model_dir:
            generated_image = pipe(
                prompt, 
                image=wm_image, 
                num_inference_steps=20, 
                image_guidance_scale=1.5, 
                guidance_scale=7, 
                generator=torch.Generator("cpu").manual_seed(42),
                last_grad_steps=args.last_grad_steps,
                output_type="pt",
                )
        else:
            generated_image = pipe(
                prompt, 
                image=wm_image, 
                num_images_per_prompt=1, 
                num_inference_steps=20,
                guidance_scale=10, 
                image_guidance_scale=1.5, 
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
    
    输出每种场景的比特错误率(BER)
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
            
            # ============ 场景1: 无失真 ============ #
            decoded_message_no_distortion = wm_model.decoder(wm_image.to(dtype=torch.float32))
            error_rate_no_distortion = decoded_message_error_rate_batch(
                message, decoded_message_no_distortion
            )
            
            results["no_distortion"] += error_rate_no_distortion
            
            # ============ 场景2: 图像编辑失真 ============ #
            generated_image = generate_image(args, test_pipe, prompt, wm_image, accelerator, is_test=True)
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
    
    logger.info(log_dict)
    
    # 释放 test_pipe 显存
    del test_pipe  
    torch.cuda.empty_cache()
    
    wm_model.train()
    return log_dict

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

    # ---------- LPIPS ----------
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()   # 不训练 LPIPS 网络
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # ---------- Patch-GAN 判别器 ----------
    class PatchDiscriminator(nn.Module):
        def __init__(self, in_channels=3, ndf=64):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf*4, 1, 4, 1, 1)         # Patch 输出
            )
        def forward(self, x): return self.model(x)

    # 定义鉴别器网络用于GAN损失
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            # 输入为3通道图像(RGB)
            self.main = nn.Sequential(
                # 输入: 3 x 512 x 512
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 状态大小: 64 x 256 x 256
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # 状态大小: 128 x 128 x 128
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # 状态大小: 256 x 64 x 64
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # 状态大小: 512 x 32 x 32
                nn.Conv2d(512, 1, 4, 2, 1, bias=False),
                # 状态大小: 1 x 16 x 16
                nn.AdaptiveAvgPool2d(1),  # 输出: 1 x 1 x 1
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    # D = PatchDiscriminator().to(device)
    D = Discriminator().to(device)
    D.apply(lambda m: nn.init.normal_(m.weight, 0.0, 0.02) if isinstance(m, nn.Conv2d) else None)

    pipe = initialize_pipeline(args, weight_dtype, device)
    print("pipe",pipe)

    train_dataset, test_dataset = get_hugging_dataset(args.train_data_dir, args.image_size, accelerator, args.train_size, args.test_size)
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

    # opt = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate,)
    G_opt = torch.optim.AdamW(
        (p for p in wm_model.parameters() if p.requires_grad), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    D_opt = torch.optim.AdamW(D.parameters(), lr=args.learning_rate*0.5, betas=(0.5, 0.999))

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=G_opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # wm_model, opt, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
    #     wm_model, opt, train_dataloader, test_dataloader, lr_scheduler
    # )
    wm_model, D, G_opt, D_opt, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        wm_model, D, G_opt, D_opt, train_dataloader, test_dataloader, lr_scheduler
    )


    def save_all(g_model, save_dir):
        unwrapped_model = accelerator.unwrap_model(g_model)
        accelerator.save(unwrapped_model.state_dict(), os.path.join(save_dir, "wm_model.ckpt"))
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        OmegaConf.save(wm_model_config, os.path.join(save_dir, "wm_model_config.yaml"))

    step = 0
    global_step = 0
    finished_flag = False
    while True:
        for data in train_dataloader:
            step += 1
            with accelerator.accumulate(wm_model):
                message = torch.randint(0, 2, (args.batch_size, args.message_length)).to(
                    device=device, dtype=torch.float32
                )
                image, prompt = data["image"], data["prompt"]

                wm_image = wm_model.encoder(image, message)

                if "sd-x2-latent-upscaler" in args.model_dir:
                    image_latents = pipe[1].vae.encode(image.to(dtype=weight_dtype)).latent_dist.mode()
                    wm_image_latents = pipe[1].vae.encode(wm_image.to(dtype=weight_dtype)).latent_dist.mode()
                else:
                    image_latents = pipe.vae.encode(image.to(dtype=weight_dtype)).latent_dist.mode()
                    wm_image_latents = pipe.vae.encode(wm_image.to(dtype=weight_dtype)).latent_dist.mode()

                decoded_message_before_edit = wm_model.decoder(wm_image.to(dtype=torch.float32))
                
                generated_image = generate_image(args, pipe, prompt, wm_image, accelerator)
                decoded_message_after_edit = wm_model.decoder(generated_image.to(dtype=torch.float32))

                # Calculate losses, decoder_weight默认0.1
                enc_pixel_loss = F.mse_loss(image.float(), wm_image.float())
                enc_latent_loss = F.mse_loss(image_latents.float(), wm_image_latents.float())
                dec_loss_before_edit = F.mse_loss(message, decoded_message_before_edit)
                dec_loss_after_edit = F.mse_loss(message, decoded_message_after_edit)
                enc_loss = enc_pixel_loss + args.enc_latent_weight * enc_latent_loss
                dec_loss = dec_loss_before_edit + args.decoder_weight * dec_loss_after_edit

                # ---------- LPIPS 感知损失 ----------
                lpips_loss = lpips_fn(wm_image, image).mean()

                # ---------- GAN 损失 ----------
                # 判别器需要对原图标记为 1，对水印图标记为 0
                pred_real = D(image)
                pred_fake = D(wm_image.detach())
                # MSE 损失适合 PatchGAN
                # loss_D = 0.5 * (F.mse_loss(pred_real, torch.ones_like(pred_real)) +
                #                 F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))
                # BCE 损失适合Sigmoid判别器
                loss_D = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real)) + \
                         F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
                
                # 先反向判别器
                accelerator.backward(loss_D)
                D_opt.step(); D_opt.zero_grad()

                # 生成器（编码器）欺骗判别器
                pred_fake_for_G = D(wm_image)
                # gan_loss_G = F.mse_loss(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
                gan_loss_G = F.binary_cross_entropy(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

                # 线性调整 enc_loss 系数
                enc_loss_coeff = 0.1 + 0.9 * min(global_step, args.max_train_steps) / args.max_train_steps
                # loss = enc_loss_coeff * enc_loss + dec_loss
                loss = enc_loss_coeff * enc_loss + dec_loss
                lpips_weight, gan_weight = 0.00001, 0.0
                if lpips_weight > 0:
                    loss += enc_loss_coeff * lpips_weight * lpips_loss
                if gan_weight > 0:
                    loss += enc_loss_coeff * gan_weight * gan_loss_G


                # accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     opt.step()
                #     lr_scheduler.step()
                #     opt.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    G_opt.step()
                    lr_scheduler.step()
                    G_opt.zero_grad()


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
                        logger.info(log_dict)

                    if global_step % args.save_steps == 0:
                        test_model(args, wm_model, test_dataloader, device, accelerator)
                        
                        # 保存模型
                        save_step_dir = os.path.join(output_with_time_dir, f"step{global_step}")
                        os.makedirs(save_step_dir, exist_ok=True)
                        save_all(wm_model, save_step_dir)
                        
                        # 保存编辑图片
                        save_image(denormalize(image[0].detach().cpu()), os.path.join(save_step_dir, "image.png"))
                        save_image(denormalize(wm_image[0].detach().cpu()), os.path.join(save_step_dir, "wm_image.png"))
                        if isinstance(generated_image, torch.Tensor):
                            save_image(denormalize(generated_image[0].detach().cpu()), os.path.join(save_step_dir, "generated_image.png"))
                        else:
                            # 如果 generated_image 是 list 或其他类型
                            save_image(denormalize(generated_image[0][0].detach().cpu()), os.path.join(save_step_dir, "generated_image.png"))
                        # 保存 prompt
                        with open(os.path.join(save_step_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                            if isinstance(prompt, list):
                                f.write(str(prompt[0]))
                            else:
                                f.write(str(prompt))

                        # 调用inference.py
                        image_file = './examples/Gadot.png'
                        cmd = (
                            f'python inference.py '
                            f'--ckpt_dir "{save_step_dir}" '
                            f'--image_file "{image_file}" '
                            f'--output_dir "{save_step_dir}"'
                        )
                        os.system(cmd)
                        
                        logger.info("save models and inference!")

            if global_step >= args.max_train_steps:
                finished_flag = True
                break

        if finished_flag:
            break

    if accelerator.is_main_process:
        save_all(wm_model, output_with_time_dir)
        # 最后一次测试
        test_model(args, wm_model, test_dataloader, device, accelerator)

    accelerator.end_training()

    # 只在主进程执行，结束训练后自动调用绘图脚本
    if accelerator.is_main_process:
        # output_with_time_dir 是 setup_logging 返回的目录
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
    args = parser.parse_args()
    main(args)