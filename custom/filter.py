import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, Features, Value, Image, concatenate_datasets
from torch.utils.data import DataLoader
from dataset import collate_fn
from PIL import Image as PILImage
from torchvision import transforms
from kornia.metrics import psnr, ssim
import lpips
import json
import logging
import datetime
import pytz
import argparse
from custom.custom_insp2p import CustomStableDiffusionInstructPix2PixPipeline
from dataset import preprocess_train
from functools import partial
from custom.qwen_tie_scorer import QwenTieScorer

# 设置日志记录器
logger = logging.getLogger(__name__)

def setup_logging(output_dir):
    """
    仿照train.py设置日志记录
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "log.txt")
    if os.path.exists(file_path):
        os.remove(file_path)
        
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
    fhlr = logging.FileHandler(file_path)
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)
    
    return logger

class ImageFilter:
    def __init__(self, args, model_dir="/public/zhangzhiling/models/timbrooks/instruct-pix2pix", weight_dtype=torch.float16, device="cuda", 
                 metric_config=None):
        self.device = device
        self.model_dir = model_dir
        self.weight_dtype = weight_dtype
        self.args = args
        
        # 默认配置
        default_config = {
            'psnr': {'enabled': True, 'min': 15.0, 'max': float('inf')},
            'ssim': {'enabled': True, 'min': 0.80, 'max': float('inf')},
            'l1': {'enabled': True, 'min': float('-inf'), 'max': 0.15},
            'l2': {'enabled': True, 'min': float('-inf'), 'max': 0.03},
            'edit_ratio': {'enabled': True, 'min': float('-inf'), 'max': 0.50},
            'lpips': {'enabled': True, 'min': float('-inf'), 'max': 0.5},
            'qwen': {'enabled': True, 'min': 6.0, 'max': float('inf')}
        }
        
        # 使用传入的配置或默认配置
        self.metric_config = metric_config if metric_config is not None else default_config
        
        # 保持向后兼容的thresholds属性
        self.thresholds = {
            metric: config.get('max', float('inf')) if config.get('max', float('inf')) != float('inf') 
                   else config.get('min', float('-inf'))
            for metric, config in self.metric_config.items()
        }
        
        # 初始化LPIPS模型
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)

        # 初始化QwenTieScorer
        self.qwen_tie_scorer = QwenTieScorer(model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        
        self.pipe = self.initialize_pipeline()

    def initialize_pipeline(self):
        if self.model_dir.split("/")[-1] == "magicbrush-jul7":
            pipe = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_dir, torch_dtype=self.weight_dtype, local_files_only=True
            ).to(self.device)
            # load and fuse lcm lora
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(
                "latent-consistency/lcm-lora-sdv1-5", local_files_only=True,
                weight_name="pytorch_lora_weights.safetensors")
        elif self.model_dir.split("/")[-1] == "instruct-pix2pix" and self.args.reverse_filter:
            from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "/public/zhangzhiling/models/timbrooks/instruct-pix2pix",
                torch_dtype=self.weight_dtype,
                local_files_only=True,
                safety_checker=None
            ).to(self.device)
            # 单独设置调度器
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # elif self.model_dir.split("/")[-1] == "sd-turbo":
        #     from custom.custom_i2i import CustomStableDiffusionImg2ImgPipeline
        #     pipe = CustomStableDiffusionImg2ImgPipeline.from_pretrained(
        #         self.model_dir, torch_dtype=self.weight_dtype, local_files_only=True
        #     ).to(self.device)
        # elif self.model_dir.split("/")[-1] == "sd-x2-latent-upscaler":
        #     from custom.custom_sd import CustomStableDiffusionPipeline, CustomStableDiffusionLatentUpscalePipeline
        #     pipe = CustomStableDiffusionPipeline.from_pretrained(
        #         "CompVis/stable-diffusion-v1-4", torch_dtype=self.weight_dtype, local_files_only=True
        #     ).to(self.device)
        #     upscaler = CustomStableDiffusionLatentUpscalePipeline.from_pretrained(
        #         self.model_dir, torch_dtype=self.weight_dtype, local_files_only=True
        #     ).to(self.device)
        #     pipe = [pipe, upscaler]
        else:
            raise ValueError("model not supported")

        # 冻结参数
        if self.model_dir.split("/")[-1] == "sd-x2-latent-upscaler":
            for p in pipe:
                p.freeze_params()
                p.text_encoder.train()
                p.unet.train()
                p.vae.train()
        elif self.model_dir.split("/")[-1] == "instruct-pix2pix" and self.args.reverse_filter:
            pipe.text_encoder.eval()
            pipe.unet.eval()
            pipe.vae.eval() 
        # else:
        #     pipe.freeze_params()
        #     pipe.text_encoder.train()
        #     pipe.unet.train()
        #     pipe.vae.train()

        return pipe

    def generate_image(self, prompt, image, seed=42):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.no_grad():
            if self.model_dir.split("/")[-1] == "instruct-pix2pix" and self.args.reverse_filter:                
                generated_image = self.pipe(
                    prompt,
                    image=image,
                    num_images_per_prompt=1,
                    num_inference_steps=self.args.num_inference_steps, #10
                    guidance_scale=self.args.guidance_scale, #7.5
                    image_guidance_scale=self.args.image_guidance_scale, #1.5
                    generator=generator,
                    output_type="pt",
                ).images[0]
            elif self.model_dir.split("/")[-1] == "magicbrush-jul7":
                generated_image = self.pipe(
                    prompt, 
                    image=image, 
                    num_inference_steps=self.args.num_inference_steps, #3
                    image_guidance_scale=self.args.image_guidance_scale, #1.0
                    guidance_scale=self.args.guidance_scale, #1
                    generator=generator,
                    output_type="pt",
                    )
            # elif self.model_dir.split("/")[-1] == "instruct-pix2pix-distill":
            #     generated_image = self.pipe(
            #         prompt, 
            #         image=image, 
            #         num_images_per_prompt=1, 
            #         num_inference_steps=4,  # 4改为2，测试效果
            #         guidance_scale=2.0,     # 使用较小的guidance_scale
            #         image_guidance_scale=1.0,  # 使用较小的image_guidance_scale
            #         generator=generator,
            #         output_type="pt"
            #     )
            # elif self.model_dir.split("/")[-1] == "sd-turbo":
            #     generated_image = self.pipe(
            #         prompt, 
            #         image=image, 
            #         num_images_per_prompt=1, 
            #         num_inference_steps=2,
            #         guidance_scale=0.0, 
            #         strength=0.5,
            #         generator=generator,
            #         output_type="pt"
            #     ).images     # pipe返回值为StableDiffusionPipelineOutput 类型，需要取images，形状 (1, C, H, W)
            #     generated_image = 2 * generated_image - 1 # 将值域从[-1,1]转为[0,1]，防止图片泛白
            # else:
            #     generated_image = self.pipe(
            #         prompt, 
            #         image=image, 
            #         num_images_per_prompt=1, 
            #         num_inference_steps=20,
            #         guidance_scale=10, 
            #         image_guidance_scale=1.5, # 1.5原图保留太少（62.98%的编辑区域），2.0还可以（38.45%的编辑区域）
            #         generator=generator,
            #         output_type="pt",
            #     )
        return generated_image
    
    def _to_tensor(self, image):
        """统一转换为设备上的tensor"""
        if isinstance(image, PILImage.Image):
            tensor = transforms.ToTensor()(image) * 2 - 1
        else:
            tensor = image
        
        tensor = tensor.to(self.device)
        return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
    
    def calculate_edit_ratio(self, before, after, thresh=15, kernel=3):
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
        
        return mask.sum() / 255 / mask.size, mask
        
    def calculate_metrics(self, before, after, prompt=None):
        before = self._to_tensor(before)
        after = self._to_tensor(after)
        
        # 转换到[0,1]用于PSNR和SSIM
        before_01 = (before + 1) / 2
        after_01 = (after + 1) / 2
        
        # 计算编辑比例和mask
        edit_ratio, mask = self.calculate_edit_ratio(before, after)
        
        metrics = {
            'psnr': psnr(after_01, before_01, max_val=1.0).item(),
            'l1': F.l1_loss(after, before).item(),
            'l2': F.mse_loss(after, before).item(),
            'edit_ratio': edit_ratio
        }
        
        # SSIM计算
        ssim_val = ssim(after_01, before_01, window_size=5)
        metrics['ssim'] = torch.mean(ssim_val).item() if ssim_val.dim() > 0 else ssim_val.item()
        
        # LPIPS计算
        metrics['lpips'] = self.lpips_fn(before, after).item()
        
        # Qwen评分计算
        before_pil = self.tensor_to_pil(before)
        after_pil = self.tensor_to_pil(after)
        metrics['qwen'] = self.qwen_tie_scorer(before_pil, after_pil, prompt)

        return metrics, mask
    
    def create_comparison_image(self, original_image, generated_image, mask, prompt=None):
        """
        创建包含原图、生成图和mask的合并图像，并在图像上添加prompt文本
        """  
        # 转换原图为numpy格式 (BGR for OpenCV)
        if isinstance(original_image, PILImage.Image):
            original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        else:
            original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        # 转换生成图为numpy格式 (BGR for OpenCV)
        generated_np = ((generated_image.squeeze(0).detach().cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()
        if generated_np.shape[0] == 3:  # CHW -> HWC
            generated_np = generated_np.transpose(1, 2, 0)
        generated_cv = cv2.cvtColor(generated_np, cv2.COLOR_RGB2BGR)
        
        # 调整图像大小以匹配
        h, w = original_cv.shape[:2]
        generated_cv = cv2.resize(generated_cv, (w, h))
        
        # 处理mask
        if mask is not None:
            mask_resized = cv2.resize(mask, (w, h))
            # 将单通道mask转换为三通道以便拼接
            mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        else:
            # 如果没有mask，创建一个空白的mask
            mask_bgr = np.zeros_like(original_cv)
        
        # 水平拼接：原图 | 生成图 | mask
        comparison = cv2.hconcat([original_cv, generated_cv, mask_bgr])
        
        # 如果有prompt，在图像底部添加文本
        if prompt:
            prompt = prompt.strip()
            
            # 文本样式设置
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_color = (255, 255, 255)  # 白色文字
            bg_color = (0, 0, 0)  # 黑色背景
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(prompt, font, font_scale, thickness)
            
            # 创建文本背景区域，添加一些padding
            padding = 15
            text_bg_height = text_height + baseline + padding * 2
            text_bg = np.full((text_bg_height, comparison.shape[1], 3), bg_color, dtype=np.uint8)
            
            # 在文本背景上绘制文本（居中显示）
            text_x = padding
            text_y = text_height + padding
            cv2.putText(text_bg, prompt, (text_x, text_y), font, font_scale, 
                        text_color, thickness, cv2.LINE_AA)
            
            # 将文本区域添加到比较图像的底部
            comparison = cv2.vconcat([comparison, text_bg])
        
        return comparison

    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 移除batch维度
        
        # 从[-1, 1]转换到[0, 1]
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        
        # 转换为numpy并调整维度顺序
        numpy_image = tensor.detach().cpu().numpy()
        if numpy_image.shape[0] == 3:  # CHW -> HWC
            numpy_image = numpy_image.transpose(1, 2, 0)
        
        # 转换为uint8并创建PIL图像
        numpy_image = (numpy_image * 255).astype(np.uint8)
        return PILImage.fromarray(numpy_image)

    def filter_and_save_dataset(self, dataset, output_dir, filter_num, original_dataset_path=None, reverse_filter=False):
        """
        合并的数据集筛选、日志记录和保存函数
        只保存通过筛选的样本ID，不保存实际图片和prompt
        """
        pass_count = 0
        total_samples = len(dataset)
        
        # 根据reverse_filter决定数据遍历顺序
        if reverse_filter:
            # 从末尾开始，创建反向索引
            indices = list(range(total_samples - 1, -1, -1))
            logger.info("启用reverse_filter：从数据集末尾开始选取")
        else:
            # 正常顺序
            indices = list(range(total_samples))
        
        # 统计指标，用来保存每一个指标的最大值和最小值以及平均值
        all_metrics = {}
        # 用于计算平均值的累计值和计数
        metrics_sum = {}
        metrics_count = 0
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"开始筛选数据集，总样本数: {total_samples}")
        logger.info(f"筛选配置: {self.metric_config}")
        logger.info(f"输出目录: {output_dir}")
        
        # 初始化保存数据结构 - 只保存样本ID和指标
        saved_data = {
            "filtered_sample_ids": [],  # 保存通过筛选的样本ID
            "original_dataset_path": original_dataset_path,  # 保存原始数据集路径
            "filter_stats": {}
        }
        
        # 保存metadata的函数
        def save_metadata():
            # 计算每个指标的平均值
            final_metrics = {}
            for metric_name in all_metrics:
                final_metrics[metric_name] = {
                    "min": all_metrics[metric_name]["min"],
                    "max": all_metrics[metric_name]["max"],
                    "avg": metrics_sum[metric_name] / metrics_count if metrics_count > 0 else 0
                }
            
            saved_data["filter_stats"] = {
                "pass": pass_count,
                "total": enum_i + 1,
                "pass_rate": pass_count / (enum_i + 1),
                "metrics_stats": final_metrics  # 包含每个指标的最大值、最小值以及平均值
            }
            with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(saved_data, f, ensure_ascii=False, indent=2)
        
        with torch.no_grad():
            for enum_i, idx in enumerate(indices):
                # 筛选filter_num个样本
                if pass_count >= filter_num:
                    break

                # 获取样本并处理
                example = dataset[idx]
                image, prompt = example["image"], example["prompt"]
                
                if not isinstance(image, torch.Tensor):
                    image = torch.tensor(image)
                image = image.to(self.device)
                if len(image.shape) == 4:
                    image = image[0]
                
                assert image.shape[-1] == 512 and image.shape[-2] == 512, f"image size must be 512x512, got {image.shape}"

                generated_image = self.generate_image([prompt] if isinstance(prompt, str) else prompt, image.unsqueeze(0))
                metrics, mask = self.calculate_metrics(image, generated_image, prompt)

                # 简化的筛选逻辑
                passed = True
                for metric_name, metric_value in metrics.items():
                    if metric_name in self.metric_config:
                        config = self.metric_config[metric_name]
                        if not config.get('enabled', True):
                            continue
                        
                        min_val = config.get('min', float('-inf'))
                        max_val = config.get('max', float('inf'))
                        
                        if (min_val != float('-inf') and metric_value < min_val) or \
                           (max_val != float('inf') and metric_value > max_val):
                            passed = False
                            break
                    
                if passed:
                    # 只保存样本ID和指标，不保存图片
                    pass_count += 1
                    
                    # 更新每个指标的最大值、最小值和累计值（用于计算平均值）
                    metrics_count += 1
                    for metric_name, metric_value in metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = {"min": metric_value, "max": metric_value}
                            metrics_sum[metric_name] = metric_value
                        else:
                            all_metrics[metric_name]["min"] = min(all_metrics[metric_name]["min"], metric_value)
                            all_metrics[metric_name]["max"] = max(all_metrics[metric_name]["max"], metric_value)
                            metrics_sum[metric_name] += metric_value
                    
                    saved_data["filtered_sample_ids"].append({"sample_id": idx, "metrics": metrics})
                    
                    logger.info({
                        "sample_idx": idx,
                        "psnr": metrics['psnr'],
                        "ssim": metrics['ssim'],
                        "l1": metrics['l1'],
                        "l2": metrics['l2'],
                        "edit_ratio": metrics['edit_ratio'],
                        "lpips": metrics['lpips'],
                        "qwen": metrics['qwen'],
                        "pass_rate": pass_count / (enum_i + 1)
                    })
                
                # 每通过2000个样本保存一次metadata
                if pass_count % 2000 == 0 and pass_count > 0:
                    save_metadata()
                    logger.info(f"检查点 - 处理: {enum_i + 1}, 通过: {pass_count}")

        # 最终保存
        save_metadata()
        logger.info(f"筛选完成")

def load_dataset_simple(dataset_dir):
    print("正在加载数据集...")
    dataset = load_dataset(dataset_dir)
    
    # 合并所有分支
    all_splits = list(dataset.keys())
    combined_dataset = concatenate_datasets([dataset[split] for split in all_splits]) if len(all_splits) > 1 else dataset[all_splits[0]]
    
    print(f"数据集大小: {len(combined_dataset)}")
    if len(combined_dataset) > 0:
        print(f"数据集键: {list(combined_dataset[0].keys())}")
        
    combined_dataset = combined_dataset.with_transform(partial(preprocess_train, image_size=512))

    return combined_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_num", type=int, default=1000, help="筛选数量")
    parser.add_argument("--data_dir", type=str, default="/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f", help="数据集路径")
    parser.add_argument("--model_dir", type=str, default="/public/zhangzhiling/models/timbrooks/instruct-pix2pix", help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./filtered_datasets/minmax1000/", help="输出路径")
    
    # 为每个指标添加启用/禁用选项
    parser.add_argument("--enable_psnr", type=lambda x: x.lower() == 'true', default=True, help="启用PSNR筛选 (True/False)")
    parser.add_argument("--enable_ssim", type=lambda x: x.lower() == 'true', default=True, help="启用SSIM筛选 (True/False)")
    parser.add_argument("--enable_l1", type=lambda x: x.lower() == 'true', default=True, help="启用L1筛选 (True/False)")
    parser.add_argument("--enable_l2", type=lambda x: x.lower() == 'true', default=True, help="启用L2筛选 (True/False)")
    parser.add_argument("--enable_edit_ratio", type=lambda x: x.lower() == 'true', default=True, help="启用编辑比例筛选 (True/False)")
    parser.add_argument("--enable_lpips", type=lambda x: x.lower() == 'true', default=True, help="启用LPIPS筛选 (True/False)")
    parser.add_argument("--enable_qwen", type=lambda x: x.lower() == 'true', default=True, help="启用Qwen筛选 (True/False)")
    
    # 为每个指标添加最大值和最小值参数
    parser.add_argument("--psnr_min", type=str, nargs='?', const="min", default="15.0", help="PSNR最小值，不指定值时表示无下限")
    parser.add_argument("--psnr_max", type=str, nargs='?', const="max", default="max", help="PSNR最大值，不指定值时表示无上限")
    parser.add_argument("--ssim_min", type=str, nargs='?', const="min", default="0.80", help="SSIM最小值，不指定值时表示无下限")
    parser.add_argument("--ssim_max", type=str, nargs='?', const="max", default="max", help="SSIM最大值，不指定值时表示无上限")
    parser.add_argument("--l1_min", type=str, nargs='?', const="min", default="min", help="L1最小值，不指定值时表示无下限")
    parser.add_argument("--l1_max", type=str, nargs='?', const="max", default="0.15", help="L1最大值，不指定值时表示无上限")
    parser.add_argument("--l2_min", type=str, nargs='?', const="min", default="min", help="L2最小值，不指定值时表示无下限")
    parser.add_argument("--l2_max", type=str, nargs='?', const="max", default="0.03", help="L2最大值，不指定值时表示无上限")
    parser.add_argument("--edit_ratio_min", type=str, nargs='?', const="min", default="min", help="编辑比例最小值，不指定值时表示无下限")
    parser.add_argument("--edit_ratio_max", type=str, nargs='?', const="max", default="0.50", help="编辑比例最大值，不指定值时表示无上限")
    parser.add_argument("--lpips_min", type=str, nargs='?', const="min", default="min", help="LPIPS最小值，不指定值时表示无下限")
    parser.add_argument("--lpips_max", type=str, nargs='?', const="max", default="0.5", help="LPIPS最大值，不指定值时表示无上限")
    parser.add_argument("--qwen_min", type=str, nargs='?', const="min", default="6.0", help="Qwen评分最小值，不指定值时表示无下限")
    parser.add_argument("--qwen_max", type=str, nargs='?', const="max", default="max", help="Qwen评分最大值，不指定值时表示无上限")
    
    # 添加reverse_filter参数
    parser.add_argument("--reverse_filter", type=lambda x: x.lower() == 'true', default=False, help="从数据集末尾开始选取 (True/False)")
    
    # 添加推理参数
    parser.add_argument("--num_inference_steps", type=int, default=10, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导尺度")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5, help="图像引导尺度")
    
    args = parser.parse_args()
    
    filter_num = args.filter_num
    data_dir = args.data_dir
    model_dir = args.model_dir
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 根据parser参数直接生成动态目录名
    def generate_output_dirname(args):
        """根据命令行参数直接生成输出目录名"""
        result = ""
        
        # 检查每个指标是否启用并直接拼接到result
        if args.enable_psnr:
            result += "psnr"
            result += "_" if args.psnr_min == 'min' else f"{args.psnr_min}"
            result += "_" if args.psnr_max == 'max' else f"{args.psnr_max}"

        if args.enable_ssim:
            result += "ssim"
            result += "_" if args.ssim_min == 'min' else f"{args.ssim_min}"
            result += "_" if args.ssim_max == 'max' else f"{args.ssim_max}"

        if args.enable_l1:
            result += "l1"
            result += "_" if args.l1_min == 'min' else f"{args.l1_min}"
            result += "_" if args.l1_max == 'max' else f"{args.l1_max}"

        if args.enable_l2:
            result += "l2"
            result += "_" if args.l2_min == 'min' else f"{args.l2_min}"
            result += "_" if args.l2_max == 'max' else f"{args.l2_max}"

        if args.enable_edit_ratio:
            result += "edit"
            result += "_" if args.edit_ratio_min == 'min' else f"{args.edit_ratio_min}"
            result += "_" if args.edit_ratio_max == 'max' else f"{args.edit_ratio_max}"

        if args.enable_lpips:
            result += "lpips"
            result += "_" if args.lpips_min == 'min' else f"{args.lpips_min}"
            result += "_" if args.lpips_max == 'max' else f"{args.lpips_max}"
        
        if args.enable_qwen:
            result += "qwen"
            result += "_" if args.qwen_min == 'min' else f"{args.qwen_min}"
            result += "_" if args.qwen_max == 'max' else f"{args.qwen_max}"
        
        # 添加filter_num，直接连接
        result += f"num{args.filter_num}"
        
        return result
    
    # 生成新的输出目录名
    dynamic_dirname = generate_output_dirname(args)
    
    # 如果启用了reverse_filter，在目录名后面添加_reverse
    if args.reverse_filter:
        dynamic_dirname += "_reverse"
    
    output_dir = os.path.join(args.output_dir, dynamic_dirname, args.data_dir.split("/")[4], args.model_dir.split("/")[-1])
    print(f"输出目录: {output_dir}")
    
    # 创建指标配置
    def parse_value(value, value_type):
        """解析命令行参数值，处理'min'和'max'特殊值"""
        if isinstance(value, str):
            if value == "min":
                return float('-inf')
            elif value == "max":
                return float('inf')
            else:
                return value_type(value)
        return value
    
    metric_config = {
        'psnr': {
            'enabled': args.enable_psnr,
            'min': parse_value(args.psnr_min, float),
            'max': parse_value(args.psnr_max, float)
        },
        'ssim': {
            'enabled': args.enable_ssim,
            'min': parse_value(args.ssim_min, float),
            'max': parse_value(args.ssim_max, float)
        },
        'l1': {
            'enabled': args.enable_l1,
            'min': parse_value(args.l1_min, float),
            'max': parse_value(args.l1_max, float)
        },
        'l2': {
            'enabled': args.enable_l2,
            'min': parse_value(args.l2_min, float),
            'max': parse_value(args.l2_max, float)
        },
        'edit_ratio': {
            'enabled': args.enable_edit_ratio,
            'min': parse_value(args.edit_ratio_min, float),
            'max': parse_value(args.edit_ratio_max, float)
        },
        'lpips': {
            'enabled': args.enable_lpips,
            'min': parse_value(args.lpips_min, float),
            'max': parse_value(args.lpips_max, float)
        },
        'qwen': {
            'enabled': args.enable_qwen,
            'min': parse_value(args.qwen_min, float),
            'max': parse_value(args.qwen_max, float)
        }
    }

    # 设置日志记录
    setup_logging(output_dir)
    
    # 记录配置信息
    config = {
        "dataset_dir": args.data_dir,
        "output_dir": output_dir,
        "model_dir": args.model_dir,
        "device": device,
        "filter_num": args.filter_num,
        "reverse_filter": args.reverse_filter,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "metric_config": metric_config
    }

    logger.info("筛选配置:")
    logger.info(json.dumps(config, indent=2, ensure_ascii=False))
    logger.info(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '')}")
    if device == "cuda":
        torch.cuda.empty_cache()

    # 加载数据集和筛选
    dataset = load_dataset_simple(data_dir)
    image_filter = ImageFilter(args=args, model_dir=model_dir, device=device, metric_config=metric_config)
    
    # 在metadata中保存原始数据集路径
    image_filter.original_dataset_path = data_dir
    
    # 使用带日志记录的筛选函数
    image_filter.filter_and_save_dataset(dataset, output_dir, filter_num, original_dataset_path=data_dir, reverse_filter=args.reverse_filter)

if __name__ == "__main__":
    main()