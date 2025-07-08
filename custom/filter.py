import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, Features, Value, Image, concatenate_datasets
from PIL import Image as PILImage
from torchvision import transforms
from kornia.metrics import psnr, ssim
import json
import logging
import datetime
import pytz
import argparse
from custom.custom_insp2p import CustomStableDiffusionInstructPix2PixPipeline

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
    def __init__(self, model_dir="/public/zhangzhiling/models/timbrooks/instruct-pix2pix", weight_dtype=torch.float16, device="cuda"):
        self.device = device
        self.model_dir = model_dir
        self.weight_dtype = weight_dtype
        self.thresholds = {
            'psnr': 20.0, 'ssim': 0.80, 'l1': 0.1, 'l2': 0.01, 'edit_ratio': 0.20 # 'edit_ratio_max': 0.20, 'edit_ratio_min': 0.10
        }
        self.pipe = self.initialize_pipeline()

    def initialize_pipeline(self):
        if "instruct-pix2pix" in self.model_dir or "magicbrush" in self.model_dir:
            pipe = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_dir, torch_dtype=self.weight_dtype, local_files_only=True
            ).to(self.device)
        elif "sd-turbo" in self.model_dir:
            from custom.custom_i2i import CustomStableDiffusionImg2ImgPipeline
            pipe = CustomStableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_dir, torch_dtype=self.weight_dtype, local_files_only=True
            ).to(self.device)
        elif "sd-x2-latent-upscaler" in self.model_dir:
            from custom.custom_sd import CustomStableDiffusionPipeline, CustomStableDiffusionLatentUpscalePipeline
            pipe = CustomStableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", torch_dtype=self.weight_dtype, local_files_only=True
            ).to(self.device)
            upscaler = CustomStableDiffusionLatentUpscalePipeline.from_pretrained(
                self.model_dir, torch_dtype=self.weight_dtype, local_files_only=True
            ).to(self.device)
            pipe = [pipe, upscaler]
        else:
            raise ValueError("model not supported")

        # 不同模型不同scheduler
        if "magicbrush" in self.model_dir:
            from diffusers import EulerAncestralDiscreteScheduler
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif "instruct-pix2pix-distill" in self.model_dir:
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            # Adapt the InstructPix2Pix model using the LoRA parameters
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

            # 冻结参数
        if "sd-x2-latent-upscaler" in self.model_dir:
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

    def generate_image(self, prompt, image, seed=42):
        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            if isinstance(image, PILImage.Image):
                image = transforms.ToTensor()(image).unsqueeze(0) * 2 - 1
                image = image.to(self.device)
            else:
                image = image.to(self.device)
            
            with torch.no_grad():
                if "instruct-pix2pix-distill" in self.model_dir:
                    generated_image = self.pipe(
                        prompt, 
                        image=image, 
                        num_images_per_prompt=1, 
                        num_inference_steps=4,  # 4改为2，测试效果
                        guidance_scale=2.0,     # 使用较小的guidance_scale
                        image_guidance_scale=1.0,  # 使用较小的image_guidance_scale
                        generator=generator,
                        output_type="pt"
                    )
                elif "sd-turbo" in self.model_dir:
                    generated_image = self.pipe(
                        prompt, 
                        image=image, 
                        num_images_per_prompt=1, 
                        num_inference_steps=2,
                        guidance_scale=0.0, 
                        strength=0.5,
                        generator=generator,
                        output_type="pt"
                    ).images     # pipe返回值为StableDiffusionPipelineOutput 类型，需要取images，形状 (1, C, H, W)
                    generated_image = 2 * generated_image - 1 # 将值域从[-1,1]转为[0,1]，防止图片泛白
                elif "magicbrush" in self.model_dir:
                    generated_image = self.pipe(
                        prompt, 
                        image=image, 
                        num_inference_steps=20, 
                        image_guidance_scale=2.0, 
                        guidance_scale=4, 
                        generator=generator,
                        output_type="pt",
                        )
                else:
                    generated_image = self.pipe(
                        prompt, 
                        image=image, 
                        num_images_per_prompt=1, 
                        num_inference_steps=20,
                        guidance_scale=10, 
                        image_guidance_scale=1.5, # 1.5原图保留太少（62.98%的编辑区域），2.0还可以（38.45%的编辑区域）
                        generator=generator,
                        output_type="pt",
                    )
            return generated_image
        except Exception as e:
            print(f"图像生成出错: {e}")
            return None
    
    def _to_tensor(self, image):
        """统一转换为设备上的tensor"""
        if isinstance(image, PILImage.Image):
            tensor = transforms.ToTensor()(image) * 2 - 1
        else:
            tensor = image
        
        tensor = tensor.to(self.device)
        return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
    
    def calculate_edit_ratio(self, before, after, thresh=15, kernel=3):
        try:
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
        except Exception as e:
            print(f"编辑区域占比计算出错: {e}")
            return 0.1
        
    def calculate_metrics(self, before, after):
        try:
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
            try:
                ssim_val = ssim(after_01, before_01, window_size=5)
                metrics['ssim'] = torch.mean(ssim_val).item() if ssim_val.dim() > 0 else ssim_val.item()
            except:
                metrics['ssim'] = 0.5
            
            return metrics, mask
        except Exception as e:
            print(f"指标计算出错: {e}")
            return None, None
    
    def create_comparison_image(self, original_image, generated_image, mask, prompt=None):
        """
        创建包含原图、生成图和mask的合并图像，并在图像上添加prompt文本
        """
        try:
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
        except Exception as e:
            print(f"创建对比图像出错: {e}")
            return None

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

    def filter_and_save_dataset(self, dataset, output_dir, filter_num):
        """
        合并的数据集筛选、日志记录和保存函数
        """
        pass_count = 0
        total_samples = len(dataset)
        
        # 统计指标
        all_metrics = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        generated_images_dir = os.path.join(output_dir, "generated_images")
        os.makedirs(generated_images_dir, exist_ok=True)
        
        logger.info(f"开始筛选数据集，总样本数: {total_samples}")
        logger.info(f"筛选阈值: {self.thresholds}")
        logger.info(f"输出目录: {output_dir}")
        
        # 初始化保存数据结构
        saved_data = {
            "edit_prompt": [],
            "all": [],
            "filter_stats": {}
        }
        
        # 添加一个函数来保存当前进度的metadata
        def save_metadata_checkpoint(current_pass_count, current_metrics, processed_samples):
            """保存当前进度的metadata"""
            checkpoint_stats = {}
            
            if current_metrics:
                # 计算各指标的统计信息
                psnr_values = [m['psnr'] for m in current_metrics]
                ssim_values = [m['ssim'] for m in current_metrics]
                l1_values = [m['l1'] for m in current_metrics]
                l2_values = [m['l2'] for m in current_metrics]
                edit_ratio_values = [m['edit_ratio'] for m in current_metrics]
                
                checkpoint_stats.update({
                    "psnr_mean": np.mean(psnr_values),
                    "ssim_mean": np.mean(ssim_values),
                    "l1_mean": np.mean(l1_values),
                    "l2_mean": np.mean(l2_values),
                    "edit_ratio_mean": np.mean(edit_ratio_values),
                    "pass": current_pass_count,
                    "processed": processed_samples,
                    "total": total_samples,
                    "pass_rate": current_pass_count / processed_samples if processed_samples > 0 else 0
                })
            
            # 更新保存数据的统计信息
            temp_saved_data = saved_data.copy()
            temp_saved_data["filter_stats"].update(checkpoint_stats)
            
            # 保存检查点metadata
            checkpoint_path = os.path.join(output_dir, "metadata.json")
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(temp_saved_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已保存检查点 - 处理样本: {processed_samples}, 通过筛选: {current_pass_count}")
        
        for i, example in enumerate(dataset):
            # 筛选filter_num个样本
            if pass_count >= filter_num:
                break
            try:
                # 查找图像和提示键
                image_key = next((k for k in ["original_image", "source_image", "source_img", "image"] if k in example), None)
                prompt_key = next((k for k in ["edit_prompt", "instruction"] if k in example), None)
                if not image_key or not prompt_key:
                    logger.warning(f"样本 {i}: 缺少必要的键 (image_key: {image_key}, prompt_key: {prompt_key})")
                    continue
                image, prompt = example[image_key], example[prompt_key]

                generated_image = self.generate_image(prompt, image)
                if generated_image is None:
                    logger.warning(f"样本 {i}: 图像生成失败")
                    continue
                
                metrics, mask = self.calculate_metrics(image, generated_image)
                if metrics is None:
                    logger.warning(f"样本 {i}: 指标计算失败")
                    continue

                passed = all([metrics['psnr'] >= self.thresholds['psnr'],
                            metrics['ssim'] >= self.thresholds['ssim'],
                            metrics['l1'] <= self.thresholds['l1'],
                            metrics['l2'] <= self.thresholds['l2'],
                            # metrics['edit_ratio'] <= self.thresholds['edit_ratio_max'],
                            # metrics['edit_ratio'] >= self.thresholds['edit_ratio_min']
                            metrics['edit_ratio'] <= self.thresholds['edit_ratio']
                        ])
                  
                if passed:
                    # 保存通过筛选的样本
                    pass_count += 1
                    
                    all_metrics.append(metrics)

                    # 保存图片
                    image_dir = os.path.join(images_dir, f"{pass_count-1:06d}.png")
                    image.save(image_dir)

                    generated_image_dir = os.path.join(generated_images_dir, f"{pass_count-1:06d}.png")
                    comparison_image = self.create_comparison_image(image, generated_image, mask, prompt)
                    cv2.imwrite(generated_image_dir, comparison_image)

                    # 添加到保存数据
                    saved_data["edit_prompt"].append(prompt)
                    saved_data["all"].append({
                        "image_path": image_dir,
                        "generated_image_path": generated_image_dir,
                        **metrics
                    })

                    # 记录详细日志
                    log_dict = {
                        "sample_idx": i,
                        "psnr": metrics['psnr'],
                        "ssim": metrics['ssim'],
                        "l1": metrics['l1'],
                        "l2": metrics['l2'],
                        "edit_ratio": metrics['edit_ratio'],
                        "pass_rate": pass_count / (i + 1)
                    }
                    logger.info(log_dict)
                
                # 每处理10000个样本保存一次metadata（无论是否通过筛选）
                if (i + 1) % 10000 == 0:
                    save_metadata_checkpoint(pass_count, all_metrics, i + 1)
                    
            except Exception as e:
                logger.error(f"样本 {i} 处理出错: {e}")                
                continue
        
        # 最终统计
        final_stats = {}
        
        if all_metrics:
            # 计算各指标的统计信息
            psnr_values = [m['psnr'] for m in all_metrics]
            ssim_values = [m['ssim'] for m in all_metrics]
            l1_values = [m['l1'] for m in all_metrics]
            l2_values = [m['l2'] for m in all_metrics]
            edit_ratio_values = [m['edit_ratio'] for m in all_metrics]
            
            final_stats.update({
                "psnr_mean": np.mean(psnr_values),
                "ssim_mean": np.mean(ssim_values),
                "l1_mean": np.mean(l1_values),
                "l2_mean": np.mean(l2_values),
                "edit_ratio_mean": np.mean(edit_ratio_values),
                "pass": pass_count,
                "total": total_samples,
                "pass_rate": pass_count / total_samples if total_samples > 0 else 0
            })
        
        # 更新保存数据的统计信息
        saved_data["filter_stats"].update(final_stats)
        
        # 保存最终元数据
        if pass_count > 0:
            with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(saved_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已保存 {pass_count} 个筛选后的样本到 {output_dir}")
        else:
            logger.warning("没有样本通过筛选，未保存任何文件")
        
        logger.info("=" * 50)
        logger.info("筛选完成 - 最终统计:")
        logger.info(json.dumps(final_stats, indent=2, ensure_ascii=False))
        logger.info("=" * 50)

def load_dataset_simple(dataset_dir):
    print("正在加载数据集...")
    dataset = load_dataset(dataset_dir)
    
    # 合并所有分支
    all_splits = list(dataset.keys())
    combined_dataset = concatenate_datasets([dataset[split] for split in all_splits]) if len(all_splits) > 1 else dataset[all_splits[0]]
    
    print(f"数据集大小: {len(combined_dataset)}")
    if len(combined_dataset) > 0:
        print(f"数据集键: {list(combined_dataset[0].keys())}")
    
    return combined_dataset

def load_filtered_dataset(data_dir):
    with open(os.path.join(data_dir, "metadata.json"), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    dataset_dict = {
        "edit_prompt": metadata["edit_prompt"],
        "original_image": [os.path.join(data_dir, "images", path) for path in metadata["image_path"]]
    }
    
    features = Features({
        "edit_prompt": Value("string"),
        "original_image": Image()
    })
    
    return Dataset.from_dict(dataset_dict, features=features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_num", type=int, default=1000, help="筛选数量")
    parser.add_argument("--data_dir", type=str, default="/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f", help="数据集路径")
    parser.add_argument("--model_dir", type=str, default="/public/zhangzhiling/models/timbrooks/instruct-pix2pix", help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./filtered_datasets/minmax1000/", help="输出路径")
    args = parser.parse_args()
    
    filter_num = args.filter_num
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = os.path.join(args.output_dir, args.data_dir.split("/")[4], args.model_dir.split("/")[-1])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设置日志记录
    setup_logging(output_dir)
    
    # 记录配置信息
    config = {
        "dataset_dir": data_dir,
        "output_dir": output_dir,
        "model_dir": model_dir,
        "device": device
    }
    logger.info("筛选配置:")
    logger.info(json.dumps(config, indent=2, ensure_ascii=False))
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 加载数据集和筛选
    dataset = load_dataset_simple(data_dir)
    image_filter = ImageFilter(model_dir=model_dir, device=device)
    
    # 使用带日志记录的筛选函数
    image_filter.filter_and_save_dataset(dataset, output_dir, filter_num)

    print("\n测试加载筛选后的数据集...")
    loaded_dataset = load_filtered_dataset(output_dir)
    print(f"加载成功，大小: {len(loaded_dataset)}")
    
    for i in range(min(3, len(loaded_dataset))):
        example = loaded_dataset[i]
        print(f"样本 {i+1}: {example['edit_prompt'][:50]}...")

if __name__ == "__main__":
    main()