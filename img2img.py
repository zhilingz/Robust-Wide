import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from diffusers import FluxFillPipeline
import numpy as np
from torchvision.transforms.functional import to_pil_image
import time

# 编辑模型路径
MODELS = [
    # "/public/zhangzhiling/models/timbrooks/instruct-pix2pix",
    # "/public/zhangzhiling/models/vinesmsuic/magicbrush-jul7",
    "/public/zhangzhiling/models/timbrooks/instruct-pix2pix-distill",
    # "black-forest-labs/FLUX.1-Fill-dev",
    # "/public/zhangzhiling/models/stabilityai/sd-turbo",
    # "/public/zhangzhiling/models/stabilityai/sd-x2-latent-upscaler",
]

# 数据集路径
DATASETS = [
    "/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f",
    "/public/zhangzhiling/datasets/BleachNick___ultra_edit_500k/default/0.0.0/8d78dc552b576027618ff2170c4c1d7bcaf27ad2",
    "/public/zhangzhiling/datasets/osunlp___magic_brush/default/0.0.0/1d8d4629150d18ca50afab66391866f2085be989",
    "/public/zhangzhiling/datasets/facebook___emu_edit_test_set/default/0.0.0/b31936a0b6c267e87d373014034cd8fb44ced2fb"
]

# 你需要根据实际情况导入对应的pipeline
from custom.custom_insp2p import CustomStableDiffusionInstructPix2PixPipeline
from custom.custom_i2i import CustomStableDiffusionImg2ImgPipeline
from custom.custom_sd import CustomStableDiffusionPipeline, CustomStableDiffusionLatentUpscalePipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

def denormalize(tensor):
    # 假设输入是[-1,1]，转为[0,1]
    return (tensor + 1) / 2

def smart_denormalize(tensor):
    # 检查张量值范围
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    # 如果值已经在 [0,1] 范围内或接近该范围
    if min_val >= -0.1 and max_val <= 1.1:
        # 限制到 [0,1] 范围
        return torch.clamp(tensor, 0, 1)
    # 否则假设是 [-1,1] 范围
    else:
        return (tensor + 1) / 2

def initialize_pipeline(model_dir, weight_dtype, device):
    if "instruct-pix2pix" in model_dir or "magicbrush" in model_dir:
        pipe = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_dir, torch_dtype=weight_dtype, local_files_only=True
        ).to(device)
    elif "sd-turbo" in model_dir:
        pipe = CustomStableDiffusionImg2ImgPipeline.from_pretrained(
            model_dir, torch_dtype=weight_dtype, local_files_only=True
        ).to(device)
    elif "sd-x2-latent-upscaler" in model_dir:
        pipe = CustomStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=weight_dtype, local_files_only=True
        ).to(device)
        upscaler = CustomStableDiffusionLatentUpscalePipeline.from_pretrained(
            model_dir, torch_dtype=weight_dtype, local_files_only=True
        ).to(device)
        pipe = [pipe, upscaler]
    elif "FLUX" in model_dir:
        pipe = FluxFillPipeline.from_pretrained(
            model_dir, torch_dtype=weight_dtype, local_files_only=True
        )
        # 请根据实际情况替换 LoRA 权重路径
        # pipe.load_lora_weights(
        #     "sanaka87/ICEdit-MoE-LoRA", local_files_only=True,
        #     weight_name="ICEdit-MoE-LoRA.safetensors")
        pipe.load_lora_weights(
            "RiverZ/normal-lora", local_files_only=True, 
            weight_name="pytorch_lora_weights.safetensors",
            lora_scale=1.0)
        pipe = pipe.to(device)
    else:
        raise ValueError("model not supported")

    # 不同模型不同scheduler
    if "magicbrush" in model_dir:
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif "instruct-pix2pix-distill" in model_dir:
        from diffusers import LCMScheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        # Adapt the InstructPix2Pix model using the LoRA parameters
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", local_files_only=True, weight_name="pytorch_lora_weights.safetensors")

    return pipe

def generate_image(model_dir, pipe, prompt, image, last_grad_steps=3):
    if "instruct-pix2pix-distill" in model_dir:
        generated_image = pipe(
            prompt, 
            image=image, 
            num_images_per_prompt=1, 
            num_inference_steps=4,
            guidance_scale=2.0,
            image_guidance_scale=1.0,
            last_grad_steps=last_grad_steps,
            output_type="pt"
        )
    elif "sd-turbo" in model_dir:
        generated_image = pipe(
            prompt, 
            image=image, 
            num_images_per_prompt=1, 
            num_inference_steps=2,
            guidance_scale=0.0, 
            strength=0.5,
            last_grad_steps=last_grad_steps,
            output_type="pt"
        ).images
        generated_image = 2 * generated_image - 1
    elif "sd-x2-latent-upscaler" in model_dir:
        with torch.cuda.amp.autocast():
            latent_dist = pipe[0].vae.encode(image).latent_dist
            low_res_latents = latent_dist.mean * pipe[0].vae.config.scaling_factor
        generated_image = pipe[1](
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=0,
            generator=torch.manual_seed(33),
            last_grad_steps=last_grad_steps,
            output_type="pt"
        )
        generated_image = F.interpolate(generated_image, size=(512, 512), mode='bilinear', align_corners=False)
        generated_image = 2 * generated_image - 1
    elif "FLUX" in model_dir:
        # 将 tensor 转成 PIL 图像
        pil_img = to_pil_image(denormalize(image[0].cpu()))
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
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="pt"
        ).images[0]
        # 裁剪右半部分并转回 tensor
        cropped = result.crop((width, 0, width*2, height))
        generated_image = transforms.ToTensor()(cropped).unsqueeze(0).to(device)
    elif "magicbrush" in model_dir:
        generated_image = pipe(
            prompt, 
            image=image, 
            num_inference_steps=20, 
            image_guidance_scale=1.5, 
            guidance_scale=7, 
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="pt"
            )
    else:
        generated_image = pipe(
            prompt, 
            image=image, 
            num_images_per_prompt=1, 
            num_inference_steps=20,
            guidance_scale=10, 
            image_guidance_scale=2.0, 
            last_grad_steps=last_grad_steps,
            output_type="pt",
        )
    # 兼容返回类型
    if hasattr(generated_image, "images"):
        return generated_image.images
    return generated_image

def get_image_and_prompt(example, image_size=512):
    # 兼容不同数据集的字段
    image_keys = ["original_image", "source_image", "source_img", "image"]
    prompt_keys = ["edit_prompt", "instruction", "prompt"]
    img = None
    for k in image_keys:
        if k in example:
            img = example[k]
            break
    if img is None:
        raise ValueError("No image key found in example")
    prompt = None
    for k in prompt_keys:
        if k in example:
            prompt = example[k]
            break
    if prompt is None:
        raise ValueError("No prompt key found in example")
    # 转为tensor
    img = img.convert("RGB").resize((image_size, image_size))
    img = transforms.ToTensor()(img)
    img = 2 * img - 1  # [-1,1]
    img = img.unsqueeze(0).to(device)
    return img, prompt

def main():
    start_time = time.time()
    os.makedirs("inference_results", exist_ok=True)
    
    all_data_to_process = []  # 用于存储所有数据集的图像和提示
    total_datasets = len(DATASETS)

    # 预加载数据
    print("开始预加载数据")
    for dataset_idx, dataset_path in enumerate(DATASETS):
        dataset_name = dataset_path.split("/")[-4]
        save_dir_dataset_specific = os.path.join("inference_results", dataset_name)
        os.makedirs(save_dir_dataset_specific, exist_ok=True)
        
        dataset = load_dataset(dataset_path)
        all_splits = list(dataset.keys())
        split_to_use = all_splits[0]
        ds = dataset[split_to_use]

        prompts_file_path = os.path.join(save_dir_dataset_specific, "prompts.txt")
        with open(prompts_file_path, "w", encoding="utf-8") as pf:
            num_samples_to_process = min(10, len(ds))
            print(f"数据集 [{dataset_idx+1}/{total_datasets}]: {dataset_name}, 加载 {num_samples_to_process} 个样本")
            
            for idx in range(num_samples_to_process):
                example = ds[idx]
                image, prompt = get_image_and_prompt(example)
                
                # 保存原始图像
                original_image_path = os.path.join(save_dir_dataset_specific, f"original_{idx}.png")
                save_image(denormalize(image[0].cpu()), original_image_path)
                
                # 写入prompt
                pf.write(f"{idx}\t{prompt}\n")
                
                # 存储待处理数据
                all_data_to_process.append({
                    "image": image,
                    "prompt": prompt,
                    "dataset_name": dataset_name,
                    "original_idx": idx
                })

    print(f"数据预加载完成，共 {len(all_data_to_process)} 条数据")

    # 加载模型并处理数据
    weight_dtype = torch.float16
    total_models = len(MODELS)
    
    for model_idx, model_path in enumerate(MODELS):
        model_name = model_path.split("/")[-1]
        print(f"加载模型 [{model_idx+1}/{total_models}]: {model_name}")
        pipe = initialize_pipeline(model_path, weight_dtype, device)
        
        # 处理所有预加载的数据
        print(f"模型 {model_name} 开始处理 {len(all_data_to_process)} 条数据")
        processed_count = 0
        
        for data_item in all_data_to_process:
            image = data_item["image"]
            prompt = data_item["prompt"]
            dataset_name = data_item["dataset_name"]
            original_idx = data_item["original_idx"]
            
            current_save_dir = os.path.join("inference_results", dataset_name)
            output_path = os.path.join(current_save_dir, f"{model_name}_{original_idx}.png")
            
            # 仅在每10个样本处理后打印一次进度，减少日志输出
            if processed_count % 10 == 0 or processed_count == len(all_data_to_process) - 1:
                print(f"模型 {model_name} 正在处理: {dataset_name}/{original_idx}, 进度: {processed_count+1}/{len(all_data_to_process)}")
            
            with torch.no_grad():
                edited_image_tensor = generate_image(model_path, pipe, prompt, image)
            
            # 兼容返回类型并保存
            if isinstance(edited_image_tensor, torch.Tensor):
                img_to_save = edited_image_tensor[0].detach().cpu()
            else:
                img_to_save = edited_image_tensor[0][0].detach().cpu()
            
            save_image(denormalize(img_to_save), output_path)
            processed_count += 1
        
        print(f"模型 {model_name} 处理完成，释放资源")
        del pipe
        torch.cuda.empty_cache()

    print("所有模型处理完毕")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"脚本总运行时间: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()