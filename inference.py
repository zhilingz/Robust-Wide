import os
import glob
import torch
from torchvision import transforms
import argparse
from PIL import Image
from omegaconf import OmegaConf
from kornia.metrics import psnr, ssim
from model import WatermarkModel

from utils import (
    normalize,
    denormalize,
    decoded_message_error_rate,
    save_image_for_tensor,
)


def load_wm_model(ckpt_dir, wm_model_config_path=None):
    if wm_model_config_path is None:
        wm_model_config_path = os.path.join(os.path.join(ckpt_dir, "wm_model_config.yaml"))
    wm_model_config = OmegaConf.load(wm_model_config_path)
    message_length = wm_model_config["wm_enc_config"]["message_length"]
    model = WatermarkModel(**wm_model_config)
    model_ckpt = torch.load(os.path.join(ckpt_dir, "wm_model.ckpt"), map_location='cpu', weights_only=True)
    model.load_state_dict(model_ckpt)
    model.eval()
    return model, message_length

def generate_wm_image(wm_model, message_length, instance_image_file, wm_data_dir, size, device="cuda:0",):
    transform_list = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    image_transforms = transforms.Compose(transform_list)
    instance_image = Image.open(instance_image_file).convert("RGB")
    message = torch.randint(0, 2, size=(1, message_length)).float().to(device)

    image = image_transforms(instance_image).unsqueeze(0).to(device)
    wm_image = wm_model.encoder(image, message)
    decoded_message = wm_model.decoder(wm_image)
    img_name = instance_image_file.split("/")[-1].split(".")[0]
    residual = wm_image - image
    residual_abs = torch.abs(residual)
    residual_abs_max = torch.max(residual_abs).item()
    residual_abs_min = torch.min(residual_abs).item()
    residual_image = normalize((residual_abs - residual_abs_min) / (residual_abs_max - residual_abs_min))
    ber = decoded_message_error_rate(message[0], decoded_message[0])
    save_image_for_tensor(image[0], f"{wm_data_dir}/{img_name}_orig.png")
    save_image_for_tensor(wm_image[0], f"{wm_data_dir}/{img_name}_wm.png")
    save_image_for_tensor(residual_image[0], f"{wm_data_dir}/{img_name}_res.png")
    psnr_value = psnr(denormalize(wm_image), denormalize(image), 1)
    ssim_value = torch.mean(ssim(denormalize(wm_image), denormalize(image), window_size=5))

    print(f"{wm_data_dir}/{img_name}_wm.png psnr: {psnr_value}, ssim: {ssim_value}, ber: {ber}")

@torch.no_grad()
def main(ckpt_dir, instance_image_file, wm_data_dir, device="cuda:0",):
    size = 512
    wm_model, message_length = load_wm_model(ckpt_dir=ckpt_dir)
    wm_model = wm_model.to(device)
    os.makedirs(wm_data_dir, exist_ok=True)
    
    # 支持的图片格式
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    # 检查输入是文件还是文件夹
    if os.path.isfile(instance_image_file):
        # 处理单个文件
        print(f"处理单个图片: {instance_image_file}")
        generate_wm_image(wm_model, message_length, instance_image_file, wm_data_dir, size, device=device)
    elif os.path.isdir(instance_image_file):
        # 处理文件夹中的所有图片
        print(f"处理文件夹: {instance_image_file}")
        image_files = []
        
        # 遍历支持的图片格式，查找所有图片文件
        for ext in supported_extensions:
            # 同时支持大小写
            pattern_lower = os.path.join(instance_image_file, f"*{ext.lower()}")
            pattern_upper = os.path.join(instance_image_file, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern_lower))
            image_files.extend(glob.glob(pattern_upper))
        
        # 去重并排序
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            print(f"错误: 在文件夹 {instance_image_file} 中未找到支持的图片文件")
            print(f"支持的格式: {', '.join(supported_extensions)}")
            return
            
        print(f"找到 {len(image_files)} 个图片文件")
        
        # 处理每个图片文件
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {os.path.basename(image_file)}")
            try:
                generate_wm_image(wm_model, message_length, image_file, wm_data_dir, size, device=device)
            except Exception as e:
                print(f"处理文件 {image_file} 时出错: {e}")
                continue
                
        print(f"\n完成! 成功处理了 {len(image_files)} 个图片文件")
    else:
        print(f"错误: 路径 '{instance_image_file}' 既不是文件也不是文件夹")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--image_file', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    main(
        ckpt_dir=args.ckpt_dir, instance_image_file=args.image_file,
        wm_data_dir=args.output_dir, device='cuda:0',
    )
