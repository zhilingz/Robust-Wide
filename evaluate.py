import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf
from kornia.metrics import psnr, ssim
from model import WatermarkModel
from datasets import load_dataset
from diffusers import StableDiffusionInstructPix2PixPipeline
from utils import (
    denormalize,
    decoded_message_error_rate,
)


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


def load_wm_model(ckpt_dir, wm_model_config_path=None):
    if wm_model_config_path is None:
        wm_model_config_path = os.path.join(os.path.join(ckpt_dir, "wm_model_config.yaml"))
    wm_model_config = OmegaConf.load(wm_model_config_path)
    message_length = wm_model_config["wm_enc_config"]["message_length"]
    model = WatermarkModel(**wm_model_config)
    model_ckpt = torch.load(os.path.join(ckpt_dir, "wm_model.ckpt"), map_location='cpu')
    model.load_state_dict(model_ckpt)
    model.eval()
    return model, message_length


@torch.no_grad()
def main(ckpt_dir, eval_img_dir, device="cuda:0"):
    insp2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    wm_model, message_length = load_wm_model(ckpt_dir=ckpt_dir)
    wm_model = wm_model.to(device)
    eval_dataset = load_dataset(eval_img_dir)['train']
    psnr_list, ssim_list, ber_list = [], [], []

    for data_dict in tqdm(eval_dataset):
        image_path = data_dict['original_image']
        instruction = data_dict['edit_prompt']

        orig_image = load_image(image_path, 512).to(device)
        message = torch.randint(0, 2, size=(1, message_length)).float().to(device)
        wm_image = wm_model.encoder(orig_image, message)
        edited_wm_image = insp2p_pipe(
            instruction, image=wm_image, num_inference_steps=20, guidance_scale=10, image_guidance_scale=1.5
        ).images[0]
        edited_wm_image = load_image(edited_wm_image, edited_wm_image.size[0]).to(device)
        watermark = wm_model.decoder(edited_wm_image)
        
        psnr_value = psnr(denormalize(wm_image), denormalize(orig_image), 1)
        ssim_value = torch.mean(ssim(denormalize(wm_image), denormalize(orig_image), window_size=5))
        ber = decoded_message_error_rate(message[0], watermark[0])
        
        psnr_list.append(psnr_value.item())
        ssim_list.append(ssim_value.item())
        ber_list.append(ber)
    
        print(f"psnr: {np.mean(psnr_list)}, ssim: {np.mean(ssim_list)}, ber: {np.mean(ber_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--eval_img_dir', type=str)
    args = parser.parse_args()
    main(ckpt_dir=args.ckpt_dir, eval_img_dir=args.eval_img_dir, device='cuda:0')
