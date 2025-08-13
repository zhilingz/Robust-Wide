# sbatch custom/llm_test.sh
print("start llm_test.py")
from accelerate.utils import set_seed
set_seed(42)

# 1. instruct-pix2pix-distill with LCM specified scheduler
# from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler
# import torch
# from diffusers.utils import load_image

# init_image = load_image("inference_results/timbrooks___instructpix2pix-clip-filtered/original_2.png")

# pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
#        "/public/zhangzhiling/models/timbrooks/instruct-pix2pix-distill", 
#        torch_dtype=torch.float16,
#        local_files_only=True
#        )
# pipe = pipe.to("cuda")
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", 
#                        local_files_only=True, 
#                        weight_name="pytorch_lora_weights.safetensors")

# edit_instruction = "put her in a windmill"
# image = pipe(prompt=edit_instruction, 
#              image=init_image,
#              num_inference_steps=4, 
#              guidance_scale=2.0,
#              image_guidance_scale=1.0,
#              ).images[0]

# image.save("inference_results/llm_test4_1.0.png")



# 2. MagicBrush
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import os
image_path = "inference_results/timbrooks___instructpix2pix-clip-filtered/original_2.png"
# image_path = "/public/zhangzhiling/code/Robust-Wide/train_results/2025-06-08T11-00-36_timbrooks___instructpix2pix-clip-filtered_magicbrush-jul7/step4000/image.png"
# wm_image_path = "/public/zhangzhiling/code/Robust-Wide/train_results/2025-06-08T11-00-36_timbrooks___instructpix2pix-clip-filtered_magicbrush-jul7/step4000/wm_image.png"
def load_local_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
prompt = "put her in a windmill"
image = load_local_image(image_path)
# wm_image = load_local_image(wm_image_path)
# prompt = "have her be a zombie"

num_inference_steps = 3
image_guidance_scale = 1.0
guidance_scale = 2.0
# num_inference_steps = 20
# image_guidance_scale = 1.5
# guidance_scale = 7
class MagicBrush():
    def __init__(self, weight="/public/zhangzhiling/models/vinesmsuic/magicbrush-jul7"):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        weight, 
                        torch_dtype=torch.float16,
                        local_files_only=True,
                        use_safetensors=False
                    )
        # self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(
            "latent-consistency/lcm-lora-sdv1-5", local_files_only=True,
            weight_name="pytorch_lora_weights.safetensors")
        from diffusers import LCMScheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.fuse_lora()
        self.pipe = self.pipe.to("cuda")
    def infer_one_image(self, src_image, instruct_prompt, seed):
        generator = torch.manual_seed(seed)
        image = self.pipe(
            instruct_prompt, 
            image=src_image, 
            num_inference_steps=num_inference_steps, 
            image_guidance_scale=image_guidance_scale, 
            guidance_scale=guidance_scale, 
            generator=generator).images
        return image
    
model = MagicBrush()
# model.pipe.text_encoder.eval()
# model.pipe.unet.eval()
# model.pipe.vae.eval()

# with torch.no_grad():
images = model.infer_one_image(image, prompt, 42)
save_dir = "inference_results/llm_test_MagicBrush_"+str(num_inference_steps)+"_"+str(image_guidance_scale)+"_"+str(guidance_scale)+".png"
images[0].save(save_dir)
print(save_dir)


# image_output2 = model.infer_one_image(wm_image, prompt, 42)
# output_path2 = "inference_results/generated_image_7.png"
# image_output2.save(output_path2)

# 调用脚本，生成inference、频谱图、编辑区域图、log图
# cmd = f'python custom/diff.py --before "{image_path}" --after "{output_path}"'
# os.system(cmd)


# 3. SD Turbo
# from diffusers import AutoPipelineForImage2Image
# from diffusers.utils import load_image
# import torch

# pipe = AutoPipelineForImage2Image.from_pretrained("/public/zhangzhiling/models/stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16",local_files_only=True)
# pipe.to("cuda")

# init_image = load_image("/public/zhangzhiling/code/Robust-Wide/inference_results/timbrooks___instructpix2pix-clip-filtered/original_6.png").resize((512, 512))
# prompt = "make the braids pink"

# image = pipe(prompt, image=init_image, num_inference_steps=4, strength=0.3, guidance_scale=0.0).images[0]
# image.save("/public/zhangzhiling/code/Robust-Wide/inference_results/junk.png")


# # 4. InstructPix2Pix
# import os
# import PIL
# import torch
# from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# model_id = "/public/zhangzhiling/models/timbrooks/instruct-pix2pix"
# pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
#     model_id, torch_dtype=torch.float16, local_files_only=True,safety_checker=None)
# pipe.to("cuda")
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# before_image = "inference_results/timbrooks___instructpix2pix-clip-filtered/original_2.png"
# # before_image = "/public/zhangzhiling/code/Robust-Wide/train_results/2025-06-08T11-00-36_timbrooks___instructpix2pix-clip-filtered_magicbrush-jul7/step4000/image.png"
# image = PIL.Image.open(before_image)
# # image = PIL.Image.open("examples/Venus.jpg")
# image = image.convert("RGB")
# prompt = "put her in a windmill"
# # prompt = "turn him into cyborg"
# # prompt = "have her be a zombie"
# num_inference_steps = 10
# image_guidance_scale = 1.5
# guidance_scale = 5
# images = pipe(prompt, image=image, 
#               num_inference_steps=num_inference_steps, 
#               image_guidance_scale=image_guidance_scale, 
#               guidance_scale=guidance_scale,
#               ).images

# save_dir = "inference_results/llm_test_instructpix2pix_"+str(num_inference_steps)+"_"+str(image_guidance_scale)+"_"+str(guidance_scale)+".png"
# images[0].save(save_dir)
# print(save_dir)

# 计算生成图像和原图的指标
import torch.nn.functional as F
from torchvision import transforms
from kornia.metrics import psnr, ssim
import lpips

# 将PIL图像转换为tensor
def pil_to_tensor(pil_image):
    """将PIL图像转换为tensor，值域[-1, 1]"""
    tensor = transforms.ToTensor()(pil_image)  # [0, 1]
    tensor = tensor * 2 - 1  # [-1, 1]
    return tensor.unsqueeze(0)  # 添加batch维度

# 转换图像
original_tensor = pil_to_tensor(image).to("cuda")  # 原图
generated_tensor = pil_to_tensor(images[0]).to("cuda")  # 生成图

# 转换到[0,1]用于PSNR和SSIM计算
original_01 = (original_tensor + 1) / 2
generated_01 = (generated_tensor + 1) / 2

# 计算指标
psnr_value = psnr(generated_01, original_01, max_val=1.0).item()
ssim_value = torch.mean(ssim(generated_01, original_01, window_size=5)).item()
l1_value = F.l1_loss(generated_tensor, original_tensor).item()
l2_value = F.mse_loss(generated_tensor, original_tensor).item()
lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to("cuda")  # 使用AlexNet作为特征提取器
lpips_value = lpips_fn(original_tensor, generated_tensor).item()


# 输出结果
print(f"图像质量指标:")
print(f"PSNR: {psnr_value:.4f} dB")
print(f"SSIM: {ssim_value:.4f}")
print(f"L1 Loss: {l1_value:.6f}")
print(f"L2 Loss: {l2_value:.6f}")
print(f"LPIPS: {lpips_value:.6f}")

# # 调用脚本，生成inference、频谱图、编辑区域图、log图
# cmd = f'python custom/diff.py --before "{before_image}" --after "{save_dir}"'
# os.system(cmd)