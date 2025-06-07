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

def load_local_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

image = load_local_image(image_path)
prompt = "put her in a windmill"
num_inference_steps = 20
image_guidance_scale = 2.0
guidance_scale = 4
# num_inference_steps = 20
# image_guidance_scale = 1.5
# guidance_scale = 7
class MagicBrush():
    def __init__(self, weight="/public/zhangzhiling/models/vinesmsuic/magicbrush-jul7"):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        weight, 
                        torch_dtype=torch.float16,
                        local_files_only=True
                    ).to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
    def infer_one_image(self, src_image, instruct_prompt, seed):
        generator = torch.manual_seed(seed)
        image = self.pipe(
            instruct_prompt, 
            image=src_image, 
            num_inference_steps=num_inference_steps, 
            image_guidance_scale=image_guidance_scale, 
            guidance_scale=guidance_scale, 
            generator=generator).images[0]
        return image

model = MagicBrush()
image_output = model.infer_one_image(image, prompt, 42)
output_path = "inference_results/llm_test_magicbrush_"+str(num_inference_steps)+"_"+str(image_guidance_scale)+"_"+str(guidance_scale)+".png"
image_output.save(output_path)
# 调用脚本，生成inference、频谱图、编辑区域图、log图
cmd = f'python custom/diff.py --before "{image_path}" --after "{output_path}"'
os.system(cmd)


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
# print(pipe.scheduler)
# before_image = "inference_results/timbrooks___instructpix2pix-clip-filtered/original_2.png"
# image = PIL.Image.open(before_image)
# # image = PIL.Image.open("examples/Venus.jpg")
# image = image.convert("RGB")
# prompt = "put her in a windmill"
# # prompt = "turn him into cyborg"
# num_inference_steps = 20
# image_guidance_scale = 2.0
# guidance_scale = 10.0    
# images = pipe(prompt, image=image, 
#               num_inference_steps=num_inference_steps, 
#               image_guidance_scale=image_guidance_scale, 
#               guidance_scale=guidance_scale,
#               ).images

# save_dir = "inference_results/llm_test_instructpix2pix_"+str(num_inference_steps)+"_"+str(image_guidance_scale)+"_"+str(guidance_scale)+".png"
# images[0].save(save_dir)
# # 调用脚本，生成inference、频谱图、编辑区域图、log图
# cmd = f'python custom/diff.py --before "{before_image}" --after "{save_dir}"'
# os.system(cmd)