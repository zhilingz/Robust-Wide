




# from PIL import Image, ImageOps
# import torch
# from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
# from PIL import Image

# image_path = "inference_results/Gadot_orig.png"

# def load_local_image(image_path):
#     image = Image.open(image_path)
#     image = ImageOps.exif_transpose(image)
#     image = image.convert("RGB")
#     return image

# image = load_local_image(image_path)
# prompt = "add green hat"

# class MagicBrush():
#     def __init__(self, weight="/public/zhangzhiling/models/vinesmsuic/magicbrush-jul7"):
#         self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
#                         weight, 
#                         torch_dtype=torch.float16,
#                         local_files_only=True
#                     ).to("cuda")
#         self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
#     def infer_one_image(self, src_image, instruct_prompt, seed):
#         generator = torch.manual_seed(seed)
#         image = self.pipe(
#             instruct_prompt, 
#             image=src_image, 
#             num_inference_steps=20, 
#             image_guidance_scale=1.5, 
#             guidance_scale=7, 
#             generator=generator).images[0]
#         return image

# model = MagicBrush()
# image_output = model.infer_one_image(image, prompt, 42)
# image_output.save("inference_results/junk.png")



from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForImage2Image.from_pretrained("/public/zhangzhiling/models/stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16",local_files_only=True)
pipe.to("cuda")

init_image = load_image("/public/zhangzhiling/code/Robust-Wide/inference_results/timbrooks___instructpix2pix-clip-filtered/original_6.png").resize((512, 512))
prompt = "make the braids pink"

image = pipe(prompt, image=init_image, num_inference_steps=4, strength=0.3, guidance_scale=0.0).images[0]
image.save("/public/zhangzhiling/code/Robust-Wide/inference_results/junk.png")

