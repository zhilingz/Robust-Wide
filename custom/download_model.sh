export HF_ENDPOINT="https://hf-mirror.com"
source ~/miniconda3/etc/profile.d/conda.sh
# conda activate Robust-Wide
conda activate qwen25vl
# 下载多个模型
echo "开始下载模型..."

# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
# MODEL_NAME="stabilityai/stable-diffusion-2-1"
MODEL_NAME="Runyi-Hu/MaskMark"
echo "下载 $MODEL_NAME 模型"
hf download "$MODEL_NAME" 



# # 下载 black-forest-labs/FLUX.1-Fill-dev 模型
# MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
# echo "下载 $MODEL_NAME 模型"
# huggingface-cli download "$MODEL_NAME" --resume-download

# # 下载 RiverZ/normal-lora 模型
# MODEL_NAME="RiverZ/normal-lora"
# echo "下载 $MODEL_NAME 模型"
# huggingface-cli download "$MODEL_NAME" --resume-download

# 如果需要，取消注释并修改以下载其他模型
# MODEL_NAME="vinesmsuic/magicbrush-jul7"
# echo "下载 $MODEL_NAME 模型"
# huggingface-cli download "$MODEL_NAME" --resume-download

# MODEL_NAME="timbrooks/instruct-pix2pix"
# echo "下载 $MODEL_NAME 模型"
# huggingface-cli download "$MODEL_NAME" --resume-download

# MODEL_NAME="timbrooks/instruct-pix2pix-distill" # 这个可能是基础模型，LoRA 分开加载
# echo "下载 $MODEL_NAME 模型"
# huggingface-cli download "$MODEL_NAME" --resume-download


# 等待所有后台下载完成 (如果并行运行，但这里是顺序的)
wait

echo "所有指定的模型下载尝试完成！"


