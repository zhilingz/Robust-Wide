#!/bin/bash
#SBATCH -p gpu5       # gpu5:A6000 48G gpu3:2080Ti 11G
#SBATCH -N 1          # 只在一个节点上运行任务
#SBATCH -c 4          # 申请 CPU 核心：4个
#SBATCH --mem 10G     # 申请内存
#SBATCH --gres gpu:1  # 分配1个GPU
#SBATCH -t 72:00:00   # 设置任务运行时间，格式为小时:分钟:秒
#SBATCH -o log/%j.out # 标准输出重定向到日志文件
#SBATCH -e log/%j.out # 错误输出重定向到日志文件

# Script Name: sbatch.sh (合并版)
# Description: SLURM job submission and training launch script
# Author: Runyi Hu
# Date: 2024-10-15

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh   # 根据你的conda安装路径修改

echo "job begin"
# 打印 SBATCH 提交参数
scontrol show job ${SLURM_JOB_ID}
# 输出当前时间（上海市区时间）
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')"

conda activate Robust-Wide

# 显示GPU信息
nvidia-smi --query-gpu=gpu_name --format=csv,noheader

# 数据集和模型配置
DATA_ID=0  # 通过修改这个数字来选择数据集
MODEL_ID=1  # 通过修改这个数字来选择模型
declare -a DATASETS=(
    "/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f"
    "/public/zhangzhiling/datasets/BleachNick___ultra_edit_500k/default/0.0.0/8d78dc552b576027618ff2170c4c1d7bcaf27ad2"
    "/public/zhangzhiling/datasets/osunlp___magic_brush/default/0.0.0/1d8d4629150d18ca50afab66391866f2085be989"
    "/public/zhangzhiling/datasets/facebook___emu_edit_test_set/default/0.0.0/b31936a0b6c267e87d373014034cd8fb44ced2fb"
    )
declare -a MODELS=(
    "/public/zhangzhiling/models/timbrooks/instruct-pix2pix"
    "/public/zhangzhiling/models/vinesmsuic/magicbrush-jul7"
    "/public/zhangzhiling/models/timbrooks/instruct-pix2pix-distill"
    "/public/zhangzhiling/models/stabilityai/sd-turbo"
    "/public/zhangzhiling/models/stabilityai/sd-x2-latent-upscaler"
    "black-forest-labs/FLUX.1-Fill-dev"
)


DATA_DIR="filtered_datasets/max20000_id_psnr12.5/facebook___emu_edit_test_set/magicbrush-jul7"
# DATA_DIR="filtered_datasets/max1000_id/timbrooks___instructpix2pix-clip-filtered/instruct-pix2pix"
# DATA_DIR=${DATASETS[$DATA_ID]}
MODEL_DIR=${MODELS[$MODEL_ID]}
echo "数据集路径: $DATA_DIR"
echo "模型路径: $MODEL_DIR"

# 根据 MODEL_DIR 设置 batch_size
if [ "$MODEL_DIR" = "/public/zhangzhiling/models/stabilityai/sd-x2-latent-upscaler" ]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=2
fi

export HF_ENDPOINT="https://hf-mirror.com"

# 启动训练
accelerate launch --config_file ./config/accelerate_config.yaml train.py \
  --train_data_dir $DATA_DIR \
  --model_dir $MODEL_DIR \
  --wm_model_config "./config/model_config.yaml" \
  --output_dir "./train_results" \
  --image_size 512 \
  --train_size 20000 \
  --test_size 1200 \
  --batch_size $BATCH_SIZE \
  --max_train_steps 20000 \
  --seed 42 \
  --learning_rate 1e-3 \
  --lr_scheduler "cosine" \
  --lr_warmup_steps 400 \
  --log_steps 20 \
  --save_steps 2000 \
  --last_grad_steps 3 \
  --decoder_weight 0.1 \
  --enc_latent_weight 0.005 \
  --gradient_accumulation_steps 1 \
  --enable_output_images \
  --enable_offline_filter \

echo "job end"

# sbatch --dependency=afterok:<jobid> sbatch.sh 在某一任务成功后运行