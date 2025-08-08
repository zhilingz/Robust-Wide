#!/bin/bash
#SBATCH -p gpu5       # gpu5:A6000 48G gpu3:2080Ti 11G
#SBATCH -N 1          # 只在一个节点上运行任务
#SBATCH -c 4          # 申请 CPU 核心：4个
#SBATCH --mem 10G     # 申请内存
#SBATCH --gres gpu:1  # 分配1个GPU
#SBATCH -t 240:00:00   # 设置任务运行时间，格式为小时:分钟:秒
#SBATCH -o log/%j.out # 标准输出重定向到日志文件
#SBATCH -e log/%j.out # 错误输出重定向到日志文件

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh   # 根据你的conda安装路径修改
echo "job begin"
# 打印 SBATCH 提交参数
scontrol show job ${SLURM_JOB_ID}
# 输出当前时间（上海市区时间）
echo "Current time: $(TZ='Asia/Shanghai' date +"%Y-%m-%d %H:%M:%S %Z")"

conda activate Robust-Wide

# 显示GPU信息
nvidia-smi --query-gpu=gpu_name --format=csv,noheader

# 数据集和模型配置
DATA_ID=0  # 通过修改这个数字来选择数据集
MODEL_ID=0  # 通过修改这个数字来选择模型
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

DATA_DIR=${DATASETS[$DATA_ID]}
MODEL_DIR=${MODELS[$MODEL_ID]}
OUTPUT_DIR="./filtered_datasets/"

python -m custom.filter --filter_num 1000 \
    --data_dir $DATA_DIR \
    --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --reverse_filter True \
    --enable_psnr True \
    --psnr_min 17.0 \
    --psnr_max max \
    --enable_ssim False \
    --ssim_min 0.80 \
    --ssim_max max \
    --enable_l1 False \
    --l1_min min \
    --l1_max 0.15 \
    --enable_l2 False \
    --l2_min min \
    --l2_max 0.03 \
    --enable_edit_ratio False \
    --edit_ratio_min min \
    --edit_ratio_max 0.50

# 示例：禁用某些指标
# --enable_psnr False \
# --enable_ssim False \

# 示例：设置自定义范围
# --psnr_min 10.0 \
# --psnr_max 40.0 \
# --ssim_min 0.7 \
# --ssim_max 0.95 \
# --l1_min 0.05 \
# --l1_max 0.20 \
# --l2_min 0.01 \
# --l2_max 0.05 \
# --edit_ratio_min 0.1 \
# --edit_ratio_max 0.8

echo "job end"

# sbatch custom/filter.sh
# tail -f log/