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

DATA_DIR=${DATASETS[$DATA_ID]}
MODEL_DIR=${MODELS[$MODEL_ID]}
OUTPUT_DIR="./filtered_datasets/max10000_id/"

python -m custom.filter --filter_num 10000 \
    --data_dir $DATA_DIR \
    --model_dir $MODEL_DIR\
    --output_dir $OUTPUT_DIR

echo "job end"

# sbatch custom/filter.sh 运行脚本