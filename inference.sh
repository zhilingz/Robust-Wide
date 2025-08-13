#!/bin/bash
#SBATCH -p gpu3       # gpu5:A6000 48G gpu3:2080Ti 11G
#SBATCH -N 1          # 只在一个节点上运行任务
#SBATCH -c 4          # 申请 CPU 核心：4个
#SBATCH --mem 10G     # 申请内存
#SBATCH --gres gpu:1  # 分配1个GPU
#SBATCH -t 00:20:00   # 设置任务运行时间，格式为小时:分钟:秒
#SBATCH -o log/%j.out # 标准输出重定向到日志文件
#SBATCH -e log/%j.out # 错误输出重定向到日志文件

# Script Name: inference.sh
# Description: generate the watermarked image
# Author: Runyi Hu
# Date: 2024-10-15
  # --ckpt_dir './checkpoints' \

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh   # 根据你的conda安装路径修改

echo "job begin"
# 打印 SBATCH 提交参数
scontrol show job ${SLURM_JOB_ID}
# 输出当前时间（上海市区时间）
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')"

conda activate Robust-Wide

python inference.py \
  --ckpt_dir './train_results/2025-08-09T10-44-07_BleachNick___ultra_edit_500k_magicbrush-jul7_257556' \
  --image_file './examples/Gadot.png' \
  --output_dir './evaluate/257647'

echo "job end"  
# sbatch inference.sh
# tail -f log/