#!/bin/bash
#SBATCH -p gpu3       # gpu5:A6000 48G gpu3:2080Ti 11G
#SBATCH -N 1          # 只在一个节点上运行任务
#SBATCH -c 4          # 申请 CPU 核心：1个
#SBATCH --mem 10G     # 申请内存
#SBATCH --gres gpu:1  # 分配1个GPU（纯 CPU 任务不用写）
#SBATCH -t 1:00:00   # 设置任务运行时间，格式为小时:分钟:秒
#SBATCH -o log/%j.out # 标准输出重定向到日志文件
#SBATCH -e log/%j.out # 错误输出重定向到日志文件

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh   # 根据你的conda安装路径修改

echo "job begin"
# 打印 SBATCH 提交参数
scontrol show job ${SLURM_JOB_ID}
# 输出当前时间（上海市区时间）
echo "Current time: $(TZ='Asia/Shanghai' date)"

conda activate Robust-Wide

exp_dir='train_results/2025-09-23T17-20-55_BleachNick___ultra_edit_500k_instruct-pix2pix-vae_261082'
# watermark_strength=0.55

python evaluate.py \
  --ckpt_dir "${exp_dir}" \
  --edit_strength small 

# python evaluate.py \
#   --ckpt_dir "${exp_dir}" \
#   --eval_img_dir '/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f' \
#   --output_dir "${exp_dir}/evaluate/large/watermark_strength_${watermark_strength}" \
#   --edit_strength large \
#   --watermark_strength ${watermark_strength}

echo "job end"

# sbatch evaluate.sh
# salloc -p gpu3 -N 1 -c 4 --mem 10G --gres gpu:1
# srun --pty bash
# conda activate Robust-Wide
# conda activate qwen25vl