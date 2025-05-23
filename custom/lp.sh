#!/bin/bash
#SBATCH -p gpu3       # gpu5:A6000 48G gpu3:2080Ti 11G
#SBATCH -N 1          # 只在一个节点上运行任务
#SBATCH -c 4          # 申请 CPU 核心：1个
#SBATCH --mem 10G     # 申请内存
#SBATCH --gres gpu:1  # 分配1个GPU（纯 CPU 任务不用写）
#SBATCH -t 48:00:00   # 设置任务运行时间，格式为小时:分钟:秒
#SBATCH -o log/%j.out # 标准输出重定向到日志文件
#SBATCH -e log/%j.out # 错误输出重定向到日志文件

# 用法检查
if [ -z "$1" ]; then
  echo "Usage: sbatch $0 <log_path> [<output_image>]"
  exit 1
fi

LOG_PATH="$1"

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh   # 根据你的conda安装路径修改

echo "job begin"
# 输出当前时间（上海市区时间）
echo "Current time: $(TZ='Asia/Shanghai' date)"

conda activate Robust-Wide

python custom/log2plt.py -l "$LOG_PATH"

echo "job end"