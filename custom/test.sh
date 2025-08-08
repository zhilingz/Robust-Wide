#!/bin/bash
#SBATCH -p gpu5       # 或者你出问题的分区
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH --gres gpu:1
#SBATCH -o log/%j.out
#SBATCH -e log/%j.out
#SBATCH -t 05:00:00   # 短时间

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh   # 根据你的conda安装路径修改

echo "job begin"
# 打印 SBATCH 提交参数
scontrol show job ${SLURM_JOB_ID}
# 输出当前时间（上海市区时间）
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')"

conda activate Robust-Wide

echo "Hello from Slurm job ${SLURM_JOB_ID}"
python custom/test.py \
  --checkpoint_dir "examples/step20000" \
  --batch_size 16 \
  --image_size 512 \
  --test_size 1200 \
  --output_file "log/test.json" \
  --seed 42

echo "Test job finished."

# sbatch custom/test.sh
# 大概1小时20分钟