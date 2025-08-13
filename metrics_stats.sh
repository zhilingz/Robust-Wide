#!/bin/bash
#SBATCH -p gpu5       # gpu5:A6000 48G gpu3:2080Ti 11G
#SBATCH -N 1          # 只在一个节点上运行任务
#SBATCH -c 4          # 申请 CPU 核心：4个
#SBATCH --mem 10G     # 申请内存
#SBATCH --gres gpu:1  # 分配1个GPU
#SBATCH -t 72:00:00   # 设置任务运行时间，格式为小时:分钟:秒
#SBATCH -o log/%j.out # 标准输出重定向到日志文件
#SBATCH -e log/%j.out # 错误输出重定向到日志文件

# 训练集编辑后指标统计脚本运行示例
# 使用方法: sbatch run_metrics_stats.sh

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh   # 根据你的conda安装路径修改

echo "job begin"
# 打印 SBATCH 提交参数
scontrol show job ${SLURM_JOB_ID}
# 输出当前时间（上海市区时间）
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')"

conda activate Robust-Wide



# 设置基本路径
TRAIN_DATA_DIR="/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f"
MODEL_DIR="/public/zhangzhiling/models/timbrooks/instruct-pix2pix"
OUTPUT_DIR="./metrics_output/timbrooks___instructpix2pix-clip-filtered"

# 设置参数
IMAGE_SIZE=512
TRAIN_SIZE=1000
BATCH_SIZE=4
SAVE_SAMPLES=20  # 保存样例图像数量
SEED=42

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "开始运行训练集编辑后指标统计..."
echo "输出目录: $OUTPUT_DIR"

# 运行指标统计脚本
python metrics_stats.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --image_size $IMAGE_SIZE \
    --train_size $TRAIN_SIZE \
    --batch_size $BATCH_SIZE \
    --save_samples $SAVE_SAMPLES \
    --seed $SEED \
    --use_fp16

echo "统计完成! 结果保存在: $OUTPUT_DIR"
echo "查看结果:"
echo "  - 统计信息: $OUTPUT_DIR/metrics_statistics.json"
echo "  - 原始数据: $OUTPUT_DIR/raw_metrics_data.json"
echo "  - 分布图: $OUTPUT_DIR/metrics_distribution.png"
echo "  - 样例图像: $OUTPUT_DIR/sample_*/" 