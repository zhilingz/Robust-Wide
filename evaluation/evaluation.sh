#!/bin/bash

# 全面水印测评脚本
# 使用方法: bash evaluation.sh <checkpoint_dir>

# 默认参数
TEST_SIZE=1000
WM_MODEL_NAME="WAM" # Robust-Wide  TrustMark VINE MaskMark WAM
# CKPT_DIR=""
# CKPT_DIR="/public/zhangzhiling/code/Robust-Wide/train_results/old/2025-03-05T19-50-18_timbrooks___instructpix2pix-clip-filtered"
# OUTPUT_DIR="${CKPT_DIR}/evaluation"
OUTPUT_DIR="/public/zhangzhiling/code/Robust-Wide/evaluation/results/${WM_MODEL_NAME}"
# EVAL_IMG_DIR="/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f"

# 根据模型名称选择conda环境
if [ "${WM_MODEL_NAME}" == "Robust-Wide" ]; then
    CONDA_ENV="Robust-Wide"
elif [ "${WM_MODEL_NAME}" == "TrustMark" ]; then
    CONDA_ENV="Robust-Wide"
elif [ "${WM_MODEL_NAME}" == "VINE" ]; then
    CONDA_ENV="VINE"
elif [ "${WM_MODEL_NAME}" == "MaskMark" ]; then
    CONDA_ENV="Robust-Wide"
elif [ "${WM_MODEL_NAME}" == "EditGuard" ]; then
    CONDA_ENV="Robust-Wide"
elif [ "${WM_MODEL_NAME}" == "WAM" ]; then
    CONDA_ENV="watermark_anything"
else
    echo "错误: 未知的模型名称 '${WM_MODEL_NAME}'"
    echo "支持的模型: Robust-Wide, TrustMark, VINE, MaskMark, EditGuard, WAM"
    exit 1
fi

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# 运行测评
python evaluation.py \
    --wm_model_name ${WM_MODEL_NAME} \
    --output_dir "${OUTPUT_DIR}" \
    --test_size ${TEST_SIZE}
    # --use_coco 

echo ""
echo "测评完成！结果保存在: ${OUTPUT_DIR}"

# salloc -p gpu4 -N 1 -c 4 --mem 10G --gres gpu:1
# srun --pty bash
# cd evaluation
# ./evaluation.sh