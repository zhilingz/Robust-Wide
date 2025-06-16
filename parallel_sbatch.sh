#!/bin/bash
# 文件名: parallel_sbatch.sh
# 描述: 并行处理所有数据集和模型组合，一次性提交所有任务

# 创建日志目录和临时文件目录
mkdir -p log
mkdir -p temp

# 定义数据集和模型数组（使用完整路径）
declare -a DATASETS=(
    "/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f"
    "/public/zhangzhiling/datasets/BleachNick___ultra_edit_500k/default/0.0.0/8d78dc552b576027618ff2170c4c1d7bcaf27ad2"
    "/public/zhangzhiling/datasets/osunlp___magic_brush/default/0.0.0/1d8d4629150d18ca50afab66391866f2085be989"
    "/public/zhangzhiling/datasets/facebook___emu_edit_test_set/default/0.0.0/b31936a0b6c267e87d373014034cd8fb44ced2fb"
)
declare -a MODELS=(
    # "/public/zhangzhiling/models/timbrooks/instruct-pix2pix"
    "/public/zhangzhiling/models/vinesmsuic/magicbrush-jul7"
    # "/public/zhangzhiling/models/timbrooks/instruct-pix2pix-distill"
    # "/public/zhangzhiling/models/stabilityai/sd-turbo"
    # "/public/zhangzhiling/models/stabilityai/sd-x2-latent-upscaler"
)

# 计算总共需要运行的任务数
TOTAL_JOBS=$((${#DATASETS[@]} * ${#MODELS[@]}))
echo "总共需要运行 $TOTAL_JOBS 个组合任务"

# 跟踪已提交的作业ID
declare -a JOB_IDS=()

# 遍历所有组合
for ((DATA_ID=0; DATA_ID<${#DATASETS[@]}; DATA_ID++)); do
    for ((MODEL_ID=0; MODEL_ID<${#MODELS[@]}; MODEL_ID++)); do
        # 获取当前数据集和模型的路径
        DATA_PATH=${DATASETS[$DATA_ID]}
        MODEL_PATH=${MODELS[$MODEL_ID]}

        # 提取简短名称用于文件命名
        DATA_NAME=$(echo $DATA_PATH | awk -F/ '{print $(NF-3)}')
        MODEL_NAME=$(echo $MODEL_PATH | awk -F/ '{print $NF}')

        # 创建临时训练脚本（放在temp文件夹中）
        TEMP_TRAIN_SCRIPT="temp/train_${DATA_ID}_${MODEL_ID}.sh"
        cp train.sh $TEMP_TRAIN_SCRIPT

        # 修改训练脚本，直接指定DATA_DIR和MODEL_DIR而不是使用索引
        sed -i "/DATA_ID=/,/MODEL_DIR=\${MODELS\[\$MODEL_ID\]}/c\\
DATA_DIR=\"${DATA_PATH}\"\\
MODEL_DIR=\"${MODEL_PATH}\"" $TEMP_TRAIN_SCRIPT

        # 创建临时sbatch脚本（放在temp文件夹中）
        TEMP_SBATCH_SCRIPT="temp/sbatch_${DATA_ID}_${MODEL_ID}.sh"
        cp sbatch.sh $TEMP_SBATCH_SCRIPT

        # 修改sbatch脚本中的日志输出路径和训练脚本,#为分隔符
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        sed -i "s#log/%j.out#log/${TIMESTAMP}_${DATA_NAME}_${MODEL_NAME}_%j.out#" $TEMP_SBATCH_SCRIPT
        sed -i "s#bash train.sh#bash ${TEMP_TRAIN_SCRIPT}#" $TEMP_SBATCH_SCRIPT

        # 提交作业
        echo "$(date '+%Y-%m-%d %H:%M:%S') 提交作业: 数据集=$DATA_NAME, 模型=$MODEL_NAME"
        JOB_ID=$(sbatch --parsable "$TEMP_SBATCH_SCRIPT")
        echo "$(date '+%Y-%m-%d %H:%M:%S') 作业ID: $JOB_ID"

        # 添加到作业ID数组
        JOB_IDS+=($JOB_ID)
    done
done

echo "所有作业已提交，等待完成..."

# 等待所有作业完成
for JOB_ID in "${JOB_IDS[@]}"; do
    scontrol wait jobid=$JOB_ID
done

echo "所有作业已完成!"

# 清理临时脚本文件（可选）
echo "清理临时文件..."
echo "不清理临时文件"
# rm -rf temp  # 取消注释此行可以完全删除temp文件夹
# 或者保留文件夹但删除文件
# rm -f temp/sbatch_*_*.sh temp/train_*_*.sh

echo "处理完成!"