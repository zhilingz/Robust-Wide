# Script Name: train.sh
# Description: launch the training process
# Author: Runyi Hu
# Date: 2024-10-15
nvidia-smi --query-gpu=gpu_name --format=csv,noheader

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
echo "数据集路径: $DATA_DIR"
echo "模型路径: $MODEL_DIR"

# 根据 MODEL_DIR 设置 batch_size
if [ "$MODEL_DIR" = "/public/zhangzhiling/models/stabilityai/sd-x2-latent-upscaler" ]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=2
fi

export HF_ENDPOINT="https://hf-mirror.com"


# accelerate config

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
  --enc_latent_weight 0.01 \
  --gradient_accumulation_steps 1 \
  --filter_threshold 0.3 \
  --enable_realtime_filter \