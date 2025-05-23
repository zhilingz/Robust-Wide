# Script Name: inference.sh
# Description: generate the watermarked image
# Author: Runyi Hu
# Date: 2024-10-15
  # --ckpt_dir './checkpoints' \
  
python inference.py \
  --ckpt_dir './train_results/2025-03-04T06-09-18/step7000/' \
  --image_file './examples/Gadot.png' \
  --output_dir './inference_results'
