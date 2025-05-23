#!/bin/bash
echo "开始下载模型和数据集..."
bash utils/download_model.sh > utils/model_download.log 2>&1 &
python utils/download_dataset.py > utils/dataset_download.log 2>&1 &
wait
echo "所有下载完成！"
