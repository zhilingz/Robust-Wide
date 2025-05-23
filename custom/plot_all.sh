#!/usr/bin/env bash
# plot_all.sh 支持传入日期参数，遍历 train_results 中所有指定日期的文件夹，调用 log2plt.py

PYTHON=python
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG2PLT="$SCRIPT_DIR/log2plt.py"
RESULTS_DIR="$SCRIPT_DIR/../train_results"

# 支持传入日期参数，多个日期用空格分隔
if [[ $# -eq 0 ]]; then
  # 没有参数，处理所有 2025-04-* 文件夹
  DIR_PATTERN="$RESULTS_DIR/2025-04-*"
else
  # 有参数，拼接所有日期
  DIR_PATTERN=""
  for date in "$@"; do
    DIR_PATTERN="$DIR_PATTERN $RESULTS_DIR/$date*"
  done
fi

for dir in $DIR_PATTERN; do
  if [[ -d "$dir" ]]; then
    log_file="$dir/log.txt"
    if [[ -f "$log_file" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] 处理日志：$log_file"
      $PYTHON "$LOG2PLT" -l "$log_file"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] 未找到日志文件：$log_file"
    fi
  fi
done