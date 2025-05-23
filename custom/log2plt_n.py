# -*- coding: utf-8 -*-
import re
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

# 在此定义你的默认日志目录（当未通过命令行指定时使用）
DEFAULT_LOG_DIRS = [
    # 5
    "train_results/2025-05-01T22-30-33_timbrooks___instructpix2pix-clip-filtered_instruct-pix2pix",
    # 20
    "train_results/2025-04-30T22-39-41_timbrooks___instructpix2pix-clip-filtered_instruct-pix2pix",
    # 100
    "train_results/2025-04-29T21-47-57_timbrooks___instructpix2pix-clip-filtered_instruct-pix2pix",
    # 500
    "train_results/2025-04-28T21-34-34_timbrooks___instructpix2pix-clip-filtered_instruct-pix2pix",
    # 2000
    "train_results/2025-04-27T21-49-56_timbrooks___instructpix2pix-clip-filtered_instruct-pix2pix",
    # 5000
    "train_results/2025-04-26T20-53-28_timbrooks___instructpix2pix-clip-filtered_instruct-pix2pix",
    # 10000
    "train_results/2025-04-26T18-13-41_timbrooks___instructpix2pix-clip-filtered_instruct-pix2pix",
    # 15000
    "train_results/2025-04-26T20-17-48_timbrooks___instructpix2pix-clip-filtered_instruct-pix2pix",
    # 20000
    "train_results/2025-04-27T21-56-22_timbrooks___instructpix2pix-clip-filtered_instruct-pix2pix",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="解析日志并在一张图中比较 BER 曲线（可通过命令行或代码中定义）"
    )
    parser.add_argument(
        "-d", "--log_dirs",
        type=str,
        nargs="+",
        help="日志所在文件夹路径列表，会递归搜索其中的 log.txt；若不指定，则使用脚本中 DEFAULT_LOG_DIRS"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="inference_results/ber_comparison.png",
        help="输出图片文件名，默认为 ber_comparison.png"
    )
    return parser.parse_args()

def extract_ber_from_log(log_file):
    """
    从单个日志文件中提取 (step, ber) 列表，
    ber 即 'error_rate_after_edit' 字段。
    """
    pattern = re.compile(
        r"'step':\s*(\d+).*?'error_rate_after_edit':\s*([\d.eE+-]+)",
        re.DOTALL
    )
    steps, bers = [], []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                bers.append(float(m.group(2)))
    return steps, bers

def extract_train_size(log_file):
    """
    返回 log_file 中 vars(args)['train_size'] 的值（字符串），若未找到返回 None
    """
    pattern = re.compile(r'"train_size":\s*(\d+)', re.IGNORECASE)
    with open(log_file, "r", encoding="utf-8") as f:
        # 仅读取前 100 行，避免全文搜索
        for _ in range(100):
            line = f.readline()
            if not line:
                break
            m = pattern.search(line)
            if m:
                return m.group(1)
    return None

def extract_first_step_below_threshold(log_file, threshold=0.01):
    """
    返回 error_rate_before_edit 首次低于 threshold 的 step，未找到则返回 None
    """
    pattern = re.compile(
        r"'step':\s*(\d+).*?'error_rate_before_edit':\s*([\d.eE+-]+)",
        re.DOTALL
    )
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                ber_before = float(m.group(2))
                if ber_before < threshold:
                    return step
    return None

def main():
    args = parse_args()

    if args.log_dirs:
        log_dirs = args.log_dirs
    else:
        log_dirs = DEFAULT_LOG_DIRS

    if not log_dirs:
        print("错误：未指定任何日志目录，请通过 --log_dirs 或在脚本中设置 DEFAULT_LOG_DIRS")
        return

    log_paths = []
    for log_dir in log_dirs:
        for root, _, files in os.walk(log_dir):
            if "log.txt" in files:
                log_paths.append(os.path.join(root, "log.txt"))

    if not log_paths:
        print("未在指定目录下找到任何 log.txt 文件")
        return

    labels = []
    steps_below = []

    for log_path in log_paths:
        step = extract_first_step_below_threshold(log_path, threshold=0.001)
        train_size = extract_train_size(log_path)
        label = f"train_size={train_size}" if train_size else os.path.basename(os.path.dirname(log_path))
        labels.append(label)
        steps_below.append(step if step is not None else -1)  # -1 表示未达到

    plt.figure(figsize=(10, 6))
    # 可选：将未达到的用不同颜色或 hatch 表示
    colors = ['C0' if s != -1 else 'C1' for s in steps_below]
    plt.bar(labels, [s if s != -1 else 0 for s in steps_below], color=colors)
    for i, s in enumerate(steps_below):
        if s == -1:
            plt.text(i, 0, "未达到", ha='center', va='bottom', color='red', fontsize=10)
        else:
            plt.text(i, s, str(s), ha='center', va='bottom', fontsize=10)

    plt.xlabel("train size")
    plt.ylabel("first step ber before edit < 0.001")
    plt.title("first step ber before edit < 0.001 for different train sizes")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"已将统计图保存至：{args.output}")

if __name__ == "__main__":
    main()