#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历 train_results 下 2025-04-24* 和 2025-04-25* 目录，
提取每个目录中 log.txt 的倒数第二行（测试结果行），
解析出时间戳和测试指标，汇总到 Excel 文件中保存。
"""

import os
import ast
import pandas as pd
from pathlib import Path
from collections import deque

def collect_and_save():
    # 脚本所在目录 custom/
    script_dir = Path(__file__).parent
    # train_results 根目录
    results_dir = script_dir.parent / "train_results"
    prefixes = ["2025-04-27", "2025-04-28"]
    # prefixes = ["2025-04-18"]
    records = []

    for prefix in prefixes:
        for run_dir in sorted(results_dir.glob(f"{prefix}*")):
            if not run_dir.is_dir():
                continue
            log_file = run_dir / "log.txt"
            if not log_file.is_file():
                continue

            # —— 提取数据集 & 模型 —— 
            run_name = run_dir.name
            # 可匹配的数据集
            dataset_list = [
                "timbrooks___instructpix2pix-clip-filtered",
                "BleachNick___ultra_edit_500k",
                "osunlp___magic_brush",
                "facebook___emu_edit_test_set"
            ]
            dataset = next((d for d in dataset_list if d in run_name), "")
            # 模型名后缀映射到完整路径
            model_map = {
                "instruct-pix2pix-distill": "quickjkee/instruct-pix2pix-distill",
                "sd-turbo":                   "stabilityai/sd-turbo",
                "instruct-pix2pix":           "timbrooks/instruct-pix2pix",
                "magicbrush-jul7":            "vinesmsuic/magicbrush-jul7",
                "sd-x2-latent-upscaler":      "stabilityai/sd-x2-latent-upscaler"
            }
            model = next((full for short, full in model_map.items() if run_name.endswith(short)), "")

            # 只读取文件末尾两行
            with open(log_file, 'r', encoding='utf-8') as f:
                last_two = deque(f, maxlen=2)
            if len(last_two) < 2:
                continue
            line = last_two[0].strip()
            # 找到字典起始位置
            idx = line.find(" - {")
            if idx == -1:
                continue
            # 提取时间戳和字典字符串
            timestamp = line[:idx].strip()
            dict_str = line[idx+3:].strip()  # 去掉前面的 " - "

            try:
                data = ast.literal_eval(dict_str)
            except Exception:
                continue

            # 构造一条记录，加入 model & dataset
            record = {
                "run_dir":   run_name,
                "model":     model,
                "dataset":   dataset,
            }
            # 添加指标
            for k, v in data.items():
                record[k] = v
            records.append(record)

    if not records:
        print("未找到任何符合条件的日志记录。")
        return

    # 转成 DataFrame 并保存为 Excel
    df = pd.DataFrame(records)
    output_path = script_dir / "summary_results.xlsx"
    df.to_excel(output_path, index=False)
    print(f"已生成汇总文件：{output_path}")

if __name__ == "__main__":
    collect_and_save()  