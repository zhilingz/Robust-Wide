#!/usr/bin/env python3
# img_diff.py
import cv2
import numpy as np
import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two images, highlight edited regions, and save result as one image."
    )
    p.add_argument("-b", "--before",  default="inference_results/timbrooks___instructpix2pix-clip-filtered/FLUX.1-Fill-dev_2.png",
                   help="Path to original image")
    p.add_argument("-a", "--after",  default="inference_results/timbrooks___instructpix2pix-clip-filtered/original_2.png",
                   help="Path to edited image")
    p.add_argument("-o", "--output", default=None,
                   help="Filename of saved result (default: auto-generated based on --after path)")
    p.add_argument("-t", "--thresh", type=int, default=15,
                   help="Pixel-value threshold for change detection (default: 20)")
    p.add_argument("-k", "--kernel", type=int, default=3,
                   help="Kernel size for morphological opening (default: 3)")
    return p

def generate_output_path(after_path):
    """根据after路径生成输出路径，在文件名后添加_diff"""
    after_path = Path(after_path)
    # 获取文件名（不含扩展名）和扩展名
    stem = after_path.stem
    suffix = after_path.suffix
    # 生成新的文件名：原文件名_diff.扩展名
    new_filename = f"{stem}_diff{suffix}"
    # 返回完整路径
    return after_path.parent / new_filename

def diff_mask(bgr_before, bgr_after, thresh=20, kernel=3):
    # 使用灰度图
    # gray_b = cv2.cvtColor(bgr_before, cv2.COLOR_BGR2GRAY)
    # gray_a = cv2.cvtColor(bgr_after,  cv2.COLOR_BGR2GRAY)
    # │差异 + 二值化│
    # diff  = cv2.absdiff(gray_a, gray_b)
    # _, m  = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    diff_b = cv2.absdiff(bgr_after[:,:,0], bgr_before[:,:,0])
    diff_g = cv2.absdiff(bgr_after[:,:,1], bgr_before[:,:,1])
    diff_r = cv2.absdiff(bgr_after[:,:,2], bgr_before[:,:,2])
    # 可以将三个通道的差异合并，例如取最大值或平均值
    color_diff_magnitude = np.maximum(np.maximum(diff_b, diff_g), diff_r)
    _, m = cv2.threshold(color_diff_magnitude, thresh, 255, cv2.THRESH_BINARY)

    # │形态学开运算│—去小杂点
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    m     = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    # │计算占比│
    ratio = m.sum() / 255 / m.size
    return m, ratio

def overlay_mask(bgr_img, mask, alpha=0.5):
    heat  = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr_img, 1, heat, alpha, 0)

def main():
    args = build_parser().parse_args()
    before_path, after_path = Path(args.before), Path(args.after)
    
    # 如果没有指定输出路径，则根据after路径自动生成
    if args.output is None:
        output_path = generate_output_path(after_path)
    else:
        output_path = Path(args.output)

    # ── 读取并尺寸检查 ───────────────────────────────────
    bgr_before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    bgr_after  = cv2.imread(str(after_path),  cv2.IMREAD_COLOR)
    if bgr_before is None or bgr_after is None:
        raise FileNotFoundError("无法读取图片，请确认路径正确。")
    if bgr_before.shape != bgr_after.shape:
        raise ValueError("两张图尺寸/通道数不一致，请先对齐。")

    # ── 计算差异 mask & 占比 ─────────────────────────────
    mask, ratio = diff_mask(bgr_before, bgr_after,
                            thresh=args.thresh, kernel=args.kernel)
    print(f"编辑区域占比: {ratio * 100:.2f}%")

    # ── 生成叠加可视化图 ────────────────────────────────
    overlay = overlay_mask(bgr_after, mask)
    concat  = cv2.hconcat([bgr_before, bgr_after, overlay])

    # ── 保存 ──────────────────────────────────────────
    cv2.imwrite(str(output_path), concat)
    print(f"已保存至: {output_path}")

if __name__ == "__main__":
    main()
