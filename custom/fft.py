import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------- 1. 读彩色图，FFT 用灰度 ----------
img_path = "/public/zhangzhiling/code/Robust-Wide/inference_results/Gadot_orig.png"
img_color = Image.open(img_path).convert("RGB")
img_gray  = img_color.convert("L")
img_np    = np.asarray(img_gray, dtype=float)

# ---------- 2. 频域 ----------
f        = np.fft.fft2(img_np)
fshift   = np.fft.fftshift(f)
magnitude = 20 * np.log(np.abs(fshift) + 1)
phase     = np.angle(fshift)

# ---------- 3. 可视化 ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Spatial & Frequency Domain (colored)", fontsize=18)

# 3-1 空间域（原彩色）
ax[0].imshow(img_color)
ax[0].set_title("Spatial domain (RGB)")

# 3-2 幅度谱 —— viridis 伪彩
im1 = ax[1].imshow(magnitude, cmap="viridis")
ax[1].set_title("Magnitude spectrum\n(viridis, log scale)")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

# 3-3 相位谱 —— hsv 更直观
im2 = ax[2].imshow(phase, cmap="twilight_shifted")   # 或 cmap="hsv"
ax[2].set_title("Phase spectrum\n(twilight_shifted)")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

for a in ax: 
    a.axis("off")

fig.subplots_adjust(top=0.86, wspace=0.05)

# ---------- 4. 保存 ----------
out_path = "/public/zhangzhiling/code/Robust-Wide/inference_results/Gadot_fft_panel_color.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", out_path)
