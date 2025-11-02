import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def create_curve(start_x, start_y, end_x, end_y):
    """
    通过起点、终点和间隔列表创建曲线
    
    参数:
        start_x: 起点的x坐标
        start_y: 起点的y坐标
        end_x: 终点的x坐标
        end_y: 终点的y坐标
        intervals: 取点间隔列表，表示从起点到终点在对数空间中的相对位置
                  例如 [0, 0.1, 0.2, 0.4, 0.6, 0.75, 0.85, 0.93, 1.0]
                  其中 0 表示起点，1.0 表示终点
    
    返回:
        x, y: 对应的x轴和y轴数据（幂律插值）
    """
    intervals = [0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t = np.array(intervals)
    
    # 在对数空间中插值
    log_start = np.log10(start_x)
    log_end = np.log10(end_x)
    log_x = log_start + t * (log_end - log_start)
    x = 10 ** log_x
    
    A = (start_y - end_y) * start_x * end_x / (end_x - start_x)
    B = (end_y * end_x - start_y * start_x) / (end_x - start_x)
    
    # 计算曲线上各点的 y 值
    y = A / x + B
    
    return x, y


# 创建图形
fig, ax = plt.subplots(figsize=(7, 5))

# 为每条线生成示例数据（递减趋势）
# 使用 create_curve(起点x, 起点y, 终点x, 终点y, 点数) 来定义曲线
# 每条曲线只在自己的起点和终点之间绘制

# S组（小尺寸）
s8_x, s8_y = create_curve(start_x=2e7, start_y=210, end_x=2e8, end_y=150)
s4_x, s4_y = create_curve(start_x=7e7, start_y=157, end_x=7e8, end_y=95)
s2_x, s2_y = create_curve(start_x=3e8, start_y=127, end_x=5e9, end_y=65)

# B组（基础尺寸）
b8_x, b8_y = create_curve(start_x=7e7, start_y=173, end_x=9e8, end_y=112)
b4_x, b4_y = create_curve(start_x=2.8e8, start_y=128, end_x=5e9, end_y=65)
b2_x, b2_y = create_curve(start_x=9e8, start_y=105, end_x=8e10, end_y=24)

# L组（大尺寸）
l8_x, l8_y = create_curve(start_x=2e8, start_y=160, end_x=2e9, end_y=117)
l4_x, l4_y = create_curve(start_x=8e8, start_y=112, end_x=5e10, end_y=24)
l2_x, l2_y = create_curve(start_x=4e9, start_y=82, end_x=1.1e11, end_y=12)

# XL组（超大尺寸）
xl8_x, xl8_y = create_curve(start_x=4e8, start_y=163, end_x=3e9, end_y=110)
xl4_x, xl4_y = create_curve(start_x=1.1e9, start_y=112, end_x=1.1e11, end_y=18)
xl2_x, xl2_y = create_curve(start_x=5e9, start_y=78, end_x=7e11, end_y=12)

# 绘制曲线 - 使用实心圆点标记
# 可以通过修改 linewidth 和 markersize 来调整线条粗细和点的大小
# linewidth: 线条粗细 (默认2, 可改为 1-5)
# markersize: 数据点大小 (默认6, 可改为 3-12)

ax.plot(s8_x, s8_y, '-o', color='#fba09c', linewidth=0.48, markersize=2.4, label='S/8')
ax.plot(s4_x, s4_y, '-o', color='#f86761', linewidth=0.72, markersize=3.6, label='S/4')
ax.plot(s2_x, s2_y, '-o', color='#f5433c', linewidth=0.96, markersize=4.8, label='S/2')

ax.plot(b8_x, b8_y, '-o', color='#f9cb8a', linewidth=0.72, markersize=3.2, label='B/8')
ax.plot(b4_x, b4_y, '-o', color='#f7ba64', linewidth=0.96, markersize=4.4, label='B/4')
ax.plot(b2_x, b2_y, '-o', color='#f4a83d', linewidth=1.2, markersize=6.0, label='B/2')

ax.plot(l8_x, l8_y, '-o', color='#b3cda2', linewidth=0.96, markersize=4.4, label='L/8')
ax.plot(l4_x, l4_y, '-o', color='#9fbf89', linewidth=1.2, markersize=5.6, label='L/4')
ax.plot(l2_x, l2_y, '-o', color='#88b06d', linewidth=1.44, markersize=7.2, label='L/2')

ax.plot(xl8_x, xl8_y, '-o', color='#adc7ff', linewidth=1.2, markersize=5.6, label='XL/8')
ax.plot(xl4_x, xl4_y, '-o', color='#86aafe', linewidth=1.44, markersize=6.8, label='XL/4')
ax.plot(xl2_x, xl2_y, '-o', color='#5d8dfe', linewidth=1.68, markersize=8.0, label='XL/2')

# 设置对数刻度
ax.set_xscale('log')

# 设置坐标轴范围
ax.set_xlim(0.8e7, 1.1e12)
ax.set_ylim(0, 220)

# 设置纵坐标刻度：每25一个刻度
ax.yaxis.set_major_locator(plt.MultipleLocator(25))

# 移除横纵坐标小刻度
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.yaxis.set_minor_locator(plt.NullLocator())

# 移除刻度线但保留刻度标签
ax.tick_params(axis='both', which='both', length=0)

# 设置网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 设置标签
ax.set_xlabel('Training Compute (Gflops)', fontsize=16, fontweight='normal')
ax.set_ylabel('FID-50K', fontsize=16, fontweight='normal')

# 设置刻度标签大小和与坐标轴的距离
# pad 参数控制刻度数字与坐标轴的距离（单位：点）
ax.tick_params(axis='x', which='major', labelsize=16, pad=8)  # 横坐标间距
ax.tick_params(axis='y', which='major', labelsize=16)  # 纵坐标使用默认间距

# 添加图例（放在右上角）
ax.legend(loc='upper right', ncol=4, fontsize=10, framealpha=0.9)

# ========== 添加放大区域 ==========
# 创建嵌入式子图
axins = inset_axes(ax, width="35%", height="50%", loc='upper right',
                   bbox_to_anchor=(0, 0, 1, 0.7), bbox_transform=ax.transAxes)

# 定义放大区域的范围
x1, x2 = 2e10, 2e11   # x轴范围
y1, y2 = 10, 30    # y轴范围
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xscale('log')

# 在放大区域重新绘制曲线（只绘制需要的曲线）
axins.plot(b2_x, b2_y, '-o', color='#f4a83d', linewidth=1.2, markersize=6.0, label='B/2')
axins.plot(l2_x, l2_y, '-o', color='#88b06d', linewidth=1.44, markersize=7.2, label='L/2')
axins.plot(xl4_x, xl4_y, '-o', color='#86aafe', linewidth=1.44, markersize=6.8, label='XL/4')
axins.plot(xl2_x, xl2_y, '-o', color='#5d8dfe', linewidth=1.68, markersize=8.0, label='XL/2')


# 设置放大区域的网格和刻度（关闭网格）
axins.grid(False)
axins.tick_params(axis='both', which='both', length=0, labelsize=9)
axins.xaxis.set_minor_locator(plt.NullLocator())
axins.yaxis.set_minor_locator(plt.NullLocator())

# 去掉子图的横坐标刻度标签
axins.set_xticklabels([])

# 添加连接线，指示放大的区域
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", linestyle='-', linewidth=1)


plt.savefig('dit_fig9.png', dpi=200, bbox_inches='tight')