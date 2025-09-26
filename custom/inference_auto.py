import os
import glob
import re
import argparse
# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Parse log file and plot metrics.")
parser.add_argument("-l", "--log_dir", type=str, required=True, help="Path to the log directory.")
args = parser.parse_args()

# 运行inference.py

# 1. 找到最新的checkpoint文件夹（找有step的文件夹）
step_dirs = glob.glob(os.path.join(args.log_dir, "step*"))
if step_dirs:
    # 按步骤数排序
    step_dirs.sort(key=lambda x: int(re.search(r'step(\d+)', x).group(1)), reverse=True)
    ckpt_dir = step_dirs[0]  # 取最大步骤的checkpoint
else:
    # 如果没有特定步骤的文件夹，则使用整个日志文件夹
    print("警告：未找到有效的检查点目录，跳过推理步骤")
    exit(0)

print(f"使用检查点目录: {ckpt_dir}")

# 2. 组装命令字符串（注意给路径加引号，防止含空格）
image_file = './examples/Gadot.png'
output_dir = args.log_dir
cmd = (
    f'python inference.py '
    f'--ckpt_dir "{ckpt_dir}" '
    f'--image_file "{image_file}" '
    f'--output_dir "{output_dir}"'
)

# 3. 执行
exit_code = os.system(cmd)

# 4. 判断是否成功
if exit_code != 0:
    print(f"推理过程中出现错误，退出码: {exit_code >> 8}")