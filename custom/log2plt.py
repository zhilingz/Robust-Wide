# -*- coding: utf-8 -*- 
import re
import matplotlib.pyplot as plt
import argparse
import os
import datetime
import glob

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Parse log file and plot metrics.")
parser.add_argument("-l", "--log_file", type=str, required=True, help="Path to the log file.")
args = parser.parse_args()

steps = []
global_steps = []
dec_loss_before_edit = []
dec_loss_after_edit = []
ssim_values = []
psnr_values = []
error_rate_before_edit_values = []
error_rate_after_edit_values = []

# 测试结果变量
test_avg_psnr = None
test_avg_ssim = None
test_avg_error_rate_before_edit = None
test_avg_error_rate_after_edit = None
test_date_time = None

# 训练起止时间
start_time_str = None
end_time_str = None

# 使用命令行参数传入的日志文件路径
log_file = args.log_file

# 获取日志文件所在的目录
log_dir = os.path.dirname(log_file)

# 更新正则表达式，匹配包含所有字段的训练日志行
train_pattern_full = r"""
    (\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2})  # 日期和时间
    \s-\sINFO\s-\s__main__\s-\s
    \{
        'step':\s(\d+),\s
        'global_step':\s(\d+),\s
        'lr':\s([\d.eE+-]+),\s
        'enc_pixel_loss':\s([\d.eE+-]+),\s
        'enc_latent_loss':\s([\d.eE+-]+),\s
        'dec_loss_before_edit':\s([\d.eE+-]+),\s
        'dec_loss_after_edit':\s([\d.eE+-]+),\s
        'psnr':\s([\d.eE+-]+),\s
        'ssim':\s([\d.eE+-]+),\s
        'error_rate_before_edit':\s([\d.eE+-]+),\s
        'error_rate_after_edit':\s([\d.eE+-]+),\s
        'processed_samples':\s\d+,\s
        'filtered_samples':\s\d+,\s
        'skipped_samples':\s\d+,\s
        'filter_rate':\s[\d.eE+-]+,\s
        'skip_rate':\s[\d.eE+-]+
    \}
"""

# 简化格式（不包含processed_samples等字段）
train_pattern_simple = r"""
    (\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2})  # 日期和时间
    \s-\sINFO\s-\s__main__\s-\s
    \{
        'step':\s(\d+),\s
        'global_step':\s(\d+),\s
        'lr':\s([\d.eE+-]+),\s
        'enc_pixel_loss':\s([\d.eE+-]+),\s
        'enc_latent_loss':\s([\d.eE+-]+),\s
        'dec_loss_before_edit':\s([\d.eE+-]+),\s
        'dec_loss_after_edit':\s([\d.eE+-]+),\s
        'psnr':\s([\d.eE+-]+),\s
        'ssim':\s([\d.eE+-]+),\s
        'error_rate_before_edit':\s([\d.eE+-]+),\s
        'error_rate_after_edit':\s([\d.eE+-]+)
    \}
"""

# 更新正则表达式，用于匹配更多 BER 类型的测试结果
test_pattern = r"""
    (\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2})  # 日期和时间
    \s-\sINFO\s-\s__main__\s-\s
    \{
        'no_distortion_BER':\s*([\d.eE+-]+),\s*
        'edit_distortion_BER':\s*([\d.eE+-]+),\s*
        'common_distortions_BER':\s*([\d.eE+-]+),\s*
        'jpeg_BER':\s*([\d.eE+-]+),\s*
        'median_blur_BER':\s*([\d.eE+-]+),\s*
        'gaussian_blur_BER':\s*([\d.eE+-]+),\s*
        'gaussian_noise_BER':\s*([\d.eE+-]+),\s*
        'sharpness_BER':\s*([\d.eE+-]+),\s*
        'brightness_BER':\s*([\d.eE+-]+),\s*
        'contrast_BER':\s*([\d.eE+-]+),\s*
        'saturation_BER':\s*([\d.eE+-]+),\s*
        'hue_BER':\s*([\d.eE+-]+),\s*
        'noise_denoise_BER':\s*([\d.eE+-]+),\s*
        'random_crop_BER':\s*([\d.eE+-]+),\s*
        'random_rotation_BER':\s*([\d.eE+-]+)
    \}
"""

# 读取日志文件并解析数据
with open(log_file, "r") as f:
    for line in f:
        # 尝试匹配训练数据 - 先尝试完整格式
        match = re.search(train_pattern_full, line.strip(), re.VERBOSE)
        if not match:
            # 如果完整格式不匹配，尝试简化格式
            match = re.search(train_pattern_simple, line.strip(), re.VERBOSE)
        
        if match:
            step = int(match.group(2))
            global_step = int(match.group(3))
            before_edit = float(match.group(7))  # dec_loss_before_edit
            after_edit = float(match.group(8))   # dec_loss_after_edit
            psnr = float(match.group(9))         # psnr
            ssim = float(match.group(10))         # ssim
            error_rate_before = float(match.group(11))  # error_rate_before_edit
            error_rate_after = float(match.group(12))   # error_rate_after_edit
            
            # 将数据添加到列表
            steps.append(step)
            global_steps.append(global_step)
            dec_loss_before_edit.append(before_edit)
            dec_loss_after_edit.append(after_edit)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            error_rate_before_edit_values.append(error_rate_before)
            error_rate_after_edit_values.append(error_rate_after)
            
            # 记录训练开始和结束时间
            if start_time_str is None:
                start_time_str = match.group(1)
            end_time_str = match.group(1)
        
        # 尝试匹配测试结果（各种 BER）
        test_match = re.search(test_pattern, line.strip(), re.VERBOSE)
        if test_match:
            test_date_time = test_match.group(1)
            test_no_distortion    = float(test_match.group(2))
            test_edit_distortion  = float(test_match.group(3))
            test_common_dist      = float(test_match.group(4))
            test_jpeg             = float(test_match.group(5))
            test_median_blur      = float(test_match.group(6))
            test_gaussian_blur    = float(test_match.group(7))
            test_gaussian_noise   = float(test_match.group(8))
            test_sharpness        = float(test_match.group(9))
            test_brightness       = float(test_match.group(10))
            test_contrast         = float(test_match.group(11))
            test_saturation       = float(test_match.group(12))
            test_hue              = float(test_match.group(13))
            test_noise_denoise    = float(test_match.group(14))
            test_random_crop      = float(test_match.group(15))
            test_random_rotation  = float(test_match.group(16))

# 创建折线图
plt.figure(figsize=(12, 18))

# 子图 1: dec_loss_before_edit 和 dec_loss_after_edit
plt.subplot(3, 1, 1)
plt.plot(global_steps, dec_loss_before_edit, label="dec_loss_before_edit", color="blue")
plt.plot(global_steps, dec_loss_after_edit, label="dec_loss_after_edit", color="red")
plt.xlabel("Global Step")
plt.ylabel("Loss")
plt.title("Decoder Loss Before and After Edit vs Global Step")
plt.legend()
plt.grid(True)

# 子图 2: psnr 和 ssim
plt.subplot(3, 1, 2)
plt.plot(global_steps, psnr_values, label="PSNR", color="green")
plt.plot(global_steps, ssim_values, label="SSIM", color="orange")
plt.xlabel("Global Step")
plt.ylabel("Metric Value")
plt.title("PSNR and SSIM vs Global Step")
plt.legend()
plt.grid(True)

# 子图 3: error_rate_before_edit 和 error_rate_after_edit
plt.subplot(3, 1, 3)
plt.plot(global_steps, error_rate_before_edit_values, label="Error Rate Before Edit", color="brown")
plt.plot(global_steps, error_rate_after_edit_values, label="Error Rate After Edit", color="gray")
plt.xlabel("Global Step")
plt.ylabel("Error Rate")
plt.title("Error Rate Before and After Edit vs Global Step")
plt.legend()
plt.grid(True)

# 调整布局
plt.tight_layout()

# 如果有测试结果，在图表底部添加各场景 BER 信息
if test_date_time:
    fmt = "%m/%d/%Y %H:%M:%S"
    # 添加安全检查
    if start_time_str and end_time_str:
        st = datetime.datetime.strptime(start_time_str, fmt)
        ed = datetime.datetime.strptime(end_time_str,   fmt)
        delta = ed - st
        duration_info = f"Total duration: {delta}"
    else:
        duration_info = "Duration: N/A"
    
    test_info = f"""
    Training start:         {start_time_str or 'N/A'}
    {duration_info}
    no_distortion_BER:      {test_no_distortion:.6f}
    edit_distortion_BER:    {test_edit_distortion:.6f}
    common_distortions_BER: {test_common_dist:.6f}
    jpeg_BER:               {test_jpeg:.6f}
    median_blur_BER:        {test_median_blur:.6f}
    gaussian_blur_BER:      {test_gaussian_blur:.6f}
    gaussian_noise_BER:     {test_gaussian_noise:.6f}
    sharpness_BER:          {test_sharpness:.6f}
    brightness_BER:         {test_brightness:.6f}
    contrast_BER:           {test_contrast:.6f}
    saturation_BER:         {test_saturation:.6f}
    hue_BER:                {test_hue:.6f}
    noise_denoise_BER:      {test_noise_denoise:.6f}
    random_crop_BER:        {test_random_crop:.6f}
    random_rotation_BER:    {test_random_rotation:.6f}
    """
    plt.figtext(0.8, 0.05, test_info, ha="left", fontsize=9,
                bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

# 保存图形为文件（包含测试结果）
output_image_path = os.path.join(log_dir, "metrics_plot.png")
plt.savefig(output_image_path, bbox_inches="tight")

# 运行inference.py

# 1. 找到最新的checkpoint文件夹（找有step的文件夹）
step_dirs = glob.glob(os.path.join(log_dir, "step*"))
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
output_dir = log_dir
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