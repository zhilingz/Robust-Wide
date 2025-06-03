#!/bin/bash

# ==================== 配置区域 ====================
# 如果不想使用命令行参数，可以直接在这里设置参数
# 设置为空字符串 "" 表示使用命令行参数或默认值

# 直接设置参数（优先级最高）
PRESET_DATE="2025-04-27"                      # 例如: "2025-05-24"
PRESET_BASE_DIR="train_results"               # 例如: "train_results" 
PRESET_SCRIPT_PATH="custom/fft.py"            # 例如: "custom/fft.py"

# ==================== 脚本开始 ====================

# 默认参数
DATE="$PRESET_DATE"
BASE_DIR="${PRESET_BASE_DIR:-train_results}"
SCRIPT_PATH="${PRESET_SCRIPT_PATH:-custom/fft.py}"

# 帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --date DATE        指定日期 (格式: YYYY-MM-DD, 例如: 2025-05-24)"
    echo "  -b, --base-dir DIR     基础目录 (默认: train_results)"
    echo "  -s, --script PATH      fft.py脚本路径 (默认: custom/fft.py)"
    echo "  -h, --help             显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -d 2025-05-24"
    echo "  $0 --date 2025-05-24 --base-dir /path/to/results"
    echo ""
    echo "也可以直接在脚本顶部的配置区域设置参数，然后直接运行:"
    echo "  $0"
}

# 只有在没有预设参数时才解析命令行参数
if [[ -z "$PRESET_DATE" ]]; then
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--date)
                DATE="$2"
                shift 2
                ;;
            -b|--base-dir)
                BASE_DIR="$2"
                shift 2
                ;;
            -s|--script)
                SCRIPT_PATH="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
else
    echo "使用脚本内预设参数..."
fi

# 检查必需参数
if [[ -z "$DATE" ]]; then
    echo "错误: 必须指定日期参数"
    echo "方法1: 使用命令行参数: $0 -d 2025-05-24"
    echo "方法2: 在脚本顶部配置区域设置 PRESET_DATE"
    show_help
    exit 1
fi

# 检查日期格式
if [[ ! "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "错误: 日期格式不正确，应为 YYYY-MM-DD"
    exit 1
fi

# 检查基础目录是否存在
if [[ ! -d "$BASE_DIR" ]]; then
    echo "错误: 基础目录不存在: $BASE_DIR"
    exit 1
fi

# 检查脚本是否存在
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "错误: fft.py脚本不存在: $SCRIPT_PATH"
    exit 1
fi

echo "开始处理日期: $DATE"
echo "基础目录: $BASE_DIR"
echo "脚本路径: $SCRIPT_PATH"
echo "----------------------------------------"

# 计数器
total_processed=0
success_count=0
error_count=0

# 查找匹配日期的文件夹
for main_folder in "$BASE_DIR"/"$DATE"T*; do
    if [[ -d "$main_folder" ]]; then
        echo "处理主文件夹: $(basename "$main_folder")"
        
        # 查找step开头的子文件夹
        step_folders_found=false
        for step_folder in "$main_folder"/step*; do
            if [[ -d "$step_folder" ]]; then
                step_folders_found=true
                step_name=$(basename "$step_folder")
                echo "  处理步骤文件夹: $step_name"
                
                # 检查必需的图片文件是否存在
                required_files=("image.png" "wm_image.png" "generated_image.png")
                all_files_exist=true
                
                for file in "${required_files[@]}"; do
                    if [[ ! -f "$step_folder/$file" ]]; then
                        echo "    警告: 缺少文件 $file，跳过此文件夹"
                        all_files_exist=false
                        break
                    fi
                done
                
                if [[ "$all_files_exist" == true ]]; then
                    # 执行fft.py
                    echo "    执行FFT分析..."
                    if python "$SCRIPT_PATH" --folder "$step_folder"; then
                        echo "    ✓ 成功生成: $step_folder/fft.png"
                        ((success_count++))
                    else
                        echo "    ✗ 处理失败: $step_folder"
                        ((error_count++))
                    fi
                else
                    ((error_count++))
                fi
                
                ((total_processed++))
                echo ""
            fi
        done
        
        if [[ "$step_folders_found" == false ]]; then
            echo "  未找到step开头的文件夹"
        fi
        
        echo "----------------------------------------"
    fi
done

# 检查是否找到匹配的文件夹
if [[ $total_processed -eq 0 ]]; then
    echo "未找到匹配日期 $DATE 的文件夹或step文件夹"
    echo "请检查:"
    echo "1. 日期格式是否正确 (YYYY-MM-DD)"
    echo "2. 基础目录是否正确: $BASE_DIR"
    echo "3. 是否存在以 $DATE 开头的文件夹"
    exit 1
fi

# 输出统计信息
echo "处理完成!"
echo "总共处理: $total_processed 个文件夹"
echo "成功: $success_count 个"
echo "失败: $error_count 个"

if [[ $error_count -gt 0 ]]; then
    exit 1
else
    exit 0
fi
