import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTHONIOENCODING'] = 'utf-8'  # 避免日志乱码

from datasets import load_dataset, load_dataset_builder

# 指定要下载的数据集名称
dataset_names = [
    'BleachNick/UltraEdit_500k',
    'osunlp/MagicBrush',
    'facebook/emu_edit_test_set',
    'timbrooks/instructpix2pix-clip-filtered'
]

# 下载并加载数据集
for name in dataset_names:
    print(f"开始下载数据集: {name}")
    try:
        # 获取数据集信息
        builder = load_dataset_builder(name)
        
        # 下载到默认缓存路径（无需手动指定 data_dir）
        builder.download_and_prepare()
        
        # 加载数据集
        dataset = load_dataset(name)
        
        print(f"已下载数据集: {name}")
        print(dataset)
    except Exception as e:
        print(f"下载 {name} 时出错: {str(e)}")

print("所有数据集下载完成！")