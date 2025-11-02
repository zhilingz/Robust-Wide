import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

dataset = load_dataset("poloclub/diffusiondb", trust_remote_code=True)

# import os

# # 设置环境变量应该在导入依赖库之前
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['PYTHONIOENCODING'] = 'utf-8'  # 避免日志乱码

# from datasets import load_dataset
# from datasets import load_dataset_builder
# # 指定要下载的数据集名称
# dataset_name = ['BleachNick/UltraEdit_500k','osunlp/MagicBrush','facebook/emu_edit_test_set','timbrooks/instructpix2pix-clip-filtered']
# # 加载数据集

# for i in range(len(dataset_name)):
#     print(f"开始下载数据集: {dataset_name[i]}")
#     # 获取数据集信息
#     builder = load_dataset_builder(dataset_name[i])
#     # 下载到指定路径
#     data_dir="/public/zhangzhiling/datasets/"+dataset_name[i]
#     builder.download_and_prepare(output_dir=data_dir)
#     # 加载已下载的数据集
#     dataset = load_dataset(dataset_name[i], cache_dir=data_dir)
#     # dataset = load_dataset(dataset_name[i], cache_dir='/public/zhangzhiling/datasets/'+dataset_name[i])
#     # datasets.append(dataset)
#     print(f"已下载数据集: {dataset_name[i]}")
#     print(dataset)

# print("所有数据集下载完成！")





