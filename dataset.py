from functools import partial
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Image
from torchvision import transforms
import random
import os
import json

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def preprocess_train(examples, image_size):
    train_transforms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
    )
    
    # 打印 examples 的 keys
    print("Examples keys:", examples.keys())
    
    # 根据键选择图像
    image_keys = ["original_image", "source_image", "source_img", "image"]
    images = None
    
    for key in image_keys:
        if key in examples:
            images = np.concatenate(
                [convert_to_np(image, image_size) for image in examples[key]]
            )
            break  # 找到第一个匹配的键后退出循环
    
    if images is None:
        raise KeyError("None of the expected image keys found in examples.")
    
    images = torch.tensor(images)
    images = 2 * (images / 255) - 1
    images = train_transforms(images)
    
    # 根据键选择 prompt
    prompt_keys = ["edit_prompt", "instruction"]
    prompt = None

    for key in prompt_keys:
        if key in examples:
            prompt = list(examples[key])  # 转换为列表
            break  # 找到第一个匹配的键后退出循环

    if prompt is None:
        raise KeyError("None of the expected prompt keys found in examples.")

    examples["image"] = images.reshape(-1, 3, image_size, image_size)
    examples["prompt"] = prompt
    return examples

def collate_fn(examples):
    image = torch.stack([example["image"] for example in examples])
    image = image.to(memory_format=torch.contiguous_format).float()
    prompt = [example["prompt"] for example in examples]
    return {"image": image, "prompt": prompt}

def get_hugging_dataset(instance_data_root, image_size, accelerator, train_size=20000, test_size=1200):
    # 加载数据集
    dataset = load_dataset(instance_data_root)
    
    # 合并所有分支的数据到一个统一的Dataset对象
    all_splits = list(dataset.keys())
    if len(all_splits) > 1:
        combined_dataset = concatenate_datasets([dataset[split] for split in all_splits])
    else:
        combined_dataset = dataset[all_splits[0]]
    
    train_dataset = combined_dataset  # 使用所有样本作为训练集
    
    # 测试集：永远从 instructpix2pix 数据集中加载
    print("从 instructpix2pix 数据集中加载测试集")
    instructpix2pix_dataset = load_dataset("/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f")  # 加载 instructpix2pix 数据集
    test_dataset = instructpix2pix_dataset["train"].select(range(test_size))  # 选择测试集样本
    
    # 确保只有主进程执行数据集转换
    with accelerator.main_process_first():
        # 应用预处理
        train_dataset = train_dataset.with_transform(partial(preprocess_train, image_size=image_size))
        test_dataset = test_dataset.with_transform(partial(preprocess_train, image_size=image_size))
        
        # 输出训练集和测试集的大小
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def get_filtered_dataset(args, image_size, accelerator, train_size=20000, test_size=1200):
    """
    从筛选后的数据集中加载数据
    """
    # 构建筛选数据集路径
    # dataset_name = args.train_data_dir.split("/")[-4]  # 提取数据集名称
    # model_name = args.model_dir.split("/")[-1]  # 提取模型名称
    # filtered_dataset_path = f"filtered_datasets/{dataset_name}/{model_name}"
    filtered_dataset_path = args.train_data_dir
    if not os.path.exists(filtered_dataset_path):
        raise FileNotFoundError(f"筛选数据集不存在: {filtered_dataset_path}")
    print(f"从筛选数据集加载: {filtered_dataset_path}")
    
    # 加载元数据
    metadata_path = os.path.join(filtered_dataset_path, "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    prompts = metadata["edit_prompt"]

    # 构建数据集
    images_dir = os.path.join(filtered_dataset_path, "images")
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    image_files.sort()  # 确保顺序一致
    
    # 处理图像文件数量与prompt数量不一致的情况
    if len(image_files) != len(prompts):
        min_count = min(len(image_files), len(prompts))
        print(f"警告: 图像文件数量({len(image_files)})与prompt数量({len(prompts)})不一致")
        print(f"将使用最小数量: {min_count}")
        
        # 截取到最小数量
        image_files = image_files[:min_count]
        prompts = prompts[:min_count]
    
    # 创建数据集字典
    dataset_dict = {
        "image": [os.path.join(images_dir, img_file) for img_file in image_files],
        "edit_prompt": prompts
    }
    
    # 创建Dataset对象
    features = Features({
        "image": Image(),
        "edit_prompt": Value("string")
    })
    
    filtered_dataset = Dataset.from_dict(dataset_dict, features=features)
    
    # 分割训练集和验证集
    total_samples = len(filtered_dataset)
    if total_samples < train_size:
        print(f"筛选数据集样本数量({total_samples})小于请求的训练集大小({train_size})，使用所有样本")
        train_dataset = filtered_dataset
    else:
        train_dataset = filtered_dataset.select(range(train_size))
    
    # 测试集：仍然从原始 instructpix2pix 数据集中加载
    print("从 instructpix2pix 数据集中加载测试集")
    instructpix2pix_dataset = load_dataset("/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f")
    test_dataset = instructpix2pix_dataset["train"].select(range(test_size))
    
    # 确保只有主进程执行数据集转换
    with accelerator.main_process_first():
        # 应用预处理
        train_dataset = train_dataset.with_transform(partial(preprocess_train, image_size=image_size))
        test_dataset = test_dataset.with_transform(partial(preprocess_train, image_size=image_size))
        
        # 输出训练集和测试集的大小
        print(f"筛选训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
    
    return train_dataset, test_dataset