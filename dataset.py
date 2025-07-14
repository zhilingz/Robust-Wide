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
    
    # 从合并的数据集中选取指定数量的训练样本
    if len(combined_dataset) >= train_size:
        train_dataset = combined_dataset.select(range(train_size))
    else:
        print(f"数据集样本数量({len(combined_dataset)})小于请求的训练集大小({train_size})，使用所有样本")
        train_dataset = combined_dataset
    
    # 测试集：永远从 instructpix2pix 数据集中加载
    print("从 instructpix2pix 数据集中加载测试集")
    instructpix2pix_dataset = load_dataset("/public/zhangzhiling/datasets/timbrooks___instructpix2pix-clip-filtered/default/0.0.0/aa665b890915f7a42f8615bee868a9f3447e178f")  # 加载 instructpix2pix 数据集
    test_dataset = instructpix2pix_dataset["train"].select(range(test_size))  # 选择测试集样本
    
    # 确保只有主进程执行数据集转换（如果有accelerator的话）
    if accelerator is not None:
        with accelerator.main_process_first():
            # 应用预处理
            train_dataset = train_dataset.with_transform(partial(preprocess_train, image_size=image_size))
            test_dataset = test_dataset.with_transform(partial(preprocess_train, image_size=image_size))
            
            # 输出训练集和测试集的大小
            print(f"训练集大小: {len(train_dataset)}")
            print(f"测试集大小: {len(test_dataset)}")
    else:
        # 没有accelerator时直接执行
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
    基于保存的样本ID从原始数据集中加载对应样本
    """
    # 构建筛选数据集路径
    filtered_dataset_path = args.train_data_dir
    if not os.path.exists(filtered_dataset_path):
        raise FileNotFoundError(f"筛选数据集不存在: {filtered_dataset_path}")
    print(f"从筛选数据集加载: {filtered_dataset_path}")
    
    # 加载元数据
    metadata_path = os.path.join(filtered_dataset_path, "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 获取筛选后的样本ID列表
    filtered_sample_ids = metadata["filtered_sample_ids"]
    
    # 确定原始数据集路径
    # 这里假设原始数据集路径存储在metadata中，或者使用默认路径
    original_dataset_path = metadata.get("original_dataset_path")
    
    # 加载原始数据集
    print(f"从原始数据集加载: {original_dataset_path}")
    original_dataset = load_dataset(original_dataset_path)
    
    # 合并所有分支的数据到一个统一的Dataset对象
    all_splits = list(original_dataset.keys())
    if len(all_splits) > 1:
        combined_original_dataset = concatenate_datasets([original_dataset[split] for split in all_splits])
    else:
        combined_original_dataset = original_dataset[all_splits[0]]
    
    # 根据筛选的样本ID创建训练数据集
    filtered_indices = [sample_info["sample_id"] for sample_info in filtered_sample_ids]
    
    # 处理样本数量
    if len(filtered_indices) < train_size:
        print(f"筛选数据集样本数量({len(filtered_indices)})小于请求的训练集大小({train_size})，使用所有样本")
        train_indices = filtered_indices
    else:
        train_indices = filtered_indices[:train_size]
    
    # 从原始数据集中选择对应的样本
    train_dataset = combined_original_dataset.select(train_indices)
    
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