from functools import partial
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from torchvision import transforms
import random

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
    
    num_samples = len(combined_dataset)

    # 检查样本数量
    if num_samples < train_size:
        print(f"数据集样本数量不足，使用所有样本作为训练集。")
        train_dataset = combined_dataset  # 使用所有样本作为训练集
    else:
        # 分割数据集为训练集
        train_dataset = combined_dataset.select(range(train_size))
    
    # 测试集：永远从 instructpix2pix 数据集中加载
    print("从 instructpix2pix 数据集中加载测试集")
    instructpix2pix_dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered")  # 加载 instructpix2pix 数据集
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