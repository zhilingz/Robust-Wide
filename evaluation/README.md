# 全面水印测评说明文档

## 概述

`evaluation.py` 是一个全面的水印测评脚本，用于评估水印模型在多种场景下的性能表现。

## 测评场景

### 场景1: 无失真测试
- **目的**: 测试在没有任何失真的情况下水印的恢复能力
- **方法**: 直接从加水印的图像中提取水印并计算BER
- **意义**: 评估基础的水印嵌入和提取能力

### 场景2: 图像编辑失真测试
- **目的**: 测试经过生成模型编辑后水印的鲁棒性
- **方法**: 使用 Instruct-Pix2Pix 模型对加水印的图像进行编辑，然后提取水印
- **意义**: 评估水印在实际图像编辑应用场景下的鲁棒性

### 场景3: 通用失真测试
- **目的**: 测试经过各种图像处理操作后水印的鲁棒性
- **方法**: 对加水印的图像应用多种失真操作，然后提取水印
- **失真类型**: 
  - **JPEG压缩**: `jpeg_50`, `jpeg_30` (质量50和30)
  - **模糊操作**: `median_blur`, `gaussian_blur`
  - **噪声**: `gaussian_noise`, `noise_denoise`
  - **锐化**: `sharpness`
  - **亮度调整**: `brightness_down`, `brightness_up`
  - **颜色调整**: `contrast`, `saturation`, `hue`
  - **几何变换**: `random_crop_80`, `random_rotation`
  - **缩放**: `resize_50` (缩小到50%再放大)

## 使用方法

### 方法1: 使用Shell脚本（推荐）

```bash
# 基本用法
bash evaluation.sh <checkpoint_dir>

# 示例
bash evaluation.sh experiments/my_watermark_model/step10000
```

### 方法2: 直接运行Python脚本

```bash
python evaluation.py \
    --ckpt_dir experiments/my_watermark_model/step10000 \
    --eval_img_dir /path/to/evaluation/dataset \
    --output_dir experiments/my_watermark_model/evaluation \
    --test_size 100 \
    --num_inference_steps 10 \
    --guidance_scale 10.0 \
    --image_guidance_scale 1.5 \
    --device cuda:0
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt_dir` | str | **必需** | 水印模型checkpoint目录路径 |
| `--eval_img_dir` | str | 默认路径 | 评估图像数据集路径 |
| `--output_dir` | str | `{ckpt_dir}/evaluation` | 输出结果目录 |
| `--test_size` | int | 100 | 测试样本数量 |
| `--num_inference_steps` | int | 10 | Instruct-Pix2Pix推理步数 |
| `--guidance_scale` | float | 10.0 | 文本引导强度 |
| `--image_guidance_scale` | float | 1.5 | 图像引导强度 |
| `--device` | str | cuda:0 | 计算设备 |

## 输出结果

测评完成后，会在输出目录中生成以下文件：

### 1. 日志文件: `evaluation.log`
记录详细的测评过程和结果

### 2. 结果JSON文件: `evaluation_results.json`
包含所有测评指标的统计结果，格式如下：

```json
{
  "test_size": 100,
  "message_length": 256,
  "image_quality": {
    "psnr_mean": 42.5,
    "psnr_std": 1.2,
    "ssim_mean": 0.985,
    "ssim_std": 0.01,
    "lpips_mean": 0.015,
    "lpips_std": 0.005
  },
  "scenario_1_no_distortion": {
    "ber_mean": 0.001,
    "ber_std": 0.0005
  },
  "scenario_2_edit_distortion": {
    "ber_mean": 0.05,
    "ber_std": 0.02
  },
  "scenario_3_common_distortions": {
    "average_ber_mean": 0.03,
    "average_ber_std": 0.015,
    "distortion_details": {
      "jpeg_50": {"ber_mean": 0.02, "ber_std": 0.01},
      "gaussian_blur": {"ber_mean": 0.025, "ber_std": 0.012},
      ...
    }
  }
}
```

### 3. 样本图像目录: `sample_images/`
保存前3个样本的可视化结果，每张图包含：
- 原始图像
- 水印图像
- 残差图像
- 编辑后图像

## 评估指标说明

### 图像质量指标
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比，越高越好（通常>40dB为优秀）
- **SSIM** (Structural Similarity Index): 结构相似性，范围[0,1]，越接近1越好
- **LPIPS** (Learned Perceptual Image Patch Similarity): 感知相似性，越小越好

### 水印性能指标
- **BER** (Bit Error Rate): 比特错误率，范围[0,1]
  - 0.0: 完美恢复，无错误
  - 0.5: 完全随机，相当于没有水印
  - <0.1: 良好性能
  - <0.05: 优秀性能

## 模型结构说明

本脚本支持评估基于以下架构的水印模型：

### 编码器 (EncoderUnet)
- 基于U-Net架构
- 输入: 原始图像 + 水印消息
- 输出: 加水印的图像

### 解码器 (DecoderResnet)
- 基于ResNet架构
- 输入: 加水印的图像（可能经过失真）
- 输出: 提取的水印消息

### 水印模型 (WatermarkModel)
完整的水印系统，包含编码器和解码器：
```python
class WatermarkModel(nn.Module):
    def __init__(self, wm_enc_config, wm_dec_config, device="cuda", weight_dtype=torch.float32):
        super(WatermarkModel, self).__init__()
        self.encoder = EncoderUnet(**wm_enc_config)
        self.decoder = DecoderResnet(**wm_dec_config)
```

## 自定义模型评估

如果要评估自定义的水印模型，需要：

1. **模型结构**: 在 `model.py` 中定义模型结构
2. **配置文件**: checkpoint目录中需要包含 `wm_model_config.yaml`
3. **模型权重**: checkpoint目录中需要包含 `wm_model.ckpt`

配置文件示例：
```yaml
wm_enc_config:
  image_size: 512
  message_length: 256
  in_channels: 4
  channels: 64
  norm_type: "batch"
  final_skip: true

wm_dec_config:
  image_size: 512
  message_length: 256
  in_channels: 3
  norm_type: "batch"

weight_dtype: "float32"
device: "cuda"
```

## 注意事项

1. **显存需求**: 测评需要加载两个大模型（水印模型 + Instruct-Pix2Pix），建议至少16GB显存
2. **评估时间**: 每个样本需要进行15+种测试，100个样本大约需要1-2小时
3. **随机种子**: 脚本中固定了随机种子（42），确保结果可复现
4. **数据集**: 默认使用 Instruct-Pix2Pix 数据集，可通过参数修改

## 常见问题

### Q: 显存不足怎么办？
A: 可以减小 `test_size` 参数，或者使用更小的 batch size（需要修改代码）

### Q: 如何只测试特定场景？
A: 可以注释掉 `comprehensive_evaluation()` 函数中不需要的场景代码

### Q: 如何添加新的失真类型？
A: 在 `apply_distortions()` 函数中添加新的失真类型，并在 `distortion_types` 列表中注册

### Q: 测评结果如何解读？
A: 
- 无失真BER应接近0（<0.01为优秀）
- 编辑失真BER取决于编辑强度（<0.1为良好）
- 通用失真BER越小越好，不同失真类型可接受的BER范围不同

## 示例输出

```
================================================================================
全面测评结果统计
================================================================================

【图像质量指标】
PSNR:  42.345 ± 1.234 dB
SSIM:  0.9856 ± 0.0123
LPIPS: 0.0145 ± 0.0056

【场景1: 无失真】
BER: 0.001234 ± 0.000567

【场景2: 图像编辑失真】
BER: 0.045678 ± 0.012345

【场景3: 通用失真】
jpeg_50             : 0.023456 ± 0.006789
jpeg_30             : 0.034567 ± 0.008901
median_blur         : 0.028901 ± 0.007123
gaussian_blur       : 0.025678 ± 0.006456
gaussian_noise      : 0.032345 ± 0.008234
...

通用失真平均BER: 0.029876 ± 0.007456
================================================================================
```

## 技术支持

如有问题或建议，请联系项目维护者。

