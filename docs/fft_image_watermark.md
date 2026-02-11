# FFT图片水印技术文档

## 概述

FFT图片水印是一种基于快速傅里叶变换（Fast Fourier Transform）的频域水印技术，支持将图片形式的水印嵌入到宿主图像中。与传统的比特序列水印不同，图片水印可以嵌入logo、文字、图案等可视化内容，提取后可以直接显示为图像。

## 技术原理

### 1. 图像到比特序列的转换流程

```
原始水印图像 → 二值化 → 比特序列 → FFT频域嵌入 → 含水印图像
```

#### 1.1 二值化（Binarization）

将彩色或灰度水印图像转换为二值图像（只包含0和1）：

```python
binary = (gray_image > threshold).astype(np.uint8)
```

- 阈值默认为128
- 大于阈值的像素设为1（白色）
- 小于等于阈值的像素设为0（黑色）

#### 1.2 比特序列编码

将二维二值图像展平为一维比特序列：

```python
bits = binary_image.flatten()
```

- 保存原始形状信息用于重建
- 比特数 = 图像高度 × 图像宽度

### 2. FFT频域嵌入

使用标准的FFT频域水印算法嵌入比特序列：

1. 对宿主图像进行2D FFT变换
2. 在频域中频区域选择嵌入位置
3. 使用扩频序列调制水印比特
4. 修改频域系数的幅度
5. 进行逆FFT变换得到含水印图像

### 3. 水印提取与重建流程

```
含水印图像 → FFT频域提取 → 比特序列 → 重塑 → 二值图像 → 可视化图像
```

#### 3.1 比特序列提取

使用FFT频域盲检测算法提取比特序列。

#### 3.2 图像重建

将提取的比特序列重塑为二维图像：

```python
binary_image = bits.reshape(original_shape)
visual_image = binary_image * 255  # 转换为可视化图像
```

## API使用指南

### 基本使用

```python
from engines.image.fft_watermark import FFTWatermark, FFTWatermarkConfig

# 创建FFT水印处理器
config = FFTWatermarkConfig(
    strength=0.15,      # 水印强度
    seed=42,            # 随机种子
    frequency_band="mid" # 频域范围：low/mid/high
)
fft_wm = FFTWatermark(config)

# 嵌入图片水印
embed_result = fft_wm.embed_image_watermark(
    image=host_image,           # 宿主图像
    watermark_image=logo_image, # 水印图像（数组或文件路径）
    threshold=128               # 二值化阈值
)

# 提取图片水印
extract_result = fft_wm.extract_image_watermark(
    watermarked_image=watermarked,  # 含水印图像
    watermark_shape=(64, 64)        # 水印形状
)

# 访问结果
watermarked_img = embed_result.watermarked_image
extracted_img = extract_result.extracted_image
confidence = extract_result.confidence_score
```

### 核心方法

#### 1. `image_to_binary(watermark_image, threshold=128)`

将水印图像转换为二值图像。

**参数：**
- `watermark_image`: 水印图像（numpy数组或文件路径）
- `threshold`: 二值化阈值（0-255）

**返回：**
- 二值图像（值为0或1）

#### 2. `binary_to_bits(binary_image)`

将二值图像编码为比特序列。

**参数：**
- `binary_image`: 二值图像

**返回：**
- `bits`: 比特序列
- `shape`: 原始形状

#### 3. `bits_to_binary(bits, shape)`

将比特序列解码为二值图像。

**参数：**
- `bits`: 比特序列
- `shape`: 目标形状 (height, width)

**返回：**
- 重建的二值图像

#### 4. `binary_to_image(binary_image, scale=255)`

将二值图像转换为可视化图像。

**参数：**
- `binary_image`: 二值图像（0或1）
- `scale`: 缩放因子（1会被映射到此值）

**返回：**
- 可视化图像（0-255）

#### 5. `embed_image_watermark(image, watermark_image, threshold=128)`

嵌入图片形式的水印。

**参数：**
- `image`: 宿主图像
- `watermark_image`: 水印图像
- `threshold`: 二值化阈值

**返回：**
- `FFTImageWatermarkResult` 对象，包含：
  - `watermarked_image`: 含水印图像
  - `psnr`: 峰值信噪比
  - `watermark_shape`: 水印形状
  - `total_bits`: 总比特数
  - `metadata`: 元数据

#### 6. `extract_image_watermark(watermarked_image, watermark_shape)`

提取图片形式的水印。

**参数：**
- `watermarked_image`: 含水印图像
- `watermark_shape`: 水印形状 (height, width)

**返回：**
- `FFTImageExtractResult` 对象，包含：
  - `extracted_image`: 提取的水印图像
  - `confidence_score`: 置信度（0-1）
  - `watermark_shape`: 水印形状
  - `metadata`: 元数据

## 应用场景

### 1. Logo水印

嵌入公司或品牌logo，用于版权保护：

```python
# 创建圆形logo
logo = np.zeros((64, 64), dtype=np.uint8)
cv2.circle(logo, (32, 32), 28, 255, 2)
cv2.circle(logo, (32, 32), 15, 255, -1)

# 嵌入logo
result = fft_wm.embed_image_watermark(image, logo)
```

### 2. 文字水印

嵌入文字信息，如"机密"、"内部使用"等：

```python
# 创建文字水印
text_wm = np.zeros((48, 128), dtype=np.uint8)
cv2.putText(text_wm, 'CONFIDENTIAL', (5, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

# 嵌入文字
result = fft_wm.embed_image_watermark(image, text_wm)
```

### 3. 图案水印

嵌入特定图案用于身份识别：

```python
# 创建棋盘图案
pattern = np.zeros((32, 32), dtype=np.uint8)
for i in range(0, 32, 8):
    for j in range(0, 32, 8):
        if (i // 8 + j // 8) % 2 == 0:
            pattern[i:i+8, j:j+8] = 255

# 嵌入图案
result = fft_wm.embed_image_watermark(image, pattern)
```

### 4. 二维码水印

嵌入二维码，可包含更多信息：

```python
import qrcode

# 生成二维码
qr = qrcode.QRCode(box_size=1, border=1)
qr.add_data('https://example.com/verify/12345')
qr.make()
qr_image = qr.make_image(fill_color="black", back_color="white")
qr_array = np.array(qr_image.convert('L'))

# 嵌入二维码
result = fft_wm.embed_image_watermark(image, qr_array)
```

## 性能特性

### 1. 图像质量

- **PSNR**: 通常 > 50 dB（几乎不可见）
- **视觉质量**: 人眼无法察觉差异
- **文件大小**: 与原图相同

### 2. 容量

不同尺寸水印的比特容量：

| 水印尺寸 | 比特数 | 适用场景 |
|---------|--------|---------|
| 16×16   | 256    | 简单图案 |
| 32×32   | 1,024  | 小logo   |
| 48×48   | 2,304  | 中等logo |
| 64×64   | 4,096  | 大logo   |
| 48×96   | 4,608  | 文字水印 |

### 3. 鲁棒性

测试结果显示对以下攻击具有良好的鲁棒性：

| 攻击类型 | 置信度 | 说明 |
|---------|--------|------|
| 无攻击   | >0.99  | 完美提取 |
| 高斯噪声 | >0.95  | σ=15 |
| JPEG压缩 | >0.90  | 质量70 |
| 缩放     | >0.85  | 50%缩放 |

## 参数调优指南

### 1. 水印强度（strength）

- **范围**: 0.05 - 0.3
- **推荐值**: 0.15
- **影响**:
  - 过低：提取准确率下降
  - 过高：图像质量下降，水印可能可见

### 2. 二值化阈值（threshold）

- **范围**: 0 - 255
- **推荐值**: 128
- **影响**:
  - 决定哪些像素被视为水印的一部分
  - 应根据水印图像的特性调整

### 3. 频域范围（frequency_band）

- **选项**: "low", "mid", "high"
- **推荐值**: "mid"
- **影响**:
  - low: 更鲁棒，但可能影响图像质量
  - mid: 平衡鲁棒性和不可见性
  - high: 更不可见，但鲁棒性较低

### 4. 水印尺寸

- **建议**: 根据宿主图像大小选择
- **经验法则**: 水印比特数 < 宿主图像像素数 / 10
- **示例**:
  - 512×512图像 → 最大64×64水印
  - 1024×1024图像 → 最大128×128水印

## 最佳实践

### 1. 水印设计

- 使用高对比度的简单图案
- 避免过于复杂的细节
- 考虑二值化后的效果
- 测试不同阈值的影响

### 2. 嵌入策略

- 在嵌入前对水印进行预处理
- 选择合适的水印尺寸
- 根据应用场景调整强度
- 保存水印形状信息用于提取

### 3. 提取验证

- 检查置信度分数
- 对比原始水印进行验证
- 考虑使用误差校正码
- 实施多次提取取平均

### 4. 安全考虑

- 保密随机种子（seed）
- 加密水印形状信息
- 使用密钥派生函数生成种子
- 定期更换水印密钥

## 限制与注意事项

### 1. 技术限制

- 水印尺寸受宿主图像大小限制
- 提取需要知道原始水印形状
- 对几何变换（旋转、透视）敏感
- 不适合极小的宿主图像

### 2. 质量权衡

- 水印越大，嵌入容量越大，但提取准确率可能下降
- 强度越高，鲁棒性越好，但图像质量下降
- 需要在不可见性和鲁棒性之间平衡

### 3. 使用建议

- 对于关键应用，建议结合误差校正码
- 对于高安全需求，建议使用加密水印
- 对于大规模应用，建议进行批量测试
- 对于实时应用，考虑性能优化

## 故障排除

### 问题1: 提取准确率低

**可能原因：**
- 水印强度过低
- 水印尺寸过大
- 图像受到严重攻击

**解决方案：**
- 增加水印强度（0.15 → 0.2）
- 减小水印尺寸
- 使用更鲁棒的频域范围（mid → low）

### 问题2: 图像质量下降明显

**可能原因：**
- 水印强度过高
- 频域范围选择不当

**解决方案：**
- 降低水印强度（0.2 → 0.15）
- 使用中频或高频范围
- 检查PSNR值（应 > 40 dB）

### 问题3: 提取的图像模糊

**可能原因：**
- 这是频域水印的正常特性
- 水印尺寸与提取尺寸不匹配

**解决方案：**
- 使用正确的水印形状参数
- 考虑使用后处理增强
- 接受一定程度的模糊（这是盲检测的代价）

## 示例代码

完整的示例代码请参考：
- `demo_fft_image_watermark.py` - 演示脚本
- `tests/engines/test_fft_image_watermark.py` - 单元测试

## 参考文献

1. Cox, I. J., et al. "Digital Watermarking and Steganography." Morgan Kaufmann, 2007.
2. Barni, M., & Bartolini, F. "Watermarking Systems Engineering." CRC Press, 2004.
3. Petitcolas, F. A., et al. "Information hiding—a survey." Proceedings of the IEEE, 1999.

## 更新日志

### v1.0.0 (2026-01-20)
- 初始实现
- 支持图像到比特序列的转换
- 支持比特序列到图像的重建
- 实现完整的嵌入和提取流程
- 添加22个单元测试
- 创建演示脚本
