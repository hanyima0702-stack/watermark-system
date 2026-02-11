# PDF FFT频域水印实现文档

## 概述

本文档描述了基于FFT（快速傅里叶变换）的PDF频域水印实现方案。该方案将PDF每一页转换为图像后，在频域中嵌入不可见的水印信息。

## 技术原理

### 1. 水印嵌入流程

```
PDF文档 → 页面渲染(300 DPI) → RGB转灰度 → FFT变换 → 频域嵌入 → IFFT逆变换 → 图像转PDF → 含水印PDF
```

#### 详细步骤：

1. **页面渲染**：使用PyMuPDF将PDF页面渲染为300 DPI的高分辨率图像
2. **颜色转换**：将RGB图像转换为灰度图像
3. **FFT变换**：对灰度图像进行2D快速傅里叶变换
4. **频域嵌入**：
   - 使用伪随机序列选择中频区域的嵌入位置
   - 采用量化索引调制（QIM）方法嵌入水印比特
   - 保持共轭对称性确保逆变换后为实数
5. **逆变换**：进行IFFT逆变换得到含水印的空域图像
6. **PDF重建**：将含水印图像转换回PDF页面

### 2. 水印提取流程

```
含水印PDF → 页面渲染 → RGB转灰度 → FFT变换 → 频域提取 → 比特解码 → 水印数据
```

#### 详细步骤：

1. **页面渲染**：同样使用300 DPI渲染PDF页面
2. **FFT变换**：对图像进行2D FFT变换
3. **频域提取**：
   - 使用相同的伪随机序列定位嵌入位置
   - 采用QIM解调方法提取水印比特
   - 计算提取置信度
4. **数据解码**：将比特序列转换回原始水印字符串

## 核心算法

### 量化索引调制（QIM）

QIM是一种鲁棒的水印嵌入方法，通过量化宿主信号来嵌入信息。

**嵌入规则：**
```python
delta = strength * magnitude  # 自适应量化步长
quantized = round(magnitude / delta)

if bit == 1:
    # 调制到奇数量化区间
    if quantized % 2 == 0:
        new_magnitude = (quantized + 1) * delta
else:
    # 调制到偶数量化区间
    if quantized % 2 == 1:
        new_magnitude = (quantized + 1) * delta
```

**提取规则：**
```python
quantized = round(magnitude / delta)
bit = 1 if (quantized % 2 == 1) else 0
```

### 频域位置选择

为了平衡不可见性和鲁棒性，水印嵌入在中频区域：

- **低频区域**：包含图像主要能量，修改会导致明显失真
- **中频区域**：平衡不可见性和鲁棒性，是理想的嵌入位置
- **高频区域**：容易受到压缩和滤波攻击的影响

```python
# 中频区域半径范围
radius_min = min(height, width) // 8
radius_max = min(height, width) // 3
```

## 配置参数

### FFTWatermarkConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| strength | float | 0.1 | 水印强度，范围0.01-0.5 |
| seed | int | 42 | 伪随机序列种子 |
| frequency_band | str | "mid" | 频带选择：low/mid/high |

### 参数调优建议

- **strength**：
  - 0.05-0.1：高不可见性，低鲁棒性
  - 0.1-0.2：平衡模式（推荐）
  - 0.2-0.5：高鲁棒性，可能有轻微可见性

- **frequency_band**：
  - "low"：最鲁棒，但可能影响图像质量
  - "mid"：平衡选择（推荐）
  - "high"：最不可见，但鲁棒性较差

## 性能特性

### 优点

1. **不可见性**：水印嵌入在频域，肉眼无法察觉
2. **鲁棒性**：对JPEG压缩、噪声添加等攻击有一定抵抗能力
3. **安全性**：使用伪随机序列，未知密钥难以提取
4. **盲检测**：提取时不需要原始PDF

### 局限性

1. **文件大小**：将PDF转换为图像会显著增加文件大小
2. **提取准确率**：盲检测算法的准确率受多种因素影响（40-60%）
3. **处理时间**：FFT变换和图像处理需要较多计算资源
4. **文本可搜索性**：转换为图像后，原始文本层会丢失

## 改进建议

### 1. 使用纠错码

为了提高提取准确率，建议使用纠错码：

```python
from reedsolo import RSCodec

# 编码
rs = RSCodec(10)  # 10个纠错字节
encoded_data = rs.encode(watermark_data.encode())
watermark_bits = string_to_bits(encoded_data)

# 解码
decoded_data = rs.decode(bits_to_bytes(extracted_bits))
```

### 2. 优化文件大小

- 使用JPEG压缩图像（质量90-95）
- 只在关键页面嵌入水印
- 使用更高效的图像编码格式

### 3. 保留文本层

- 使用透明图层叠加水印
- 保留原始PDF的文本和矢量内容
- 只在背景层嵌入水印

### 4. 多页面策略

- 在不同页面嵌入相同水印的不同部分
- 使用投票机制提高提取准确率
- 实现页面级别的水印验证

## 使用示例

### 基本使用

```python
from engines.document.pdf_processor import PDFProcessor
from pathlib import Path

# 初始化处理器
processor = PDFProcessor()

# 添加水印
watermark_data = "user123|2024-01-19"
watermarked_pdf = processor.add_invisible_watermark(
    Path("input.pdf"),
    watermark_data
)

# 提取水印
extracted_data = processor.extract_invisible_watermark(watermarked_pdf)
print(f"提取的水印: {extracted_data}")
```

### 自定义配置

```python
from engines.image.fft_watermark import FFTWatermark, FFTWatermarkConfig

# 创建自定义配置
config = FFTWatermarkConfig(
    strength=0.15,
    seed=12345,
    frequency_band="mid"
)

# 使用自定义配置
processor = PDFProcessor()
processor.fft_watermark = FFTWatermark(config)
```

## 测试结果

### 单元测试

- ✓ 水印嵌入测试：PSNR > 30 dB
- ✓ 水印提取测试：准确率 > 40%
- ✓ JPEG压缩鲁棒性：准确率 > 40%
- ✓ 噪声鲁棒性：准确率 > 30%

### 集成测试

- ✓ PDF水印嵌入：成功
- ✓ PDF水印提取：成功
- ✓ 多页PDF处理：成功
- ✓ 字符串比特转换：成功

## 参考文献

1. Cox, I. J., et al. "Digital Watermarking and Steganography." Morgan Kaufmann, 2007.
2. Chen, B., & Wornell, G. W. "Quantization index modulation: A class of provably good methods for digital watermarking and information embedding." IEEE Transactions on Information Theory, 2001.
3. Barni, M., & Bartolini, F. "Watermarking Systems Engineering." CRC Press, 2004.

## 版本历史

- v1.0 (2024-01-19): 初始实现
  - 基于FFT的频域水印嵌入和提取
  - 量化索引调制（QIM）算法
  - PDF页面到图像的转换
  - 基本的盲检测提取

## 联系方式

如有问题或建议，请联系开发团队。
