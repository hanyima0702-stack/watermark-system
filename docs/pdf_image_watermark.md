# PDF图片水印功能文档

## 概述

PDF图片水印功能允许将图片形式的水印嵌入到PDF文档中，并能够提取出可视化的水印图像。该功能基于FFT（快速傅里叶变换）频域水印技术实现。

## 核心特性

### 1. 图片水印嵌入
- 将任意图片转换为二值图像作为水印
- 使用FFT频域变换嵌入到PDF的每一页
- 水印不可见，不影响文档的视觉效果
- 支持自定义二值化阈值

### 2. 图片水印提取
- 从含水印的PDF中提取水印图像
- 生成包含可视化水印的PDF报告
- 提供置信度评估
- 支持从元数据自动读取水印形状

## 技术原理

### 嵌入流程

```
原始PDF → 每页转图像 → FFT变换 → 频域嵌入 → 逆FFT → 含水印PDF
                ↑
            水印图片 → 二值化 → 比特编码
```

1. **水印预处理**
   - 将水印图像转换为灰度图
   - 二值化处理（阈值可调）
   - 编码为比特序列

2. **PDF页面处理**
   - 将PDF每一页渲染为高分辨率图像（300 DPI）
   - 转换为RGB格式

3. **频域嵌入**
   - 对图像进行2D FFT变换
   - 在频域中嵌入水印比特序列
   - 使用扩频技术增强鲁棒性
   - 进行逆FFT变换得到含水印图像

4. **PDF重建**
   - 将含水印图像转换回PDF页面
   - 在元数据中存储水印信息
   - 保存为新的PDF文件

### 提取流程

```
含水印PDF → 页面转图像 → FFT变换 → 频域提取 → 比特解码 → 水印图像 → 可视化PDF
```

1. **PDF解析**
   - 打开含水印的PDF
   - 从元数据读取水印形状信息
   - 渲染页面为图像

2. **频域提取**
   - 对图像进行2D FFT变换
   - 从频域中提取水印比特序列
   - 计算提取置信度

3. **水印重建**
   - 将比特序列解码为二值图像
   - 转换为可视化图像（0-255灰度）

4. **报告生成**
   - 创建PDF报告
   - 插入提取的水印图像
   - 添加元数据和置信度信息

## 使用方法

### 基本用法

```python
from pathlib import Path
from engines.document.pdf_processor import PDFProcessor
import cv2

# 初始化处理器
processor = PDFProcessor(watermark_method='fft')

# 1. 嵌入图片水印
watermark_img = cv2.imread("watermark.png")
watermarked_pdf = processor.add_invisible_image_watermark(
    file_path=Path("document.pdf"),
    watermark_image=watermark_img,
    threshold=128
)

# 2. 提取图片水印
watermark_shape = (64, 64)  # 水印图像的形状
extracted_pdf = processor.extract_invisible_image_watermark_to_pdf(
    file_path=watermarked_pdf,
    watermark_shape=watermark_shape
)
```

### 使用文件路径

```python
# 直接使用图片文件路径
watermarked_pdf = processor.add_invisible_image_watermark(
    file_path=Path("document.pdf"),
    watermark_image="watermark.png",  # 文件路径
    threshold=128
)
```

### 自动读取水印形状

```python
# 从元数据自动读取水印形状
extracted_pdf = processor.extract_invisible_image_watermark_to_pdf(
    file_path=watermarked_pdf,
    watermark_shape=None  # 自动从元数据读取
)
```

### 创建自定义水印图像

```python
import numpy as np
import cv2

# 创建简单的文字水印
def create_text_watermark(text="DEMO", size=(64, 64)):
    img = np.ones(size, dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_BOLD
    cv2.putText(img, text, (10, 40), font, 1.0, 0, 2)
    return img

watermark = create_text_watermark("SECRET")
```

## API参考

### PDFProcessor.add_invisible_image_watermark()

嵌入图片形式的水印到PDF。

**参数：**
- `file_path` (Path): PDF文件路径
- `watermark_image` (Union[np.ndarray, str, Path]): 水印图像（数组或文件路径）
- `threshold` (int): 二值化阈值，默认128

**返回：**
- `Path`: 含水印的PDF文件路径

**异常：**
- `DocumentProcessingError`: 文档处理失败
- `WatermarkEmbeddingError`: 水印嵌入失败

### PDFProcessor.extract_invisible_image_watermark_to_pdf()

提取图片水印并生成可视化PDF报告。

**参数：**
- `file_path` (Path): 含水印的PDF文件路径
- `watermark_shape` (Tuple[int, int]): 水印图像形状 (height, width)，可选
- `output_path` (Path): 输出PDF路径，可选

**返回：**
- `Path`: 生成的报告PDF路径

**异常：**
- `DocumentProcessingError`: 文档处理失败

## 配置参数

### 二值化阈值 (threshold)

控制水印图像的二值化过程：
- **低阈值 (64)**: 更多像素被视为黑色，水印更密集
- **中阈值 (128)**: 平衡的二值化，推荐使用
- **高阈值 (192)**: 更多像素被视为白色，水印更稀疏

### FFT配置

可以通过 `FFTWatermarkConfig` 调整FFT参数：

```python
from engines.image.fft_watermark import FFTWatermarkConfig

config = FFTWatermarkConfig(
    strength=0.15,        # 嵌入强度
    seed=42,              # 随机种子
    frequency_band="mid"  # 频带选择: low, mid, high
)

processor = PDFProcessor(watermark_method='fft')
processor.watermark_processor.config = config
```

## 性能指标

### 图像质量
- **PSNR**: 通常 > 40 dB
- **视觉效果**: 水印完全不可见

### 处理速度
- **嵌入**: ~2-5秒/页（300 DPI）
- **提取**: ~2-5秒/页（300 DPI）

### 水印容量
- 取决于水印图像大小
- 64x64图像 = 4096比特
- 128x128图像 = 16384比特

## 鲁棒性

### 支持的攻击
- ✓ JPEG压缩（质量 > 70）
- ✓ 轻微噪声
- ✓ 小幅度缩放

### 不支持的攻击
- ✗ 大幅度裁剪
- ✗ 严重压缩（质量 < 50）
- ✗ 几何变换（旋转、倾斜）

## 应用场景

1. **版权保护**
   - 嵌入公司Logo作为水印
   - 证明文档所有权

2. **文档溯源**
   - 嵌入二维码水印
   - 追踪文档来源

3. **防伪标识**
   - 嵌入特殊图案
   - 验证文档真实性

4. **隐蔽通信**
   - 嵌入隐藏信息
   - 秘密数据传输

## 示例代码

### 完整示例

```python
from pathlib import Path
from engines.document.pdf_processor import PDFProcessor
import numpy as np
import cv2

# 创建水印图像
def create_logo_watermark():
    img = np.ones((64, 64), dtype=np.uint8) * 255
    cv2.circle(img, (32, 32), 20, 0, -1)
    cv2.putText(img, "©", (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return img

# 初始化
processor = PDFProcessor(watermark_method='fft')
watermark = create_logo_watermark()

# 嵌入
print("嵌入水印...")
watermarked = processor.add_invisible_image_watermark(
    file_path=Path("document.pdf"),
    watermark_image=watermark,
    threshold=128
)
print(f"完成: {watermarked}")

# 提取
print("提取水印...")
report = processor.extract_invisible_image_watermark_to_pdf(
    file_path=watermarked,
    watermark_shape=(64, 64)
)
print(f"报告: {report}")
```

## 注意事项

1. **水印尺寸**
   - 建议使用64x64或128x128
   - 过大的水印会降低嵌入质量
   - 过小的水印提取准确度低

2. **PDF质量**
   - 使用300 DPI渲染以保证质量
   - 处理时间与页数和分辨率成正比

3. **元数据存储**
   - 水印形状信息存储在PDF元数据中
   - 删除元数据会导致无法自动提取

4. **兼容性**
   - 支持所有标准PDF格式
   - 加密PDF需要先解密

## 故障排除

### 问题：提取的水印模糊不清

**解决方案：**
- 增加嵌入强度（FFTWatermarkConfig.strength）
- 使用更大的水印图像
- 确保原始PDF质量良好

### 问题：无法从元数据读取水印形状

**解决方案：**
- 手动指定watermark_shape参数
- 检查PDF元数据是否被修改

### 问题：处理速度慢

**解决方案：**
- 降低渲染DPI（但会影响质量）
- 只处理必要的页面
- 使用更小的水印图像

## 相关文档

- [FFT水印技术文档](fft_image_watermark.md)
- [PDF处理器文档](pdf_processor.md)
- [水印提取报告生成器](watermark_report_generator.md)

## 更新日志

### v1.0.0 (2026-01-20)
- 初始版本
- 支持图片水印嵌入和提取
- 生成可视化PDF报告
- 支持元数据自动读取
