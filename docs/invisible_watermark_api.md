# 暗水印系统API文档

## 概述

暗水印系统实现了基于Block-Based Repetition机制的图片暗水印嵌入与识别，针对屏摄攻击、几何变换、局部遮挡等强对抗场景设计。系统将64位水印数据通过纠错编码扩展为128位，在图片中以32×32像素宏块为单位重复嵌入。

## 主要特性

- **高鲁棒性**: 支持旋转±45°、缩放50%-200%、裁剪30%、JPEG压缩等攻击
- **纠错能力**: 使用Reed-Solomon码，支持20%-30%的位错误自动纠正
- **几何校正**: 基于FFT的自动几何变换检测和校正
- **多数投票**: 从多个宏块中提取数据并融合，提高可靠性
- **不可见性**: 使用PPM调制，调制强度可配置，保持图像质量

## 快速开始

### 安装依赖

```bash
pip install numpy opencv-python scipy reedsolo pyyaml
```

### 基本使用

```python
from engines.image.invisible_watermark import InvisibleWatermarkProcessor

# 创建处理器
processor = InvisibleWatermarkProcessor()

# 嵌入水印
watermark = "1010101010101010101010101010101010101010101010101010101010101010"
result = processor.embed_watermark(
    image_path="input.jpg",
    watermark=watermark,
    output_path="watermarked.png"
)

if result.success:
    print(f"嵌入成功! PSNR: {result.quality_metrics['psnr']:.2f}dB")

# 提取水印
result = processor.extract_watermark(image_path="watermarked.png")

if result.success:
    print(f"提取成功! 水印: {result.watermark_data}")
    print(f"置信度: {result.confidence:.3f}")
```

## API参考

### InvisibleWatermarkProcessor

主控制器类，协调所有模块。

#### 构造函数

```python
InvisibleWatermarkProcessor(config=None, config_path=None)
```

**参数:**
- `config` (WatermarkConfig, optional): 配置对象
- `config_path` (str, optional): 配置文件路径

**示例:**

```python
# 使用默认配置
processor = InvisibleWatermarkProcessor()

# 使用自定义配置
config = WatermarkConfig(modulation_strength=12)
processor = InvisibleWatermarkProcessor(config=config)

# 从配置文件加载
processor = InvisibleWatermarkProcessor(
    config_path="config/my_config.yaml"
)
```

#### embed_watermark()

嵌入水印到图像。

```python
embed_watermark(image_path, watermark, output_path) -> EmbedResult
```

**参数:**
- `image_path` (str): 原始图像路径
- `watermark` (str): 64位水印数据（二进制字符串）
- `output_path` (str): 输出图像路径

**返回:**
- `EmbedResult`: 嵌入结果对象

**EmbedResult字段:**
- `success` (bool): 是否成功
- `watermark_data` (str): 原始64位水印
- `encoded_data` (str): 编码后的128位数据
- `block_count` (tuple): 宏块数量 (行数, 列数)
- `processing_time` (float): 处理时间（秒）
- `image_size` (tuple): 图像尺寸 (height, width)
- `quality_metrics` (dict): 质量指标 (PSNR, SSIM)
- `error_message` (str, optional): 错误信息

**示例:**

```python
result = processor.embed_watermark(
    image_path="input.jpg",
    watermark="1010" * 16,  # 64位
    output_path="output.png"
)

if result.success:
    print(f"处理时间: {result.processing_time:.2f}秒")
    print(f"宏块数量: {result.block_count}")
    print(f"PSNR: {result.quality_metrics['psnr']:.2f}dB")
    print(f"SSIM: {result.quality_metrics['ssim']:.4f}")
```

#### extract_watermark()

从图像中提取水印。

```python
extract_watermark(image_path, visualize=False) -> ExtractionResult
```

**参数:**
- `image_path` (str): 带水印的图像路径
- `visualize` (bool): 是否生成可视化输出

**返回:**
- `ExtractionResult`: 提取结果对象

**ExtractionResult字段:**
- `success` (bool): 是否成功
- `watermark_data` (str): 提取的64位水印
- `confidence` (float): 整体置信度 [0, 1]
- `bit_confidences` (list): 每位的置信度
- `detected_rotation` (float): 检测到的旋转角度（度）
- `detected_scale` (float): 检测到的缩放比例
- `grid_offset` (tuple): 网格偏移量 (x, y)
- `total_blocks` (int): 总宏块数
- `valid_blocks` (int): 有效宏块数
- `error_rate` (float): 错误率
- `processing_time` (float): 处理时间（秒）
- `fft_spectrum` (ndarray, optional): FFT频谱（可视化）
- `block_map` (ndarray, optional): 宏块地图（可视化）
- `error_message` (str, optional): 错误信息

**示例:**

```python
result = processor.extract_watermark(
    image_path="watermarked.png",
    visualize=True
)

if result.success:
    print(f"水印: {result.watermark_data}")
    print(f"置信度: {result.confidence:.3f}")
    print(f"有效宏块: {result.valid_blocks}/{result.total_blocks}")
    print(f"旋转: {result.detected_rotation:.2f}°")
    print(f"缩放: {result.detected_scale:.2f}")
else:
    print(f"提取失败: {result.error_message}")
```

### WatermarkConfig

配置类，定义系统参数。

#### 构造函数

```python
WatermarkConfig(
    ecc_type="bch",
    ecc_n=127,
    ecc_k=64,
    scramble_seed=12345,
    block_size=32,
    header_pattern="1110010",
    modulation_strength=10,
    color_space="yuv",
    min_confidence=0.3,
    max_rotation=45.0,
    max_scale=2.0,
    enable_180_retry=True,
    enable_gpu=False,
    num_threads=4,
    enable_cache=True,
    visualization_enabled=False,
    visualization_output_dir="./debug_output",
    save_fft_spectrum=True,
    save_block_map=True
)
```

**主要参数说明:**

**编码配置:**
- `ecc_type`: 纠错编码类型 ("bch" 或 "rs")
- `ecc_n`: 编码后长度（比特数）
- `ecc_k`: 原始数据长度（比特数）
- `scramble_seed`: 加扰种子

**宏块配置:**
- `block_size`: 宏块大小（像素），支持16/32/64
- `header_pattern`: Header模式字符串

**PPM配置:**
- `modulation_strength`: 调制强度（建议8-12）
- `color_space`: 色彩空间 ("yuv")

**提取配置:**
- `min_confidence`: 最小置信度阈值 [0, 1]
- `max_rotation`: 最大旋转角度（度）
- `max_scale`: 最大缩放比例
- `enable_180_retry`: 是否启用180度旋转重试

**性能配置:**
- `enable_gpu`: 是否启用GPU加速
- `num_threads`: 线程数
- `enable_cache`: 是否启用缓存

**可视化配置:**
- `visualization_enabled`: 是否启用可视化
- `visualization_output_dir`: 可视化输出目录
- `save_fft_spectrum`: 是否保存FFT频谱
- `save_block_map`: 是否保存宏块地图

#### 类方法

**from_yaml()**

从YAML文件加载配置。

```python
@classmethod
from_yaml(cls, yaml_path) -> WatermarkConfig
```

**示例:**

```python
config = WatermarkConfig.from_yaml("config/my_config.yaml")
processor = InvisibleWatermarkProcessor(config=config)
```

**from_dict()**

从字典创建配置。

```python
@classmethod
from_dict(cls, config_dict) -> WatermarkConfig
```

**示例:**

```python
config_dict = {
    'encoding': {'type': 'bch', 'n': 127, 'k': 64},
    'modulation': {'strength': 12}
}
config = WatermarkConfig.from_dict(config_dict)
```

#### 实例方法

**to_yaml()**

保存配置到YAML文件。

```python
to_yaml(yaml_path)
```

**示例:**

```python
config = WatermarkConfig(modulation_strength=15)
config.to_yaml("output/my_config.yaml")
```

**to_dict()**

转换为字典格式。

```python
to_dict() -> dict
```

**validate()**

验证配置的有效性。

```python
validate() -> bool
```

如果配置无效，会抛出 `ValueError`。

## 配置文件格式

YAML配置文件示例:

```yaml
invisible_watermark:
  enabled: true
  
  encoding:
    type: "bch"
    n: 127
    k: 64
    scramble_seed: 12345
  
  macro_block:
    size: 32
    header_pattern: "1110010"
  
  modulation:
    strength: 10
    color_space: "yuv"
  
  extraction:
    min_confidence: 0.3
    max_rotation: 45.0
    max_scale: 2.0
    enable_180_retry: true
  
  performance:
    enable_gpu: false
    num_threads: 4
    enable_cache: true
  
  visualization:
    enabled: false
    output_dir: "./debug_output"
    save_fft_spectrum: true
    save_block_map: true
```

## 使用场景

### 场景1: 基本嵌入和提取

```python
processor = InvisibleWatermarkProcessor()

# 嵌入
watermark = "1010101010101010101010101010101010101010101010101010101010101010"
embed_result = processor.embed_watermark("input.jpg", watermark, "output.png")

# 提取
extract_result = processor.extract_watermark("output.png")

# 验证
if extract_result.success and extract_result.watermark_data == watermark:
    print("水印验证成功!")
```

### 场景2: 批量处理

```python
processor = InvisibleWatermarkProcessor()

images = [
    ("image1.jpg", "watermark1"),
    ("image2.jpg", "watermark2"),
    ("image3.jpg", "watermark3")
]

for input_path, watermark in images:
    output_path = f"output/{Path(input_path).stem}_watermarked.png"
    result = processor.embed_watermark(input_path, watermark, output_path)
    print(f"{input_path}: {'成功' if result.success else '失败'}")
```

### 场景3: 自定义配置

```python
# 创建高强度配置（更鲁棒但可能更可见）
config = WatermarkConfig(
    modulation_strength=15,
    min_confidence=0.4,
    enable_180_retry=True
)

processor = InvisibleWatermarkProcessor(config=config)
```

### 场景4: 攻击测试

```python
import cv2

processor = InvisibleWatermarkProcessor()

# 嵌入水印
watermark = "1010" * 16
processor.embed_watermark("input.jpg", watermark, "watermarked.png")

# 模拟旋转攻击
image = cv2.imread("watermarked.png")
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("rotated.png", rotated)

# 尝试提取
result = processor.extract_watermark("rotated.png")
print(f"旋转后提取: {'成功' if result.success else '失败'}")
print(f"置信度: {result.confidence:.3f}")
```

## 性能指标

### 图像质量

- **PSNR**: 通常 > 40dB（高质量）
- **SSIM**: 通常 > 0.95（高相似度）

### 处理速度

- **嵌入**: 1920×1080图像约2-5秒
- **提取**: 1920×1080图像约5-10秒

### 鲁棒性

- **旋转**: ±45°
- **缩放**: 50%-200%
- **裁剪**: 30%
- **JPEG压缩**: 质量 > 50
- **噪声**: 高斯噪声 σ ≤ 15

## 注意事项

1. **图像尺寸**: 建议至少64×64像素，最好≥512×512
2. **水印格式**: 必须是64位二进制字符串
3. **输出格式**: 建议使用PNG格式保存以避免压缩损失
4. **置信度**: 提取置信度 < 0.3 时结果可能不可靠
5. **180度模糊**: 系统会自动尝试180度旋转重试

## 错误处理

```python
try:
    result = processor.embed_watermark("input.jpg", watermark, "output.png")
    if not result.success:
        print(f"嵌入失败: {result.error_message}")
except ValueError as e:
    print(f"参数错误: {e}")
except FileNotFoundError as e:
    print(f"文件不存在: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 调试和可视化

启用可视化输出:

```python
config = WatermarkConfig(
    visualization_enabled=True,
    visualization_output_dir="./debug"
)

processor = InvisibleWatermarkProcessor(config=config)

result = processor.extract_watermark("watermarked.png", visualize=True)

if result.fft_spectrum is not None:
    cv2.imwrite("debug/fft_spectrum.png", result.fft_spectrum)
```

## 常见问题

**Q: 为什么提取失败?**

A: 可能的原因:
- 图像经过了过强的攻击（旋转>45°、缩放<50%等）
- 图像质量太差（噪声过大、模糊严重）
- 水印未正确嵌入
- 配置参数不匹配

**Q: 如何提高鲁棒性?**

A: 
- 增加调制强度（但会降低不可见性）
- 降低最小置信度阈值
- 启用180度旋转重试
- 使用更大的图像

**Q: 如何提高不可见性?**

A:
- 降低调制强度（但会降低鲁棒性）
- 在YUV色彩空间的Y通道操作
- 避免在边缘和纹理区域嵌入

## 技术支持

如有问题，请查看:
- 设计文档: `docs/design.md`
- 需求文档: `docs/requirements.md`
- 测试用例: `tests/engines/image/test_invisible_watermark.py`
