# 水印提取可视化PDF报告生成器

## 概述

`WatermarkReportGenerator` 是一个专门用于生成水印提取结果可视化PDF报告的工具类。它将水印提取的技术数据转换为易于理解和审计的PDF格式报告。

## 功能特性

### 1. 完整的报告结构

报告包含以下几个部分：

- **标题页**：文件基本信息和提取结果汇总
- **详细结果页**：每页的水印提取详情
- **技术细节页**：提取方法和置信度分布统计
- **说明页**：报告使用说明和技术解释

### 2. 可视化展示

- **置信度颜色编码**：
  - 绿色：高置信度 (>50%)
  - 橙色：中置信度 (30-50%)
  - 红色：低置信度 (<30%)

- **多格式数据展示**：
  - 水印文本内容
  - 比特序列预览（前32位）
  - 十六进制表示

### 3. 灵活配置

通过 `ReportConfig` 类可以自定义：
- 页面尺寸和边距
- 字体大小
- 颜色方案
- 间距设置

## 使用方法

### 基本用法

```python
from engines.extraction.report_generator import WatermarkReportGenerator, ExtractionResult
import numpy as np

# 创建报告生成器
generator = WatermarkReportGenerator()

# 准备提取结果
extraction_results = [
    ExtractionResult(
        page_num=1,
        watermark_data="user_admin|2024-01-20 10:30:00|confidential",
        confidence=0.85,
        bits=np.array([1, 0, 1, 1, 0, 0, 1, 0] * 16),
        metadata={
            'method': 'FFT',
            'frequency_band': 'mid',
            'extracted_length': 128
        }
    )
]

# 生成报告
report_path = generator.generate_extraction_report(
    source_file="document.pdf",
    extraction_results=extraction_results,
    output_path="extraction_report.pdf",
    include_technical_details=True
)
```

### 使用字典格式的结果

```python
# 也可以使用字典格式
dict_results = [
    {
        'page_num': 1,
        'watermark_data': "test_watermark",
        'confidence': 0.75,
        'bits': np.array([1, 0, 1, 0] * 8),
        'metadata': {'method': 'FFT'}
    }
]

report_path = generator.generate_extraction_report(
    source_file="document.pdf",
    extraction_results=dict_results,
    output_path="report.pdf"
)
```

### 自定义配置

```python
from engines.extraction.report_generator import ReportConfig

# 创建自定义配置
custom_config = ReportConfig(
    page_width=600,
    page_height=850,
    title_fontsize=20,
    body_fontsize=14,
    color_high_confidence=(0, 1, 0),  # 纯绿色
    margin_left=60
)

# 使用自定义配置创建生成器
generator = WatermarkReportGenerator(custom_config)
```

## 与PDF处理器集成

`PDFProcessor` 类已经集成了报告生成功能：

```python
from engines.document.pdf_processor import PDFProcessor

processor = PDFProcessor()

# 提取水印并生成可视化报告
report_path = processor.extract_invisible_watermark_to_pdf(
    file_path="watermarked_document.pdf",
    output_path="extraction_report.pdf"
)
```

## 报告内容详解

### 1. 标题页

包含以下信息：
- 报告标题（中英文）
- 源文件名称
- 提取时间
- 处理页数
- 文件大小
- 平均置信度
- 成功提取数量
- 提取方法

### 2. 详细结果页

对每一页的提取结果展示：
- 页码
- 水印内容（如果无法解码则显示提示）
- 置信度（带颜色编码）
- 比特序列预览（前32位）
- 十六进制表示

### 3. 技术细节页

包含：
- 提取方法信息
- 频域范围
- 水印长度
- 置信度分布统计（高/中/低）

### 4. 说明页

提供：
- 报告用途说明
- 置信度解释
- 颜色编码说明
- 技术背景
- 使用建议

## API 参考

### WatermarkReportGenerator

#### `__init__(config: Optional[ReportConfig] = None)`

初始化报告生成器。

**参数：**
- `config`: 报告配置对象，如果为None则使用默认配置

#### `generate_extraction_report(...)`

生成水印提取的可视化PDF报告。

**参数：**
- `source_file`: 源文件路径
- `extraction_results`: 提取结果列表
- `output_path`: 输出报告路径
- `include_technical_details`: 是否包含技术细节（默认True）

**返回：**
- 生成的报告文件路径

### ExtractionResult

水印提取结果数据类。

**字段：**
- `page_num`: 页码
- `watermark_data`: 水印数据（字符串或None）
- `confidence`: 置信度（0.0-1.0）
- `bits`: 比特序列（numpy数组）
- `metadata`: 元数据字典

### ReportConfig

报告配置数据类。

**主要字段：**
- `page_width`: 页面宽度（默认595，A4）
- `page_height`: 页面高度（默认842，A4）
- `margin_left/right/top/bottom`: 页边距
- `title_fontsize`: 标题字体大小
- `heading_fontsize`: 标题字体大小
- `body_fontsize`: 正文字体大小
- `small_fontsize`: 小字体大小
- `color_high_confidence`: 高置信度颜色
- `color_medium_confidence`: 中置信度颜色
- `color_low_confidence`: 低置信度颜色

## 测试

运行单元测试：

```bash
pytest tests/engines/extraction/test_report_generator.py -v
```

测试覆盖：
- 报告生成基本功能
- 字典格式结果支持
- 置信度颜色编码
- 比特序列格式化
- 十六进制转换
- 水印数据格式化
- 文件大小格式化
- 技术细节页面
- 多页结果汇总
- 说明页面内容
- 自定义配置
- 空结果处理

## 示例输出

运行演示脚本查看示例输出：

```bash
python demo_pdf_watermark_extraction.py
```

这将生成：
1. 原始PDF文档
2. 含水印的PDF文档
3. 水印提取可视化报告

## 应用场景

1. **审计追踪**：为审计人员提供清晰的水印提取证据
2. **法律证据**：生成可用于法律程序的格式化报告
3. **质量检查**：验证水印嵌入和提取的质量
4. **技术分析**：分析水印算法的性能和鲁棒性
5. **用户报告**：向非技术用户展示水印提取结果

## 注意事项

1. **文件锁定**：在Windows系统上，确保在生成报告后正确关闭PDF文档，避免文件锁定问题
2. **内存使用**：处理大量页面时注意内存使用，建议分批处理
3. **字体支持**：使用PyMuPDF内置字体，确保跨平台兼容性
4. **颜色空间**：使用RGB颜色空间，值范围0.0-1.0

## 未来改进

- [ ] 支持自定义字体
- [ ] 添加图表和可视化
- [ ] 支持多语言报告
- [ ] 添加水印图像预览
- [ ] 支持批量报告生成
- [ ] 添加报告模板系统

## 相关文档

- [PDF处理器文档](pdf_fft_watermark.md)
- [FFT水印算法文档](fft_image_watermark.md)
- [水印系统设计文档](../design.md)
