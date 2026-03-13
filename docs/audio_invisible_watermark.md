# 音频暗水印处理器文档

## 概述

音频暗水印处理器实现了两种主要的音频水印技术：
1. **回声隐藏 (Echo Hiding)** - 通过在音频信号中添加延迟回声来嵌入水印
2. **相位编码 (Phase Coding)** - 通过修改频域相位来嵌入水印

这些技术确保水印对人耳不可察觉，同时在一定程度上抵抗常见的音频处理操作。

## 技术原理

### 回声隐藏 (Echo Hiding)

回声隐藏技术利用人耳对短延迟回声的不敏感性来嵌入水印。

**工作原理：**
1. 将音频分成多个段，每段对应一个水印比特
2. 对于比特0，添加延迟为 `delay_0` 的回声
3. 对于比特1，添加延迟为 `delay_1` 的回声
4. 回声强度由 `decay` 和 `strength` 参数控制

**提取过程：**
1. 对每个音频段计算自相关函数
2. 检测 `delay_0` 和 `delay_1` 位置的峰值
3. 根据峰值大小判断嵌入的比特值

**优点：**
- 实现简单
- 对音量调整具有鲁棒性
- 不可察觉性好（当参数设置合理时）

**缺点：**
- 对重采样敏感
- 对时域裁剪敏感
- 容量相对较低

### 相位编码 (Phase Coding)

相位编码技术通过修改音频频谱的相位来嵌入水印，利用人耳对相位变化不敏感的特性。

**工作原理：**
1. 使用短时傅里叶变换(STFT)将音频分帧
2. 对每帧进行FFT变换得到频域表示
3. 在中频区域修改相位：比特1增加相位，比特0减少相位
4. 使用逆FFT和重叠相加重构音频信号

**提取过程：**
1. 对含水印音频进行STFT分帧
2. 分析中频区域的平均相位
3. 根据相位偏移方向判断比特值

**优点：**
- 对压缩有一定鲁棒性
- 不可察觉性好
- 可以嵌入较多数据

**缺点：**
- 对时域操作（如裁剪）敏感
- 计算复杂度较高
- 需要仔细调整参数以平衡质量和鲁棒性

## API 使用指南

### 基本使用

```python
from engines.media.audio_invisible_watermark import (
    AudioInvisibleWatermark,
    AudioInvisibleWatermarkConfig
)

# 创建配置
config = AudioInvisibleWatermarkConfig(
    method='echo_hiding',  # 或 'phase_coding'
    strength=0.5,
    echo_delay=100,
    echo_decay=0.5
)

# 创建水印处理器
watermark = AudioInvisibleWatermark(config)

# 嵌入水印
success = watermark.embed_file(
    input_path='original.wav',
    output_path='watermarked.wav',
    watermark_data='User123'
)

# 提取水印
extracted_data, confidence = watermark.extract_file(
    input_path='watermarked.wav',
    watermark_length=7  # 'User123' 的长度
)

print(f"提取的水印: {extracted_data}")
print(f"置信度: {confidence:.2%}")
```

### 配置参数

#### AudioInvisibleWatermarkConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | str | 'echo_hiding' | 水印方法：'echo_hiding' 或 'phase_coding' |
| `strength` | float | 0.5 | 水印强度 (0.0-1.0)，越高越鲁棒但可能更可察觉 |
| `echo_delay` | int | 100 | 回声延迟（样本数），仅用于回声隐藏 |
| `echo_decay` | float | 0.5 | 回声衰减系数 (0.0-1.0)，仅用于回声隐藏 |
| `sample_rate` | int | 44100 | 采样率 |
| `frame_size` | int | 2048 | FFT帧大小，仅用于相位编码 |
| `hop_size` | int | 512 | 帧跳跃大小，仅用于相位编码 |

### 质量测量

```python
# 测量音频质量
quality = watermark.measure_quality(
    original_path='original.wav',
    watermarked_path='watermarked.wav'
)

print(f"SNR: {quality['snr_db']:.2f} dB")
print(f"PSNR: {quality['psnr_db']:.2f} dB")
print(f"相关系数: {quality['correlation']:.4f}")
print(f"不可察觉: {quality['imperceptible']}")
```

**质量指标说明：**
- **SNR (信噪比)**: > 20 dB 通常认为不可察觉
- **PSNR**: 峰值信噪比，越高越好
- **相关系数**: 接近1表示音频相似度高
- **imperceptible**: SNR > 20 dB 时为 True

## 参数调优建议

### 回声隐藏参数

**高质量、低鲁棒性：**
```python
config = AudioInvisibleWatermarkConfig(
    method='echo_hiding',
    strength=0.3,
    echo_delay=80,
    echo_decay=0.4
)
```

**平衡质量和鲁棒性：**
```python
config = AudioInvisibleWatermarkConfig(
    method='echo_hiding',
    strength=0.5,
    echo_delay=100,
    echo_decay=0.5
)
```

**高鲁棒性、可能可察觉：**
```python
config = AudioInvisibleWatermarkConfig(
    method='echo_hiding',
    strength=0.8,
    echo_delay=120,
    echo_decay=0.6
)
```

### 相位编码参数

**高质量：**
```python
config = AudioInvisibleWatermarkConfig(
    method='phase_coding',
    strength=0.3,
    frame_size=2048,
    hop_size=512
)
```

**平衡：**
```python
config = AudioInvisibleWatermarkConfig(
    method='phase_coding',
    strength=0.5,
    frame_size=2048,
    hop_size=512
)
```

## 鲁棒性特性

### 回声隐藏

| 攻击类型 | 鲁棒性 | 说明 |
|---------|--------|------|
| 音量调整 | ✓✓✓ | 非常鲁棒 |
| 噪声添加 | ✓✓ | 中等鲁棒性 |
| MP3压缩 | ✓ | 低鲁棒性 |
| 重采样 | ✗ | 不鲁棒 |
| 时域裁剪 | ✗ | 不鲁棒 |

### 相位编码

| 攻击类型 | 鲁棒性 | 说明 |
|---------|--------|------|
| 音量调整 | ✓✓✓ | 非常鲁棒 |
| 噪声添加 | ✓✓ | 中等鲁棒性 |
| MP3压缩 | ✓✓ | 中等鲁棒性 |
| 重采样 | ✓ | 低鲁棒性 |
| 时域裁剪 | ✗ | 不鲁棒 |

## 最佳实践

### 1. 选择合适的方法

- **回声隐藏**：适用于需要简单实现和对音量调整鲁棒的场景
- **相位编码**：适用于需要更好压缩鲁棒性的场景

### 2. 音频要求

- **最小长度**：建议至少3秒以上
- **采样率**：44100 Hz 或更高
- **格式**：WAV格式（无损）

### 3. 水印数据

- **长度**：建议8-16个字符
- **内容**：使用ASCII可打印字符
- **格式**：建议包含用户ID和时间戳

### 4. 质量控制

```python
# 嵌入后检查质量
quality = watermark.measure_quality(original_path, watermarked_path)

if quality['snr_db'] < 15:
    print("警告：音频质量可能不够好，考虑降低强度")
elif quality['snr_db'] > 25:
    print("提示：可以适当提高强度以增强鲁棒性")
```

### 5. 提取验证

```python
# 提取后验证置信度
extracted_data, confidence = watermark.extract_file(watermarked_path, watermark_length)

if confidence < 0.5:
    print("警告：提取置信度较低，水印可能已损坏")
elif confidence > 0.7:
    print("提取成功，置信度高")
```

## 限制和注意事项

1. **格式支持**：当前仅支持WAV格式，其他格式需要先转换
2. **立体声处理**：水印仅嵌入在第一个声道
3. **提取准确性**：音频水印的提取准确性受多种因素影响，不能保证100%准确
4. **鲁棒性权衡**：提高鲁棒性通常会降低不可察觉性
5. **计算开销**：相位编码比回声隐藏计算量更大

## 性能考虑

### 处理时间

- **回声隐藏**：约 0.5-1 秒/分钟音频
- **相位编码**：约 2-3 秒/分钟音频

### 内存使用

- 处理5分钟的44.1kHz立体声音频约需要100MB内存

## 故障排除

### 问题：提取的水印完全错误

**可能原因：**
- 音频经过了重采样或格式转换
- 水印强度太低
- 音频长度不足

**解决方案：**
- 使用更高的强度参数
- 确保音频至少3秒以上
- 避免对含水印音频进行格式转换

### 问题：音频质量下降明显

**可能原因：**
- 水印强度过高
- 参数设置不当

**解决方案：**
- 降低 `strength` 参数
- 对于回声隐藏，减小 `echo_decay`
- 对于相位编码，减小 `phase_shift`

### 问题：提取置信度很低

**可能原因：**
- 音频受到了攻击或修改
- 提取长度不正确
- 水印强度不足

**解决方案：**
- 确保提取长度与嵌入时一致
- 使用更高的强度参数
- 检查音频是否经过了处理

## 示例代码

完整的示例代码请参考：`examples/audio_watermark_example.py`

该示例演示了：
- 回声隐藏水印的嵌入和提取
- 相位编码水印的嵌入和提取
- 质量测量
- 鲁棒性测试（噪声、音量调整）

## 技术参考

1. Bender, W., et al. (1996). "Techniques for data hiding." IBM Systems Journal.
2. Cox, I. J., et al. (2007). "Digital Watermarking and Steganography."
3. Cvejic, N., & Seppänen, T. (2004). "Spread spectrum audio watermarking using frequency hopping and attack characterization."

## 未来改进方向

1. 支持更多音频格式（MP3、AAC等）
2. 实现更鲁棒的水印算法（如扩频技术）
3. 添加同步码以提高裁剪鲁棒性
4. 实现自适应强度调整
5. 支持多声道独立水印
