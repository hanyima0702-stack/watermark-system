"""
配置模块

定义水印系统的所有配置参数和结果数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml
import os

from dataclasses import dataclass, field
import yaml


@dataclass
class WatermarkConfig:
    """水印配置 (抗攻击增强版)"""

    # 编码配置
    ecc_type: str = "bch"
    ecc_n: int = 127
    ecc_k: int = 64
    scramble_seed: int = 12345

    # === 增强 1: 调制强度大幅提升 ===
    # 以前是 30，现在提升到 50 甚至更高，低频系数数值大，需要更大的修改量才能抗住干扰
    modulation_strength: int = 80
    block_size: int = 8

    # === 增强 2: 分辨率金字塔搜索步长 ===
    # 提取时，将图片缩放到这些宽度尝试提取
    # 范围覆盖极小图到大图，步长越小越慢但越准
    search_step: int = 8
    min_width: int = 128
    max_width: int = 1024  # 如果你的原图是 4K，这里可以设大一点，比如 2048

    # 性能配置
    enable_gpu: bool = False


# ==========================================
# 结果数据类 (Result Dataclasses)
# ==========================================

@dataclass
class EmbedResult:
    """嵌入结果"""
    success: bool
    watermark_data: str          # 原始水印数据 (64位)
    encoded_data: str            # 编码后的数据 (128位)
    block_count: tuple           # (行数, 列数) - 仅供参考
    processing_time: float       # 处理耗时 (秒)
    image_size: tuple            # 图像尺寸 (height, width)
    error_message: Optional[str] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict) # PSNR, SSIM

@dataclass
class ExtractionResult:
    """提取结果"""
    success: bool
    watermark_data: Optional[str] = None  # 提取出的原始数据

    # 提取置信度 (分数越高越可靠)
    # 对于 DCT 算法，通常 >5.0 表示可靠，>10.0 表示非常可靠
    confidence: float = 0.0

    # 检测到的几何变换参数
    detected_scale: float = 1.0           # 检测到的缩放比例
    grid_offset: tuple = (0, 0)           # 检测到的网格偏移 (dx, dy)

    processing_time: float = 0.0          # 处理耗时
    error_message: Optional[str] = None   # 错误信息