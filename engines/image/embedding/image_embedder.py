"""
图像嵌入器模块

该模块实现将水印宏块平铺到整个图像的功能。
"""

import numpy as np
import cv2
from typing import Tuple
from .macro_block import MacroBlockGenerator
from .ppm_modulator import PPMModulator


class ImageEmbedder:
    """
    图像嵌入器
    
    负责将水印数据以宏块形式平铺到整个图像中。
    处理色彩空间转换和宏块布局计算。
    """
    
    def __init__(self, block_generator: MacroBlockGenerator, modulator: PPMModulator):
        """
        初始化图像嵌入器
        
        Args:
            block_generator: 宏块生成器实例
            modulator: PPM调制器实例
        """
        self.block_generator = block_generator
        self.modulator = modulator
        self.block_size = block_generator.get_block_size()
    
    def calculate_block_count(self, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        计算可容纳的宏块数量
        
        Args:
            image_shape: (height, width) 图像尺寸
        
        Returns:
            (行数, 列数) 宏块数量
        """
        height, width = image_shape
        rows = height // self.block_size
        cols = width // self.block_size
        return rows, cols
    
    def embed(self, image: np.ndarray, watermark_data: np.ndarray) -> np.ndarray:
        """
        嵌入水印到图像
        
        将128位水印数据以宏块形式平铺到整个图像中。
        处理BGR到YUV的色彩空间转换，在Y通道进行调制。
        
        Args:
            image: 原始图像 (BGR格式)
            watermark_data: 128位水印数据 (numpy数组)
        
        Returns:
            带水印的图像 (BGR格式)
        
        Raises:
            ValueError: 如果图像尺寸过小或水印数据长度不正确
        """
        # 验证输入
        if len(watermark_data) != 128:
            raise ValueError(f"水印数据必须是128位，当前为{len(watermark_data)}位")
        
        height, width = image.shape[:2]
        if height < self.block_size or width < self.block_size:
            raise ValueError(
                f"图像尺寸({height}×{width})过小，"
                f"至少需要{self.block_size}×{self.block_size}"
            )
        
        # 转换BGR到YUV色彩空间
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = image_yuv[:, :, 0].copy()
        
        # 计算宏块数量
        rows, cols = self.calculate_block_count((height, width))
        
        # 获取宏块布局
        layout = self.block_generator.generate_block_layout()
        header_pattern = self.block_generator.get_header_pattern()
        
        # 准备Header数据（8位）
        header_bits = np.array([int(b) for b in header_pattern], dtype=np.uint8)
        
        # 嵌入到每个宏块
        for row in range(rows):
            for col in range(cols):
                # 计算当前宏块在图像中的起始位置
                start_y = row * self.block_size
                start_x = col * self.block_size
                
                # 提取当前宏块区域
                block = y_channel[start_y:start_y + self.block_size,
                                 start_x:start_x + self.block_size].copy()
                
                # 嵌入Header（使用宏块内的相对坐标）
                block = self.modulator.modulate(block, layout['header'], header_bits)
                
                # 嵌入Payload（使用宏块内的相对坐标）
                block = self.modulator.modulate(block, layout['payload'], watermark_data)
                
                # 将调制后的宏块写回Y通道
                y_channel[start_y:start_y + self.block_size,
                         start_x:start_x + self.block_size] = block
        
        # 更新YUV图像的Y通道
        image_yuv[:, :, 0] = y_channel
        
        # 转换回BGR色彩空间
        watermarked_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
        
        return watermarked_image

