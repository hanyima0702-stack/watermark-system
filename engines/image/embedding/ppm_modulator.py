"""
PPM (Pulse Position Modulation) 调制器

实现脉冲位置调制，将bit映射到像素对并调整亮度。
"""

import numpy as np
from typing import List, Tuple


class PPMModulator:
    """PPM调制器，用于将bit数据调制到图像像素对中"""
    
    def __init__(self, strength: int = 10):
        """
        初始化PPM调制器
        
        Args:
            strength: 调制强度 (建议8-12)，控制亮度调整幅度
        """
        self.strength = strength
    
    def modulate(self, image: np.ndarray, positions: List[Tuple[int, int, int, int]], 
                 bits: np.ndarray) -> np.ndarray:
        """
        调制水印到图像
        
        调制原理:
        - Bit=1: Pixel(A) += strength, Pixel(B) -= strength
        - Bit=0: Pixel(A) -= strength, Pixel(B) += strength
        
        Args:
            image: 原始图像 (YUV格式的Y通道或灰度图)
            positions: [(x1, y1, x2, y2), ...] 像素对位置列表
            bits: 要嵌入的bit数组
        
        Returns:
            调制后的图像
        
        Raises:
            ValueError: 如果positions和bits长度不匹配
        """
        if len(positions) != len(bits):
            raise ValueError(f"位置数量({len(positions)})与bit数量({len(bits)})不匹配")
        
        # 创建图像副本以避免修改原图
        modulated_image = image.copy().astype(np.float32)
        
        for (x1, y1, x2, y2), bit in zip(positions, bits):
            # 确保坐标在图像范围内
            if not self._is_valid_position(modulated_image.shape, x1, y1, x2, y2):
                continue
            
            if bit == 1:
                # Bit=1: A增加亮度，B减少亮度
                modulated_image[y1, x1] += self.strength
                modulated_image[y2, x2] -= self.strength
            else:
                # Bit=0: A减少亮度，B增加亮度
                modulated_image[y1, x1] -= self.strength
                modulated_image[y2, x2] += self.strength
        
        # 确保像素值在[0, 255]范围内
        modulated_image = np.clip(modulated_image, 0, 255)
        
        return modulated_image.astype(np.uint8)
    
    def demodulate(self, image: np.ndarray, positions: List[Tuple[int, int, int, int]]) \
                   -> Tuple[np.ndarray, np.ndarray]:
        """
        从图像解调水印
        
        解调原理:
        - 如果 Pixel(A) > Pixel(B)，则bit=1
        - 如果 Pixel(A) < Pixel(B)，则bit=0
        - 如果 Pixel(A) = Pixel(B)，则不确定，置信度为0
        
        Args:
            image: 带水印图像 (YUV格式的Y通道或灰度图)
            positions: 像素对位置列表
        
        Returns:
            (bit数组, 置信度数组)
            - bit数组: 解调出的bit值
            - 置信度数组: 每个bit的置信度 [0, 1]
        """
        bits = np.zeros(len(positions), dtype=np.uint8)
        confidences = np.zeros(len(positions), dtype=np.float32)
        
        image_float = image.astype(np.float32)
        
        for i, (x1, y1, x2, y2) in enumerate(positions):
            # 确保坐标在图像范围内
            if not self._is_valid_position(image.shape, x1, y1, x2, y2):
                bits[i] = 0
                confidences[i] = 0.0
                continue
            
            # 读取像素对的亮度值
            pixel_a = image_float[y1, x1]
            pixel_b = image_float[y2, x2]
            
            # 计算差值
            diff = pixel_a - pixel_b
            
            # 解调bit值
            if diff > 0:
                bits[i] = 1
            elif diff < 0:
                bits[i] = 0
            else:
                # 差值为0，不确定
                bits[i] = 0
                confidences[i] = 0.0
                continue
            
            # 计算置信度: |diff| / (2 * strength)
            # 理想情况下，diff应该接近±2*strength
            confidence = min(abs(diff) / (2.0 * self.strength), 1.0)
            confidences[i] = confidence
        
        return bits, confidences
    
    def _is_valid_position(self, shape: Tuple[int, ...], 
                          x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        检查像素对位置是否在图像范围内
        
        Args:
            shape: 图像形状 (height, width) 或 (height, width, channels)
            x1, y1: 第一个像素坐标
            x2, y2: 第二个像素坐标
        
        Returns:
            是否有效
        """
        height, width = shape[:2]
        
        return (0 <= x1 < width and 0 <= y1 < height and
                0 <= x2 < width and 0 <= y2 < height)
