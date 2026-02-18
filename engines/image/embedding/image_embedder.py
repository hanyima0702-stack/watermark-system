import numpy as np
import cv2
from .ppm_modulator import DCTModulator

class ImageEmbedder:
    """
    图像嵌入器 (简化版)
    负责将水印数据通过 DCT 调制器平铺到整个图像。
    """
    
    def __init__(self, modulator: DCTModulator):
        self.modulator = modulator

    def embed(self, image: np.ndarray, watermark_data: np.ndarray) -> np.ndarray:
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = image_yuv[:, :, 0].copy()

        # 直接传入 watermark_data，modulator 内部处理平铺
        y_modulated = self.modulator.modulate(y_channel, watermark_data)

        image_yuv[:, :, 0] = y_modulated
        return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    
    def calculate_capacity(self, shape: tuple) -> int:
        """计算图像能容纳多少位数据"""
        h, w = shape
        return (h // 8) * (w // 8)