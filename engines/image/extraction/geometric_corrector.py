"""
几何校正器模块

该模块实现仿射变换校正功能，用于校正图像的旋转和缩放变换。
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GeometricCorrector:
    """几何校正器类，用于校正图像的几何变换"""
    
    def __init__(self):
        """初始化几何校正器"""
        logger.info("GeometricCorrector initialized")
    
    def create_transform_matrix(self, rotation: float, scale: float, 
                               center: Tuple[int, int]) -> np.ndarray:
        """
        创建仿射变换矩阵
        
        Args:
            rotation: 旋转角度(度)，正值表示逆时针旋转
            scale: 缩放比例
            center: 旋转中心坐标 (x, y)
        
        Returns:
            2×3变换矩阵
        """
        # 创建逆变换矩阵（用于校正）
        # 如果检测到顺时针旋转θ度，需要逆时针旋转-θ度来校正
        # 如果检测到缩放S倍，需要缩放1/S倍来校正
        inverse_rotation = -rotation
        inverse_scale = 1.0 / scale if scale != 0 else 1.0
        
        # 使用OpenCV创建旋转矩阵
        # getRotationMatrix2D参数: (center, angle, scale)
        # angle: 正值表示逆时针旋转
        # scale: 缩放因子
        transform_matrix = cv2.getRotationMatrix2D(
            center=center,
            angle=inverse_rotation,
            scale=inverse_scale
        )
        
        logger.debug(f"Created transform matrix: rotation={inverse_rotation}°, scale={inverse_scale}")
        return transform_matrix
    
    def correct(self, image: np.ndarray, rotation: float, scale: float,
                center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        校正图像的几何变换
        
        Args:
            image: 待校正图像 (BGR或灰度格式)
            rotation: 检测到的旋转角度(度)
            scale: 检测到的缩放比例
            center: 旋转中心，如果为None则使用图像中心
        
        Returns:
            校正后的图像
        """
        if image is None or image.size == 0:
            logger.error("Invalid input image")
            raise ValueError("Input image is empty or None")
        
        height, width = image.shape[:2]
        
        # 如果未指定中心点，使用图像中心
        if center is None:
            center = (width // 2, height // 2)
        
        logger.info(f"Correcting image: rotation={rotation}°, scale={scale}, center={center}")
        
        # 创建变换矩阵
        transform_matrix = self.create_transform_matrix(rotation, scale, center)
        
        # 应用仿射变换
        # 使用双线性插值进行变换
        corrected_image = cv2.warpAffine(
            src=image,
            M=transform_matrix,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        logger.info("Image correction completed")
        return corrected_image
