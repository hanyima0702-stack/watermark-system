"""
网格对齐器模块

该模块实现Header搜索和网格对齐功能，用于精确定位32×32宏块的起始位置。
"""

import numpy as np
from typing import Tuple, List, Optional
import logging
from scipy import signal

logger = logging.getLogger(__name__)


class GridAligner:
    """
    网格对齐器
    
    负责在几何校正后的图像中搜索Header标志，并精确定位宏块网格的起始位置。
    """
    
    def __init__(self, block_size: int = 32, header_pattern: str = "1110010"):
        """
        初始化网格对齐器
        
        Args:
            block_size: 宏块大小（像素）
            header_pattern: Header模式字符串
        """
        self.block_size = block_size
        self.header_pattern = header_pattern
        self.header_length = len(header_pattern)
        
        logger.info(f"GridAligner initialized: block_size={block_size}, "
                   f"header_pattern={header_pattern}")
    
    def align(self, image: np.ndarray, ppm_modulator=None, 
              search_range: int = 32) -> Tuple[int, int, float]:
        """
        对齐网格
        
        在图像中搜索Header标志，确定宏块网格的起始位置。
        
        Args:
            image: 校正后的图像 (YUV格式的Y通道或灰度图)
            ppm_modulator: PPM解调器实例，用于读取Header
            search_range: 搜索范围（像素），默认32
        
        Returns:
            (x_offset, y_offset, confidence)
            - x_offset: 网格在x方向的偏移量
            - y_offset: 网格在y方向的偏移量
            - confidence: 对齐置信度 [0, 1]
        
        Raises:
            ValueError: 如果图像太小或ppm_modulator为None
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        if ppm_modulator is None:
            raise ValueError("PPM modulator is required for header detection")
        
        height, width = image.shape[:2]
        
        # 检查图像尺寸
        if height < self.block_size or width < self.block_size:
            logger.error(f"Image too small: {width}x{height}, "
                        f"minimum required: {self.block_size}x{self.block_size}")
            raise ValueError(f"Image too small for block size {self.block_size}")
        
        logger.info(f"Starting grid alignment on image {width}x{height}")
        
        # 搜索Header位置
        candidates = self.search_header(image, ppm_modulator, search_range)
        
        if not candidates:
            logger.warning("No header candidates found")
            return (0, 0, 0.0)
        
        # 选择置信度最高的候选位置
        best_candidate = max(candidates, key=lambda x: x[2])
        x_offset, y_offset, confidence = best_candidate
        
        logger.info(f"Grid aligned at offset ({x_offset}, {y_offset}) "
                   f"with confidence {confidence:.3f}")
        
        return (x_offset, y_offset, confidence)

    
    def search_header(self, image: np.ndarray, ppm_modulator, 
                     search_range: int = 32) -> List[Tuple[int, int, float]]:
        """
        搜索Header位置
        
        使用滑动窗口在指定范围内搜索Header标志，并使用互相关算法计算匹配分数。
        
        Args:
            image: 图像 (YUV格式的Y通道或灰度图)
            ppm_modulator: PPM解调器实例
            search_range: 搜索范围（像素）
        
        Returns:
            候选位置列表 [(x, y, score), ...]
            - x, y: 候选的网格起始位置
            - score: 匹配分数 [0, 1]
        """
        height, width = image.shape[:2]
        candidates = []
        
        # 将Header模式转换为numpy数组
        expected_header = np.array([int(bit) for bit in self.header_pattern], 
                                   dtype=np.uint8)
        
        logger.debug(f"Searching for header in range [0, {search_range})")
        
        # 在搜索范围内滑动窗口
        # 注意：x是行（height方向），y是列（width方向）
        for x_offset in range(min(search_range, height - self.block_size + 1)):
            for y_offset in range(min(search_range, width - self.block_size + 1)):
                # 提取Header区域的像素对位置
                header_positions = self._get_header_positions(x_offset, y_offset)
                
                # 检查所有位置是否在图像范围内
                if not self._check_positions_valid(header_positions, width, height):
                    continue
                
                # 使用PPM解调器读取Header
                try:
                    detected_bits, confidences = ppm_modulator.demodulate(
                        image, header_positions
                    )
                    
                    # 只有当所有bit都有合理的置信度时才计算分数
                    if np.mean(confidences) < 0.1:
                        continue
                    
                    # 计算匹配分数（使用互相关）
                    score = self._calculate_correlation_score(
                        detected_bits, confidences, expected_header
                    )
                    
                    # 如果分数足够高，添加到候选列表
                    if score > 0.3:  # 最小阈值
                        candidates.append((x_offset, y_offset, score))
                        logger.debug(f"Found candidate at ({x_offset}, {y_offset}) "
                                   f"with score {score:.3f}")
                
                except Exception as e:
                    logger.debug(f"Error at offset ({x_offset}, {y_offset}): {e}")
                    continue
        
        # 按分数降序排序
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(candidates)} header candidates")
        return candidates
    
    def _get_header_positions(self, x_offset: int, y_offset: int) \
                             -> List[Tuple[int, int, int, int]]:
        """
        获取Header区域的像素对位置
        
        Header位于宏块的第一行，从位置(0, 4)开始。
        x_offset和y_offset是宏块的起始位置。
        
        Args:
            x_offset: 宏块在x方向的偏移量（行）
            y_offset: 宏块在y方向的偏移量（列）
        
        Returns:
            像素对位置列表 [(x1, y1, x2, y2), ...]
        """
        header_positions = []
        
        for i in range(self.header_length):
            # Header在宏块内的第一行(x=0)，从y=4开始
            # 宏块的起始位置是(x_offset, y_offset)
            # 所以Header的绝对位置是(x_offset + 0, y_offset + 4 + i*2)
            x1 = x_offset
            y1 = y_offset + 4 + i * 2
            x2 = x_offset
            y2 = y_offset + 4 + i * 2 + 1
            
            header_positions.append((x1, y1, x2, y2))
        
        return header_positions
    
    def _check_positions_valid(self, positions: List[Tuple[int, int, int, int]], 
                               width: int, height: int) -> bool:
        """
        检查所有位置是否在图像范围内
        
        Args:
            positions: 像素对位置列表 [(x1, y1, x2, y2), ...]
            width: 图像宽度
            height: 图像高度
        
        Returns:
            是否所有位置都有效
        """
        for x1, y1, x2, y2 in positions:
            if not (0 <= x1 < width and 0 <= y1 < height and
                   0 <= x2 < width and 0 <= y2 < height):
                return False
        return True
    
    def _calculate_correlation_score(self, detected_bits: np.ndarray, 
                                     confidences: np.ndarray,
                                     expected_bits: np.ndarray) -> float:
        """
        计算互相关匹配分数
        
        使用加权互相关算法，考虑每个bit的置信度。
        
        Args:
            detected_bits: 检测到的bit数组
            confidences: 每个bit的置信度数组
            expected_bits: 期望的bit数组（Header模式）
        
        Returns:
            匹配分数 [0, 1]
        """
        if len(detected_bits) != len(expected_bits):
            return 0.0
        
        # 计算匹配的bit数量（加权）
        matches = (detected_bits == expected_bits).astype(np.float32)
        weighted_matches = matches * confidences
        
        # 计算总权重
        total_weight = np.sum(confidences)
        
        if total_weight == 0:
            return 0.0
        
        # 归一化分数
        score = np.sum(weighted_matches) / total_weight
        
        return float(score)
    
    def get_block_positions(self, x_offset: int, y_offset: int, 
                           image_shape: Tuple[int, int]) \
                          -> List[Tuple[int, int]]:
        """
        获取所有宏块的起始位置
        
        Args:
            x_offset: 网格在x方向的偏移量
            y_offset: 网格在y方向的偏移量
            image_shape: 图像形状 (height, width)
        
        Returns:
            宏块起始位置列表 [(x, y), ...]
        """
        height, width = image_shape
        block_positions = []
        
        # 计算可以容纳的宏块数量
        x = x_offset
        while x + self.block_size <= width:
            y = y_offset
            while y + self.block_size <= height:
                block_positions.append((x, y))
                y += self.block_size
            x += self.block_size
        
        logger.debug(f"Generated {len(block_positions)} block positions")
        return block_positions
