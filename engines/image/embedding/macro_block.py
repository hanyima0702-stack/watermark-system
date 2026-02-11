"""
宏块生成器模块

该模块实现32×32宏块结构设计和同步信号生成。
"""

import numpy as np
from typing import Dict, List, Tuple


class MacroBlockGenerator:
    """
    宏块生成器
    
    负责设计32×32宏块结构，包含同步信号、Header和Payload区域。
    
    宏块结构:
    - 同步信号区: 4个角点 (用于FFT检测)
    - Header区: 8位非对称标志 (1110010)
    - Payload区: 128位水印数据
    """
    
    BLOCK_SIZE = 32
    HEADER_PATTERN = "1110010"  # 非对称Header，用于方向判断
    
    def __init__(self):
        """初始化宏块生成器"""
        self._layout = None
    
    def generate_block_layout(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        生成宏块布局
        
        定义sync、header、payload在32×32宏块中的位置。
        使用PPM调制，每个bit需要2个像素位置(A, B)。
        
        Returns:
            包含sync、header、payload位置的字典
            - sync: [(x, y), ...] 同步信号位置
            - header: [(x1, y1, x2, y2), ...] Header的像素对位置
            - payload: [(x1, y1, x2, y2), ...] Payload的像素对位置
        """
        if self._layout is not None:
            return self._layout
        
        layout = {
            'sync': [],
            'header': [],
            'payload': []
        }
        
        # 同步信号位置：4个角点
        # 放置在距离角落2个像素的位置，避免边缘效应
        sync_positions = [
            (2, 2),      # 左上
            (2, 29),     # 右上
            (29, 2),     # 左下
            (29, 29)     # 右下
        ]
        layout['sync'] = sync_positions
        
        # Header区域：第一行，从位置(0, 4)开始
        # 7位Header，每位需要2个像素，共14个像素
        header_pairs = []
        header_len = len(self.HEADER_PATTERN)
        for i in range(header_len):
            x1 = 0
            y1 = 4 + i * 2
            x2 = 0
            y2 = 4 + i * 2 + 1
            header_pairs.append((x1, y1, x2, y2))
        layout['header'] = header_pairs
        
        # Payload区域：128位数据，每位需要2个像素
        # 从第二行开始，按行优先顺序排列
        # 避开同步信号位置
        payload_pairs = []
        positions = []
        
        # 收集所有可用位置（避开同步信号）
        for x in range(self.BLOCK_SIZE):
            for y in range(self.BLOCK_SIZE):
                # 跳过第一行（Header区域）
                if x == 0:
                    continue
                # 跳过同步信号位置
                if (x, y) in sync_positions:
                    continue
                positions.append((x, y))
        
        # 为128位数据分配像素对
        for i in range(128):
            if i * 2 + 1 < len(positions):
                x1, y1 = positions[i * 2]
                x2, y2 = positions[i * 2 + 1]
                payload_pairs.append((x1, y1, x2, y2))
        
        layout['payload'] = payload_pairs
        
        self._layout = layout
        return layout
    
    def create_sync_pattern(self) -> np.ndarray:
        """
        创建同步信号模式
        
        生成32×32的同步信号数组，在4个角点位置标记为1，其他位置为0。
        这个模式将在FFT频域中产生可检测的峰值。
        
        Returns:
            32×32的同步信号数组
        """
        sync_pattern = np.zeros((self.BLOCK_SIZE, self.BLOCK_SIZE), dtype=np.uint8)
        
        # 获取同步信号位置
        layout = self.generate_block_layout()
        sync_positions = layout['sync']
        
        # 在同步信号位置标记为1
        for x, y in sync_positions:
            sync_pattern[x, y] = 1
        
        return sync_pattern
    
    def get_header_pattern(self) -> str:
        """
        获取Header模式
        
        Returns:
            Header模式字符串
        """
        return self.HEADER_PATTERN
    
    def get_block_size(self) -> int:
        """
        获取宏块大小
        
        Returns:
            宏块大小（像素）
        """
        return self.BLOCK_SIZE
