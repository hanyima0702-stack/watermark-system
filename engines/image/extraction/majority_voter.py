"""
多数投票器模块

该模块实现多宏块数据融合功能，通过加权投票机制从多个宏块中提取最可靠的水印数据。
"""

import numpy as np
from typing import List, Tuple


class MajorityVoter:
    """
    多数投票器
    
    从多个宏块中提取的水印数据进行加权投票，得到最可靠的结果。
    """
    
    def __init__(self, min_confidence: float = 0.3):
        """
        初始化投票器
        
        Args:
            min_confidence: 最小置信度阈值，低于此值的数据将被忽略
        """
        self.min_confidence = min_confidence
    
    def calculate_bit_confidence(self, bit_values: List[int], 
                                 confidences: List[float]) -> Tuple[int, float]:
        """
        计算单个bit的投票结果
        
        Args:
            bit_values: 该位的所有值 (0或1)
            confidences: 对应的置信度
        
        Returns:
            (最终bit值, 置信度)
        """
        if not bit_values or not confidences:
            return 0, 0.0
        
        if len(bit_values) != len(confidences):
            raise ValueError("bit_values和confidences长度必须相同")
        
        # 过滤掉低置信度的数据
        valid_indices = [i for i, conf in enumerate(confidences) 
                        if conf >= self.min_confidence]
        
        if not valid_indices:
            # 如果没有满足最小置信度的数据，使用所有数据但返回低置信度
            valid_indices = list(range(len(bit_values)))
        
        # 计算加权投票
        vote_0 = 0.0
        vote_1 = 0.0
        
        for idx in valid_indices:
            if bit_values[idx] == 0:
                vote_0 += confidences[idx]
            elif bit_values[idx] == 1:
                vote_1 += confidences[idx]
        
        total_votes = vote_0 + vote_1
        
        if total_votes == 0:
            return 0, 0.0
        
        # 选择票数多的作为最终值
        if vote_1 > vote_0:
            final_bit = 1
            final_confidence = vote_1 / total_votes
        else:
            final_bit = 0
            final_confidence = vote_0 / total_votes
        
        return final_bit, final_confidence
    
    def vote(self, blocks_data: List[np.ndarray], 
             blocks_confidence: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        对多个宏块数据进行投票
        
        Args:
            blocks_data: 宏块数据列表，每个为128位数组
            blocks_confidence: 对应的置信度列表
        
        Returns:
            (融合后的128位数据, 每位的置信度)
        """
        if not blocks_data or not blocks_confidence:
            raise ValueError("blocks_data和blocks_confidence不能为空")
        
        if len(blocks_data) != len(blocks_confidence):
            raise ValueError("blocks_data和blocks_confidence长度必须相同")
        
        # 验证所有数据长度一致
        data_length = len(blocks_data[0])
        for i, data in enumerate(blocks_data):
            if len(data) != data_length:
                raise ValueError(f"第{i}个宏块数据长度不一致: {len(data)} != {data_length}")
            if len(blocks_confidence[i]) != data_length:
                raise ValueError(f"第{i}个宏块置信度长度不一致")
        
        # 对每个bit位置进行投票
        final_data = np.zeros(data_length, dtype=np.uint8)
        final_confidences = np.zeros(data_length, dtype=np.float32)
        
        for bit_pos in range(data_length):
            # 收集该位置的所有bit值和置信度
            bit_values = [int(block[bit_pos]) for block in blocks_data]
            confidences = [float(conf[bit_pos]) for conf in blocks_confidence]
            
            # 计算投票结果
            final_bit, confidence = self.calculate_bit_confidence(bit_values, confidences)
            
            final_data[bit_pos] = final_bit
            final_confidences[bit_pos] = confidence
        
        return final_data, final_confidences
