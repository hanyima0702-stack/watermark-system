"""
加扰器模块

该模块实现基于LFSR的PN序列加扰功能。
"""

import numpy as np
from typing import Optional


class Scrambler:
    """
    加扰器类
    
    使用线性反馈移位寄存器(LFSR)生成伪随机序列(PN序列)，
    通过异或操作对数据进行加扰和解扰。
    """
    
    def __init__(self, seed: int = 12345, length: int = 128):
        """
        初始化加扰器
        
        Args:
            seed: 伪随机序列种子，用于生成LFSR初始状态
            length: PN序列长度（比特数），默认128位
        
        Raises:
            ValueError: 如果参数不合法
        """
        if seed <= 0:
            raise ValueError(f"种子必须为正整数: {seed}")
        
        if length <= 0:
            raise ValueError(f"序列长度必须为正整数: {length}")
        
        self.seed = seed
        self.length = length
        
        # 生成PN序列并缓存
        self._pn_sequence = self._generate_pn_sequence()
    
    def _generate_pn_sequence(self) -> np.ndarray:
        """
        使用LFSR生成PN序列
        
        使用16位LFSR，多项式为x^16 + x^14 + x^13 + x^11 + 1
        抽头位置: [16, 14, 13, 11]
        
        Returns:
            长度为self.length的PN序列（numpy数组，dtype=uint8）
        """
        # 使用种子初始化LFSR状态（16位）
        # 确保至少有一位为1
        lfsr_state = self.seed & 0xFFFF
        if lfsr_state == 0:
            lfsr_state = 1
        
        # LFSR多项式抽头位置（从右往左数，0-indexed）
        # x^16 + x^14 + x^13 + x^11 + 1 对应位置 [15, 13, 12, 10]
        taps = [15, 13, 12, 10]
        
        pn_sequence = []
        
        for _ in range(self.length):
            # 输出当前最低位
            output_bit = lfsr_state & 1
            pn_sequence.append(output_bit)
            
            # 计算反馈位（抽头位置的异或）
            feedback = 0
            for tap in taps:
                feedback ^= (lfsr_state >> tap) & 1
            
            # 右移并将反馈位放到最高位
            lfsr_state = (lfsr_state >> 1) | (feedback << 15)
        
        return np.array(pn_sequence, dtype=np.uint8)
    
    def scramble(self, data: np.ndarray) -> np.ndarray:
        """
        加扰数据
        
        使用PN序列与数据进行异或操作。
        
        Args:
            data: 待加扰的数据（numpy数组，dtype=uint8，值为0或1）
        
        Returns:
            加扰后的数据（numpy数组，dtype=uint8）
        
        Raises:
            ValueError: 如果输入数据格式不正确
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(f"输入必须是numpy数组，当前类型: {type(data)}")
        
        if len(data) != self.length:
            raise ValueError(
                f"数据长度必须为{self.length}位，当前: {len(data)}"
            )
        
        # 确保数据是二进制的
        if not np.all((data == 0) | (data == 1)):
            raise ValueError("数据必须只包含0和1")
        
        # 异或加扰
        scrambled = np.bitwise_xor(data, self._pn_sequence)
        
        return scrambled.astype(np.uint8)
    
    def descramble(self, scrambled_data: np.ndarray) -> np.ndarray:
        """
        解扰数据
        
        使用相同的PN序列与加扰数据进行异或操作，恢复原始数据。
        由于异或操作的可逆性，解扰操作与加扰操作完全相同。
        
        Args:
            scrambled_data: 加扰后的数据（numpy数组，dtype=uint8，值为0或1）
        
        Returns:
            解扰后的原始数据（numpy数组，dtype=uint8）
        
        Raises:
            ValueError: 如果输入数据格式不正确
        """
        # 解扰操作与加扰操作相同（异或的可逆性）
        return self.scramble(scrambled_data)
    
    def get_pn_sequence(self) -> np.ndarray:
        """
        获取当前的PN序列
        
        Returns:
            PN序列的副本（numpy数组）
        """
        return self._pn_sequence.copy()
