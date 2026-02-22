# utils/m_sequence.py (新建或添加到现有工具文件)

import numpy as np


class MSequenceGenerator:
    """
    最大长度序列 (m-sequence) 生成器。
    用于生成具有优异自相关特性的正交锚点信号。
    """

    def __init__(self, degree: int = 7, poly: int = 0x83):
        """
        初始化 LFSR.
        Args:
            degree: 寄存器位数，序列长度为 2^degree - 1. 默认 7 位产生长度 127 的序列。
            poly: 反馈多项式 (十六进制). 0x83 (x^7 + x + 1) 是 7 位的常用本原多项式。
        """
        self.degree = degree
        self.poly = poly
        self.length = (1 << degree) - 1
        self.state = 1  # 初始状态不能为 0

    def generate(self) -> np.ndarray:
        """生成一个周期的 m-序列，映射为 bipolar (-1, 1)"""
        seq = np.zeros(self.length, dtype=np.float32)
        state = self.state

        for i in range(self.length):
            # 输出最低位，映射 0 -> -1, 1 -> 1
            output_bit = state & 1
            seq[i] = 1.0 if output_bit else -1.0

            # 计算反馈位
            feedback = 0
            temp_state = state
            for j in range(self.degree):
                if (self.poly >> j) & 1:
                    feedback ^= (temp_state & 1)
                temp_state >>= 1

            # 移位并输入反馈
            state = (state >> 1) | (feedback << (self.degree - 1))

        self.state = state  # 更新状态（虽然对于生成固定锚点不需要）
        return seq

    def generate_2d_orthogonal_pattern(self) -> np.ndarray:
        """
        通过两个 1D m-序列的外积生成 2D 正交图案。
        这种图案具有类似于 delta 函数的二维自相关特性。
        """
        seq_x = self.generate()
        # 为了保证行列的弱相关性，可以使用不同的多项式或简单的循环移位
        seq_y = np.roll(seq_x, self.length // 2)

        # 外积生成二维矩阵 (127x127)
        pattern_2d = np.outer(seq_y, seq_x)
        return pattern_2d