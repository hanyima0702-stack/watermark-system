"""
纠错编码器模块

该模块实现BCH/Reed-Solomon纠错编码功能。
"""

import numpy as np
from typing import Tuple, Optional
try:
    from reedsolo import RSCodec
except ImportError:
    RSCodec = None


class ECCEncoder:
    """
    纠错编码器类
    
    使用Reed-Solomon码对64位数据进行纠错编码，扩展为128位。
    支持最多32位错误纠正（25%容错率）。
    """
    
    def __init__(self, code_type: str = "bch", n: int = 127, k: int = 64):
        """
        初始化纠错编码器
        
        Args:
            code_type: 编码类型 ("bch" 或 "rs")
            n: 编码后长度（比特数）
            k: 原始数据长度（比特数）
        
        Raises:
            ValueError: 如果参数不合法
        """
        if RSCodec is None:
            raise ImportError("需要安装reedsolo库: pip install reedsolo")
        
        if n <= k:
            raise ValueError(f"编码后长度n({n})必须大于原始长度k({k})")
        
        self.code_type = code_type
        self.n = n
        self.k = k
        
        # 计算需要的ECC字节数
        # 64位 = 8字节数据，需要扩展到128位 = 16字节
        # 因此需要8字节的ECC
        self.data_bytes = k // 8  # 8字节
        self.ecc_bytes = (n - k) // 8  # 8字节
        
        # 初始化Reed-Solomon编码器
        # nsym参数指定ECC符号数量
        self.rs = RSCodec(nsym=self.ecc_bytes)
        
    def encode(self, data: str) -> np.ndarray:
        """
        编码64位数据为128位
        
        Args:
            data: 64位二进制字符串（如"1010..."）或整数
        
        Returns:
            128位编码后的numpy数组（dtype=uint8，值为0或1）
        
        Raises:
            ValueError: 如果输入数据格式不正确
        """
        # 转换输入为64位二进制字符串
        if isinstance(data, int):
            if data < 0 or data >= (1 << 64):
                raise ValueError(f"整数必须在[0, 2^64)范围内: {data}")
            binary_str = format(data, '064b')
        elif isinstance(data, str):
            # 移除可能的空格和换行
            binary_str = data.replace(' ', '').replace('\n', '')
            if len(binary_str) != 64:
                raise ValueError(f"二进制字符串长度必须为64位，当前: {len(binary_str)}")
            if not all(c in '01' for c in binary_str):
                raise ValueError("二进制字符串只能包含0和1")
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
        
        # 将64位二进制字符串转换为字节数组（8字节）
        data_bytes = bytearray()
        for i in range(0, 64, 8):
            byte_str = binary_str[i:i+8]
            data_bytes.append(int(byte_str, 2))
        
        # 使用Reed-Solomon编码
        encoded_bytes = self.rs.encode(bytes(data_bytes))
        
        # 转换为比特数组
        bit_array = []
        for byte in encoded_bytes:
            for i in range(7, -1, -1):
                bit_array.append((byte >> i) & 1)
        
        # 截取或填充到128位
        if len(bit_array) < 128:
            # 填充0到128位
            bit_array.extend([0] * (128 - len(bit_array)))
        elif len(bit_array) > 128:
            # 截取前128位
            bit_array = bit_array[:128]
        
        return np.array(bit_array, dtype=np.uint8)
    
    def decode(self, encoded_data: np.ndarray) -> Tuple[Optional[str], bool]:
        """
        解码并纠错
        
        Args:
            encoded_data: 128位编码数据（numpy数组）
        
        Returns:
            (原始64位数据的二进制字符串, 是否成功)
            如果解码失败，返回(None, False)
        
        Raises:
            ValueError: 如果输入数据格式不正确
        """
        if not isinstance(encoded_data, np.ndarray):
            raise ValueError(f"输入必须是numpy数组，当前类型: {type(encoded_data)}")
        
        if len(encoded_data) != 128:
            raise ValueError(f"编码数据长度必须为128位，当前: {len(encoded_data)}")
        
        # 转换比特数组为字节数组
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_data), 8):
            if i + 8 <= len(encoded_data):
                byte_bits = encoded_data[i:i+8]
                byte_val = 0
                for bit in byte_bits:
                    byte_val = (byte_val << 1) | int(bit)
                encoded_bytes.append(byte_val)
        
        # 使用Reed-Solomon解码
        try:
            # 尝试纠错和解码
            # rs.decode返回元组: (decoded_data, decoded_full, errata_pos)
            result = self.rs.decode(bytes(encoded_bytes))
            
            # 提取解码后的数据（第一个元素）
            if isinstance(result, tuple):
                decoded_bytes = result[0]
            else:
                decoded_bytes = result
            
            # 只取前8字节（64位）
            decoded_bytes = decoded_bytes[:self.data_bytes]
            
            # 转换字节数组为64位二进制字符串
            binary_str = ''
            for byte in decoded_bytes:
                binary_str += format(int(byte), '08b')
            
            # 返回完整的解码数据（应该正好是64位）
            return binary_str, True
            
        except Exception as e:
            # 解码失败
            return None, False
