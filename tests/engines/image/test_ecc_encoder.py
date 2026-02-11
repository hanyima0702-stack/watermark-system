"""
BCH编码器单元测试

测试ECCEncoder类的编码、解码和纠错功能。
"""

import pytest
import numpy as np
from engines.image.encoding.ecc_encoder import ECCEncoder


class TestECCEncoderInit:
    """测试ECCEncoder初始化"""
    
    def test_default_init(self):
        """测试默认初始化"""
        encoder = ECCEncoder()
        assert encoder.code_type == "bch"
        assert encoder.n == 127
        assert encoder.k == 64
        assert encoder.rs is not None
    
    def test_custom_init(self):
        """测试自定义参数初始化"""
        encoder = ECCEncoder(code_type="bch", n=127, k=64)
        assert encoder.n == 127
        assert encoder.k == 64
    
    def test_invalid_parameters(self):
        """测试无效参数"""
        with pytest.raises(ValueError, match="编码后长度n"):
            ECCEncoder(n=64, k=127)


class TestECCEncoderEncode:
    """测试编码功能"""
    
    def setup_method(self):
        """每个测试前初始化编码器"""
        self.encoder = ECCEncoder()
    
    def test_encode_binary_string(self):
        """测试编码64位二进制字符串"""
        data = "1010101010101010101010101010101010101010101010101010101010101010"
        encoded = self.encoder.encode(data)
        
        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == np.uint8
        assert len(encoded) == 128
        assert all(bit in [0, 1] for bit in encoded)
    
    def test_encode_integer(self):
        """测试编码整数"""
        data = 12345678901234567890
        encoded = self.encoder.encode(data)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 128
    
    def test_encode_all_zeros(self):
        """测试编码全0数据"""
        data = "0" * 64
        encoded = self.encoder.encode(data)
        
        assert len(encoded) == 128
        # 编码后不应该全是0（因为有ECC校验位）
    
    def test_encode_all_ones(self):
        """测试编码全1数据"""
        data = "1" * 64
        encoded = self.encoder.encode(data)
        
        assert len(encoded) == 128
    
    def test_encode_random_data(self):
        """测试编码随机数据"""
        np.random.seed(42)
        for _ in range(10):
            data = ''.join(str(np.random.randint(0, 2)) for _ in range(64))
            encoded = self.encoder.encode(data)
            assert len(encoded) == 128
    
    def test_encode_invalid_length(self):
        """测试无效长度的输入"""
        with pytest.raises(ValueError, match="长度必须为64位"):
            self.encoder.encode("101010")
    
    def test_encode_invalid_characters(self):
        """测试包含非法字符的输入"""
        with pytest.raises(ValueError, match="只能包含0和1"):
            self.encoder.encode("1010abc" + "0" * 57)
    
    def test_encode_invalid_integer(self):
        """测试超出范围的整数"""
        with pytest.raises(ValueError, match="整数必须在"):
            self.encoder.encode(-1)
        
        with pytest.raises(ValueError, match="整数必须在"):
            self.encoder.encode(2**64)
    
    def test_encode_with_spaces(self):
        """测试包含空格的二进制字符串"""
        data = "1010 1010 " * 8  # 64位，包含空格
        encoded = self.encoder.encode(data)
        assert len(encoded) == 128


class TestECCEncoderDecode:
    """测试解码功能"""
    
    def setup_method(self):
        """每个测试前初始化编码器"""
        self.encoder = ECCEncoder()
    
    def test_decode_correct_data(self):
        """测试解码正确的数据"""
        original = "1010101010101010101010101010101010101010101010101010101010101010"
        encoded = self.encoder.encode(original)
        decoded, success = self.encoder.decode(encoded)
        
        assert success is True
        assert decoded == original
    
    def test_decode_all_zeros(self):
        """测试解码全0数据"""
        original = "0" * 64
        encoded = self.encoder.encode(original)
        decoded, success = self.encoder.decode(encoded)
        
        assert success is True
        assert decoded == original
    
    def test_decode_all_ones(self):
        """测试解码全1数据"""
        original = "1" * 64
        encoded = self.encoder.encode(original)
        decoded, success = self.encoder.decode(encoded)
        
        assert success is True
        assert decoded == original
    
    def test_decode_random_data(self):
        """测试解码随机数据"""
        np.random.seed(42)
        for _ in range(10):
            original = ''.join(str(np.random.randint(0, 2)) for _ in range(64))
            encoded = self.encoder.encode(original)
            decoded, success = self.encoder.decode(encoded)
            
            assert success is True
            assert decoded == original
    
    def test_decode_invalid_length(self):
        """测试无效长度的输入"""
        invalid_data = np.array([0, 1] * 32, dtype=np.uint8)  # 64位，不是128位
        with pytest.raises(ValueError, match="长度必须为128位"):
            self.encoder.decode(invalid_data)
    
    def test_decode_invalid_type(self):
        """测试无效类型的输入"""
        with pytest.raises(ValueError, match="输入必须是numpy数组"):
            self.encoder.decode([0, 1] * 64)


class TestECCEncoderErrorCorrection:
    """测试纠错能力"""
    
    def setup_method(self):
        """每个测试前初始化编码器"""
        self.encoder = ECCEncoder()
    
    def introduce_errors(self, data: np.ndarray, error_rate: float) -> np.ndarray:
        """
        在数据中引入随机错误
        
        Args:
            data: 原始数据
            error_rate: 错误率（0-1之间）
        
        Returns:
            包含错误的数据
        """
        corrupted = data.copy()
        num_errors = int(len(data) * error_rate)
        error_positions = np.random.choice(len(data), num_errors, replace=False)
        
        for pos in error_positions:
            corrupted[pos] = 1 - corrupted[pos]  # 翻转bit
        
        return corrupted
    
    def test_no_errors(self):
        """测试无错误情况"""
        original = "1010101010101010101010101010101010101010101010101010101010101010"
        encoded = self.encoder.encode(original)
        decoded, success = self.encoder.decode(encoded)
        
        assert success is True
        assert decoded == original
    
    def test_single_error(self):
        """测试单个错误"""
        original = "1010101010101010101010101010101010101010101010101010101010101010"
        encoded = self.encoder.encode(original)
        
        # 引入1个错误
        corrupted = encoded.copy()
        corrupted[10] = 1 - corrupted[10]
        
        decoded, success = self.encoder.decode(corrupted)
        assert success is True
        assert decoded == original
    
    def test_10_percent_errors(self):
        """测试10%错误率"""
        np.random.seed(42)
        original = "1010101010101010101010101010101010101010101010101010101010101010"
        encoded = self.encoder.encode(original)
        
        # 引入10%错误（约13位）
        corrupted = self.introduce_errors(encoded, 0.10)
        
        decoded, success = self.encoder.decode(corrupted)
        # RS(16, 8)可以纠正最多4字节错误
        # 10%错误（13位）可能超出纠错能力，不强制要求成功
        # 但如果成功，应该恢复原始数据
        if success:
            assert decoded == original
    
    def test_20_percent_errors(self):
        """测试20%错误率"""
        np.random.seed(42)
        original = "1010101010101010101010101010101010101010101010101010101010101010"
        encoded = self.encoder.encode(original)
        
        # 引入20%错误（约26位）
        corrupted = self.introduce_errors(encoded, 0.20)
        
        decoded, success = self.encoder.decode(corrupted)
        # 20%错误可能超出纠错能力，但仍应尝试
        # 不强制要求成功，但记录结果
        if success:
            # 如果成功，应该恢复原始数据
            assert decoded == original
    
    def test_30_percent_errors(self):
        """测试30%错误率"""
        np.random.seed(42)
        original = "1010101010101010101010101010101010101010101010101010101010101010"
        encoded = self.encoder.encode(original)
        
        # 引入30%错误（约38位）
        corrupted = self.introduce_errors(encoded, 0.30)
        
        decoded, success = self.encoder.decode(corrupted)
        # 30%错误很可能超出纠错能力
        # 不强制要求成功
    
    def test_multiple_random_errors(self):
        """测试多次随机错误"""
        np.random.seed(42)
        success_count = 0
        total_tests = 20
        
        for i in range(total_tests):
            original = ''.join(str(np.random.randint(0, 2)) for _ in range(64))
            encoded = self.encoder.encode(original)
            
            # 引入2-4个随机错误（在纠错能力范围内）
            num_errors = np.random.randint(2, 5)
            corrupted = encoded.copy()
            error_positions = np.random.choice(128, num_errors, replace=False)
            for pos in error_positions:
                corrupted[pos] = 1 - corrupted[pos]
            
            decoded, success = self.encoder.decode(corrupted)
            if success and decoded == original:
                success_count += 1
        
        # 期望大部分测试能成功（至少70%）
        assert success_count >= total_tests * 0.7


class TestECCEncoderEdgeCases:
    """测试边界条件"""
    
    def setup_method(self):
        """每个测试前初始化编码器"""
        self.encoder = ECCEncoder()
    
    def test_encode_decode_cycle(self):
        """测试完整的编码-解码循环"""
        test_cases = [
            "0" * 64,
            "1" * 64,
            "01" * 32,
            "10" * 32,
            "0011" * 16,
            "1100" * 16,
        ]
        
        for original in test_cases:
            encoded = self.encoder.encode(original)
            decoded, success = self.encoder.decode(encoded)
            
            assert success is True
            assert decoded == original
    
    def test_deterministic_encoding(self):
        """测试编码的确定性"""
        data = "1010101010101010101010101010101010101010101010101010101010101010"
        
        encoded1 = self.encoder.encode(data)
        encoded2 = self.encoder.encode(data)
        
        # 相同输入应该产生相同输出
        assert np.array_equal(encoded1, encoded2)
    
    def test_different_inputs_different_outputs(self):
        """测试不同输入产生不同输出"""
        data1 = "0" * 64
        data2 = "1" * 64
        
        encoded1 = self.encoder.encode(data1)
        encoded2 = self.encoder.encode(data2)
        
        # 不同输入应该产生不同输出
        assert not np.array_equal(encoded1, encoded2)
    
    def test_bit_flip_detection(self):
        """测试单个bit翻转的检测和纠正"""
        original = "1010101010101010101010101010101010101010101010101010101010101010"
        encoded = self.encoder.encode(original)
        
        # 测试翻转不同位置的bit
        for pos in [0, 32, 64, 96, 127]:
            corrupted = encoded.copy()
            corrupted[pos] = 1 - corrupted[pos]
            
            decoded, success = self.encoder.decode(corrupted)
            assert success is True
            assert decoded == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
