"""
加扰器单元测试

测试Scrambler类的加扰、解扰和PN序列生成功能。
"""

import pytest
import numpy as np
from engines.image.encoding.scrambler import Scrambler


class TestScramblerInit:
    """测试Scrambler初始化"""
    
    def test_default_init(self):
        """测试默认初始化"""
        scrambler = Scrambler()
        assert scrambler.seed == 12345
        assert scrambler.length == 128
        assert scrambler._pn_sequence is not None
        assert len(scrambler._pn_sequence) == 128
    
    def test_custom_init(self):
        """测试自定义参数初始化"""
        scrambler = Scrambler(seed=54321, length=256)
        assert scrambler.seed == 54321
        assert scrambler.length == 256
        assert len(scrambler._pn_sequence) == 256
    
    def test_invalid_seed(self):
        """测试无效种子"""
        with pytest.raises(ValueError, match="种子必须为正整数"):
            Scrambler(seed=0)
        
        with pytest.raises(ValueError, match="种子必须为正整数"):
            Scrambler(seed=-1)
    
    def test_invalid_length(self):
        """测试无效长度"""
        with pytest.raises(ValueError, match="序列长度必须为正整数"):
            Scrambler(length=0)
        
        with pytest.raises(ValueError, match="序列长度必须为正整数"):
            Scrambler(length=-10)


class TestScramblerPNSequence:
    """测试PN序列生成"""
    
    def test_pn_sequence_length(self):
        """测试PN序列长度"""
        for length in [64, 128, 256, 512]:
            scrambler = Scrambler(length=length)
            pn_seq = scrambler.get_pn_sequence()
            assert len(pn_seq) == length
    
    def test_pn_sequence_binary(self):
        """测试PN序列只包含0和1"""
        scrambler = Scrambler()
        pn_seq = scrambler.get_pn_sequence()
        assert all(bit in [0, 1] for bit in pn_seq)
    
    def test_pn_sequence_deterministic(self):
        """测试相同种子生成相同序列"""
        scrambler1 = Scrambler(seed=12345)
        scrambler2 = Scrambler(seed=12345)
        
        pn_seq1 = scrambler1.get_pn_sequence()
        pn_seq2 = scrambler2.get_pn_sequence()
        
        assert np.array_equal(pn_seq1, pn_seq2)
    
    def test_different_seeds_different_sequences(self):
        """测试不同种子生成不同序列"""
        scrambler1 = Scrambler(seed=12345)
        scrambler2 = Scrambler(seed=54321)
        
        pn_seq1 = scrambler1.get_pn_sequence()
        pn_seq2 = scrambler2.get_pn_sequence()
        
        # 不同种子应该生成不同序列
        assert not np.array_equal(pn_seq1, pn_seq2)
    
    def test_pn_sequence_distribution(self):
        """测试PN序列的分布均匀性"""
        scrambler = Scrambler(length=1024)
        pn_seq = scrambler.get_pn_sequence()
        
        # 统计0和1的数量
        ones_count = np.sum(pn_seq)
        zeros_count = len(pn_seq) - ones_count
        
        # 期望0和1的比例接近1:1（允许20%偏差）
        ratio = ones_count / zeros_count
        assert 0.8 <= ratio <= 1.25
    
    def test_pn_sequence_not_all_same(self):
        """测试PN序列不是全0或全1"""
        scrambler = Scrambler()
        pn_seq = scrambler.get_pn_sequence()
        
        # 不应该全是0
        assert not np.all(pn_seq == 0)
        # 不应该全是1
        assert not np.all(pn_seq == 1)


class TestScramblerScramble:
    """测试加扰功能"""
    
    def setup_method(self):
        """每个测试前初始化加扰器"""
        self.scrambler = Scrambler()
    
    def test_scramble_basic(self):
        """测试基本加扰功能"""
        data = np.array([0, 1] * 64, dtype=np.uint8)
        scrambled = self.scrambler.scramble(data)
        
        assert isinstance(scrambled, np.ndarray)
        assert scrambled.dtype == np.uint8
        assert len(scrambled) == 128
        assert all(bit in [0, 1] for bit in scrambled)
    
    def test_scramble_all_zeros(self):
        """测试加扰全0数据"""
        data = np.zeros(128, dtype=np.uint8)
        scrambled = self.scrambler.scramble(data)
        
        # 加扰后应该等于PN序列
        pn_seq = self.scrambler.get_pn_sequence()
        assert np.array_equal(scrambled, pn_seq)
    
    def test_scramble_all_ones(self):
        """测试加扰全1数据"""
        data = np.ones(128, dtype=np.uint8)
        scrambled = self.scrambler.scramble(data)
        
        # 加扰后应该等于PN序列的反码
        pn_seq = self.scrambler.get_pn_sequence()
        expected = 1 - pn_seq
        assert np.array_equal(scrambled, expected)
    
    def test_scramble_random_data(self):
        """测试加扰随机数据"""
        np.random.seed(42)
        for _ in range(10):
            data = np.random.randint(0, 2, 128, dtype=np.uint8)
            scrambled = self.scrambler.scramble(data)
            
            assert len(scrambled) == 128
            assert all(bit in [0, 1] for bit in scrambled)
    
    def test_scramble_invalid_length(self):
        """测试无效长度的输入"""
        data = np.array([0, 1] * 32, dtype=np.uint8)  # 64位，不是128位
        with pytest.raises(ValueError, match="数据长度必须为128位"):
            self.scrambler.scramble(data)
    
    def test_scramble_invalid_type(self):
        """测试无效类型的输入"""
        with pytest.raises(ValueError, match="输入必须是numpy数组"):
            self.scrambler.scramble([0, 1] * 64)
    
    def test_scramble_invalid_values(self):
        """测试包含非法值的输入"""
        data = np.array([0, 1, 2] * 42 + [0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="数据必须只包含0和1"):
            self.scrambler.scramble(data)
    
    def test_scramble_changes_data(self):
        """测试加扰确实改变了数据"""
        # 使用非全0或全1的数据
        data = np.array([0, 1, 0, 0] * 32, dtype=np.uint8)
        scrambled = self.scrambler.scramble(data)
        
        # 加扰后的数据应该与原数据不同（高概率）
        # 除非PN序列恰好与数据相同（概率极低）
        assert not np.array_equal(data, scrambled)


class TestScramblerDescramble:
    """测试解扰功能"""
    
    def setup_method(self):
        """每个测试前初始化加扰器"""
        self.scrambler = Scrambler()
    
    def test_descramble_basic(self):
        """测试基本解扰功能"""
        data = np.array([0, 1] * 64, dtype=np.uint8)
        scrambled = self.scrambler.scramble(data)
        descrambled = self.scrambler.descramble(scrambled)
        
        assert isinstance(descrambled, np.ndarray)
        assert descrambled.dtype == np.uint8
        assert len(descrambled) == 128
    
    def test_descramble_all_zeros(self):
        """测试解扰全0数据"""
        data = np.zeros(128, dtype=np.uint8)
        scrambled = self.scrambler.scramble(data)
        descrambled = self.scrambler.descramble(scrambled)
        
        assert np.array_equal(descrambled, data)
    
    def test_descramble_all_ones(self):
        """测试解扰全1数据"""
        data = np.ones(128, dtype=np.uint8)
        scrambled = self.scrambler.scramble(data)
        descrambled = self.scrambler.descramble(scrambled)
        
        assert np.array_equal(descrambled, data)
    
    def test_descramble_random_data(self):
        """测试解扰随机数据"""
        np.random.seed(42)
        for _ in range(10):
            data = np.random.randint(0, 2, 128, dtype=np.uint8)
            scrambled = self.scrambler.scramble(data)
            descrambled = self.scrambler.descramble(scrambled)
            
            assert np.array_equal(descrambled, data)


class TestScramblerReversibility:
    """测试加扰/解扰的可逆性"""
    
    def test_scramble_descramble_cycle(self):
        """测试完整的加扰-解扰循环"""
        scrambler = Scrambler()
        
        test_cases = [
            np.zeros(128, dtype=np.uint8),
            np.ones(128, dtype=np.uint8),
            np.array([0, 1] * 64, dtype=np.uint8),
            np.array([1, 0] * 64, dtype=np.uint8),
            np.array([0, 0, 1, 1] * 32, dtype=np.uint8),
            np.array([1, 1, 0, 0] * 32, dtype=np.uint8),
        ]
        
        for original in test_cases:
            scrambled = scrambler.scramble(original)
            descrambled = scrambler.descramble(scrambled)
            
            assert np.array_equal(descrambled, original)
    
    def test_double_scramble_equals_original(self):
        """测试两次加扰等于原始数据（异或的性质）"""
        scrambler = Scrambler()
        data = np.array([0, 1, 0, 0] * 32, dtype=np.uint8)
        
        scrambled_once = scrambler.scramble(data)
        scrambled_twice = scrambler.scramble(scrambled_once)
        
        # 两次异或应该恢复原始数据
        assert np.array_equal(scrambled_twice, data)
    
    def test_reversibility_with_different_seeds(self):
        """测试不同种子的可逆性"""
        seeds = [12345, 54321, 99999, 11111]
        
        for seed in seeds:
            scrambler = Scrambler(seed=seed)
            data = np.random.randint(0, 2, 128, dtype=np.uint8)
            
            scrambled = scrambler.scramble(data)
            descrambled = scrambler.descramble(scrambled)
            
            assert np.array_equal(descrambled, data)
    
    def test_multiple_cycles(self):
        """测试多次加扰-解扰循环"""
        scrambler = Scrambler()
        data = np.random.randint(0, 2, 128, dtype=np.uint8)
        
        current = data.copy()
        for _ in range(10):
            current = scrambler.scramble(current)
            current = scrambler.descramble(current)
        
        # 多次循环后应该仍然等于原始数据
        assert np.array_equal(current, data)


class TestScramblerDataDistribution:
    """测试加扰后数据分布的均匀性"""
    
    def test_scramble_improves_distribution_all_zeros(self):
        """测试加扰改善全0数据的分布"""
        scrambler = Scrambler(length=1024)
        data = np.zeros(1024, dtype=np.uint8)
        scrambled = scrambler.scramble(data)
        
        # 加扰后应该有接近50%的1
        ones_ratio = np.sum(scrambled) / len(scrambled)
        assert 0.4 <= ones_ratio <= 0.6
    
    def test_scramble_improves_distribution_all_ones(self):
        """测试加扰改善全1数据的分布"""
        scrambler = Scrambler(length=1024)
        data = np.ones(1024, dtype=np.uint8)
        scrambled = scrambler.scramble(data)
        
        # 加扰后应该有接近50%的1
        ones_ratio = np.sum(scrambled) / len(scrambled)
        assert 0.4 <= ones_ratio <= 0.6
    
    def test_scramble_maintains_randomness(self):
        """测试加扰保持随机数据的随机性"""
        np.random.seed(42)
        scrambler = Scrambler(length=1024)
        data = np.random.randint(0, 2, 1024, dtype=np.uint8)
        scrambled = scrambler.scramble(data)
        
        # 加扰后仍应该有接近50%的1
        ones_ratio = np.sum(scrambled) / len(scrambled)
        assert 0.4 <= ones_ratio <= 0.6
    
    def test_scramble_breaks_patterns(self):
        """测试加扰打破重复模式"""
        scrambler = Scrambler(length=256)
        # 创建重复模式
        data = np.array([0, 1, 0, 1] * 64, dtype=np.uint8)
        scrambled = scrambler.scramble(data)
        
        # 加扰后不应该保持原有的简单模式
        # 检查连续相同bit的最大长度
        max_run = 1
        current_run = 1
        for i in range(1, len(scrambled)):
            if scrambled[i] == scrambled[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        # 最大连续长度不应该太长（说明打破了模式）
        assert max_run < 20  # 对于256位，连续20位相同的概率很低


class TestScramblerEdgeCases:
    """测试边界条件"""
    
    def test_different_lengths(self):
        """测试不同长度的加扰器"""
        lengths = [32, 64, 128, 256, 512, 1024]
        
        for length in lengths:
            scrambler = Scrambler(length=length)
            data = np.random.randint(0, 2, length, dtype=np.uint8)
            
            scrambled = scrambler.scramble(data)
            descrambled = scrambler.descramble(scrambled)
            
            assert len(scrambled) == length
            assert np.array_equal(descrambled, data)
    
    def test_seed_zero_handling(self):
        """测试种子为0的处理（应该被拒绝）"""
        with pytest.raises(ValueError):
            Scrambler(seed=0)
    
    def test_large_seed(self):
        """测试大种子值"""
        scrambler = Scrambler(seed=999999999)
        data = np.random.randint(0, 2, 128, dtype=np.uint8)
        
        scrambled = scrambler.scramble(data)
        descrambled = scrambler.descramble(scrambled)
        
        assert np.array_equal(descrambled, data)
    
    def test_get_pn_sequence_returns_copy(self):
        """测试get_pn_sequence返回副本而非引用"""
        scrambler = Scrambler()
        pn_seq1 = scrambler.get_pn_sequence()
        pn_seq2 = scrambler.get_pn_sequence()
        
        # 修改一个不应该影响另一个
        pn_seq1[0] = 1 - pn_seq1[0]
        assert not np.array_equal(pn_seq1, pn_seq2)
        
        # 也不应该影响内部序列
        pn_seq3 = scrambler.get_pn_sequence()
        assert np.array_equal(pn_seq2, pn_seq3)


class TestScramblerIntegration:
    """集成测试"""
    
    def test_with_ecc_encoder(self):
        """测试与ECC编码器的集成"""
        try:
            from engines.image.encoding.ecc_encoder import ECCEncoder
            
            # 创建编码器和加扰器
            encoder = ECCEncoder()
            scrambler = Scrambler(length=128)
            
            # 原始数据
            original = "1010101010101010101010101010101010101010101010101010101010101010"
            
            # 编码
            encoded = encoder.encode(original)
            
            # 加扰
            scrambled = scrambler.scramble(encoded)
            
            # 解扰
            descrambled = scrambler.descramble(scrambled)
            
            # 解码
            decoded, success = encoder.decode(descrambled)
            
            assert success is True
            assert decoded == original
            
        except ImportError:
            pytest.skip("ECCEncoder not available")
    
    def test_deterministic_behavior(self):
        """测试确定性行为"""
        seed = 12345
        data = np.array([0, 1, 0, 0] * 32, dtype=np.uint8)
        
        # 多次创建相同种子的加扰器，结果应该相同
        results = []
        for _ in range(5):
            scrambler = Scrambler(seed=seed)
            scrambled = scrambler.scramble(data)
            results.append(scrambled)
        
        # 所有结果应该相同
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
