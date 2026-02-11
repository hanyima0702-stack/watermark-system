"""
多数投票器单元测试
"""

import pytest
import numpy as np
from engines.image.extraction.majority_voter import MajorityVoter


class TestMajorityVoter:
    """测试多数投票器"""
    
    def test_init(self):
        """测试初始化"""
        voter = MajorityVoter(min_confidence=0.5)
        assert voter.min_confidence == 0.5
        
        # 测试默认值
        voter_default = MajorityVoter()
        assert voter_default.min_confidence == 0.3
    
    def test_calculate_bit_confidence_simple(self):
        """测试简单的bit置信度计算"""
        voter = MajorityVoter(min_confidence=0.3)
        
        # 测试全部投1
        bit_values = [1, 1, 1]
        confidences = [0.8, 0.9, 0.7]
        final_bit, confidence = voter.calculate_bit_confidence(bit_values, confidences)
        
        assert final_bit == 1
        assert confidence == 1.0  # 所有票都投给1
    
    def test_calculate_bit_confidence_majority(self):
        """测试多数投票"""
        voter = MajorityVoter(min_confidence=0.3)
        
        # 3票投1，2票投0
        bit_values = [1, 1, 1, 0, 0]
        confidences = [0.8, 0.7, 0.6, 0.5, 0.4]
        final_bit, confidence = voter.calculate_bit_confidence(bit_values, confidences)
        
        assert final_bit == 1
        # 置信度 = vote_1 / (vote_1 + vote_0) = (0.8+0.7+0.6) / (0.8+0.7+0.6+0.5+0.4)
        expected_confidence = (0.8 + 0.7 + 0.6) / (0.8 + 0.7 + 0.6 + 0.5 + 0.4)
        assert abs(confidence - expected_confidence) < 1e-6
    
    def test_calculate_bit_confidence_weighted(self):
        """测试加权投票机制"""
        voter = MajorityVoter(min_confidence=0.3)
        
        # 虽然投0的数量多，但投1的置信度高
        bit_values = [1, 0, 0, 0]
        confidences = [0.9, 0.4, 0.4, 0.4]
        final_bit, confidence = voter.calculate_bit_confidence(bit_values, confidences)
        
        # vote_0 = 0.4 + 0.4 + 0.4 = 1.2
        # vote_1 = 0.9
        # 0应该获胜
        assert final_bit == 0
    
    def test_calculate_bit_confidence_min_threshold(self):
        """测试最小置信度阈值"""
        voter = MajorityVoter(min_confidence=0.5)
        
        # 有些置信度低于阈值
        bit_values = [1, 1, 0, 0]
        confidences = [0.8, 0.6, 0.3, 0.2]  # 后两个低于0.5
        final_bit, confidence = voter.calculate_bit_confidence(bit_values, confidences)
        
        # 只有前两个参与投票，都是1
        assert final_bit == 1
        assert confidence == 1.0
    
    def test_calculate_bit_confidence_all_below_threshold(self):
        """测试所有数据都低于阈值的情况"""
        voter = MajorityVoter(min_confidence=0.8)
        
        bit_values = [1, 0, 1]
        confidences = [0.5, 0.6, 0.4]  # 都低于0.8
        final_bit, confidence = voter.calculate_bit_confidence(bit_values, confidences)
        
        # 应该使用所有数据
        # vote_1 = 0.5 + 0.4 = 0.9
        # vote_0 = 0.6
        assert final_bit == 1
    
    def test_calculate_bit_confidence_empty_input(self):
        """测试空输入"""
        voter = MajorityVoter()
        
        final_bit, confidence = voter.calculate_bit_confidence([], [])
        assert final_bit == 0
        assert confidence == 0.0
    
    def test_calculate_bit_confidence_mismatched_length(self):
        """测试长度不匹配"""
        voter = MajorityVoter()
        
        with pytest.raises(ValueError, match="长度必须相同"):
            voter.calculate_bit_confidence([1, 0], [0.5])
    
    def test_calculate_bit_confidence_zero_total_votes(self):
        """测试总票数为0的情况"""
        voter = MajorityVoter()
        
        bit_values = []
        confidences = []
        final_bit, confidence = voter.calculate_bit_confidence(bit_values, confidences)
        
        assert final_bit == 0
        assert confidence == 0.0
    
    def test_vote_simple(self):
        """测试简单的投票"""
        voter = MajorityVoter(min_confidence=0.3)
        
        # 3个宏块，每个8位数据
        blocks_data = [
            np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
            np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
            np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
        ]
        blocks_confidence = [
            np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32),
            np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], dtype=np.float32),
            np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32),
        ]
        
        final_data, final_confidences = voter.vote(blocks_data, blocks_confidence)
        
        # 所有宏块一致，应该得到相同的结果
        np.testing.assert_array_equal(final_data, blocks_data[0])
        # 所有位的置信度应该是1.0（完全一致）
        np.testing.assert_array_almost_equal(final_confidences, np.ones(8))
    
    def test_vote_with_disagreement(self):
        """测试有分歧的投票"""
        voter = MajorityVoter(min_confidence=0.3)
        
        # 3个宏块，第一位有分歧
        blocks_data = [
            np.array([1, 0, 1, 0], dtype=np.uint8),
            np.array([1, 0, 1, 0], dtype=np.uint8),
            np.array([0, 0, 1, 0], dtype=np.uint8),  # 第一位不同
        ]
        blocks_confidence = [
            np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32),
            np.array([0.7, 0.7, 0.7, 0.7], dtype=np.float32),
            np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32),
        ]
        
        final_data, final_confidences = voter.vote(blocks_data, blocks_confidence)
        
        # 第一位：2票投1（0.8+0.7），1票投0（0.6），1应该获胜
        assert final_data[0] == 1
        # 其他位应该都是原值
        assert final_data[1] == 0
        assert final_data[2] == 1
        assert final_data[3] == 0
    
    def test_vote_128_bits(self):
        """测试128位数据的投票"""
        voter = MajorityVoter(min_confidence=0.3)
        
        # 创建128位的测试数据
        np.random.seed(42)
        blocks_data = [
            np.random.randint(0, 2, 128, dtype=np.uint8),
            np.random.randint(0, 2, 128, dtype=np.uint8),
            np.random.randint(0, 2, 128, dtype=np.uint8),
        ]
        blocks_confidence = [
            np.random.uniform(0.5, 1.0, 128).astype(np.float32),
            np.random.uniform(0.5, 1.0, 128).astype(np.float32),
            np.random.uniform(0.5, 1.0, 128).astype(np.float32),
        ]
        
        final_data, final_confidences = voter.vote(blocks_data, blocks_confidence)
        
        # 验证输出长度
        assert len(final_data) == 128
        assert len(final_confidences) == 128
        
        # 验证输出类型
        assert final_data.dtype == np.uint8
        assert final_confidences.dtype == np.float32
        
        # 验证所有bit值都是0或1
        assert np.all((final_data == 0) | (final_data == 1))
        
        # 验证置信度在[0, 1]范围内
        assert np.all(final_confidences >= 0)
        assert np.all(final_confidences <= 1)
    
    def test_vote_empty_input(self):
        """测试空输入"""
        voter = MajorityVoter()
        
        with pytest.raises(ValueError, match="不能为空"):
            voter.vote([], [])
    
    def test_vote_mismatched_length(self):
        """测试blocks_data和blocks_confidence长度不匹配"""
        voter = MajorityVoter()
        
        blocks_data = [
            np.array([1, 0, 1, 0], dtype=np.uint8),
        ]
        blocks_confidence = [
            np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32),
            np.array([0.7, 0.7, 0.7, 0.7], dtype=np.float32),
        ]
        
        with pytest.raises(ValueError, match="长度必须相同"):
            voter.vote(blocks_data, blocks_confidence)
    
    def test_vote_inconsistent_data_length(self):
        """测试宏块数据长度不一致"""
        voter = MajorityVoter()
        
        blocks_data = [
            np.array([1, 0, 1, 0], dtype=np.uint8),
            np.array([1, 0, 1], dtype=np.uint8),  # 长度不同
        ]
        blocks_confidence = [
            np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32),
            np.array([0.7, 0.7, 0.7], dtype=np.float32),
        ]
        
        with pytest.raises(ValueError, match="数据长度不一致"):
            voter.vote(blocks_data, blocks_confidence)
    
    def test_vote_inconsistent_confidence_length(self):
        """测试宏块置信度长度不一致"""
        voter = MajorityVoter()
        
        blocks_data = [
            np.array([1, 0, 1, 0], dtype=np.uint8),
            np.array([1, 0, 1, 0], dtype=np.uint8),
        ]
        blocks_confidence = [
            np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32),
            np.array([0.7, 0.7], dtype=np.float32),  # 长度不同
        ]
        
        with pytest.raises(ValueError, match="置信度长度不一致"):
            voter.vote(blocks_data, blocks_confidence)
    
    def test_vote_with_low_confidence_blocks(self):
        """测试包含低置信度宏块的投票"""
        voter = MajorityVoter(min_confidence=0.5)
        
        blocks_data = [
            np.array([1, 1, 1, 1], dtype=np.uint8),
            np.array([0, 0, 0, 0], dtype=np.uint8),
            np.array([1, 1, 1, 1], dtype=np.uint8),
        ]
        blocks_confidence = [
            np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),  # 高置信度
            np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),  # 低置信度，应被过滤
            np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32),  # 高置信度
        ]
        
        final_data, final_confidences = voter.vote(blocks_data, blocks_confidence)
        
        # 第二个宏块应该被过滤，结果应该是1
        np.testing.assert_array_equal(final_data, np.array([1, 1, 1, 1]))
    
    def test_vote_realistic_scenario(self):
        """测试真实场景：多个宏块有部分错误"""
        voter = MajorityVoter(min_confidence=0.3)
        
        # 模拟5个宏块，原始数据应该是 [1, 0, 1, 1, 0, 0, 1, 0]
        # 但有些宏块有错误
        blocks_data = [
            np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8),  # 完全正确
            np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8),  # 完全正确
            np.array([1, 1, 1, 1, 0, 0, 1, 0], dtype=np.uint8),  # 第2位错误
            np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8),  # 第3位错误
            np.array([1, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8),  # 第5位错误
        ]
        blocks_confidence = [
            np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32),
            np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32),
            np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], dtype=np.float32),
            np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        ]
        
        final_data, final_confidences = voter.vote(blocks_data, blocks_confidence)
        
        # 应该恢复出正确的数据（多数投票）
        expected = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(final_data, expected)
        
        # 有错误的位置置信度应该较低
        assert final_confidences[1] < 1.0  # 第2位有1个错误
        assert final_confidences[2] < 1.0  # 第3位有1个错误
        assert final_confidences[4] < 1.0  # 第5位有1个错误
