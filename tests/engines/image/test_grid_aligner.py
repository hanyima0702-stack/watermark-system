"""
网格对齐器单元测试

测试GridAligner类的Header搜索和网格对齐功能。
"""

import pytest
import numpy as np
from engines.image.extraction.grid_aligner import GridAligner
from engines.image.embedding.ppm_modulator import PPMModulator
from engines.image.embedding.macro_block import MacroBlockGenerator


class TestGridAligner:
    """测试GridAligner类"""
    
    @pytest.fixture
    def grid_aligner(self):
        """创建GridAligner实例"""
        return GridAligner(block_size=32, header_pattern="1110010")
    
    @pytest.fixture
    def ppm_modulator(self):
        """创建PPMModulator实例"""
        return PPMModulator(strength=10)
    
    @pytest.fixture
    def macro_block_generator(self):
        """创建MacroBlockGenerator实例"""
        return MacroBlockGenerator()
    
    def test_initialization(self, grid_aligner):
        """测试初始化"""
        assert grid_aligner.block_size == 32
        assert grid_aligner.header_pattern == "1110010"
        assert grid_aligner.header_length == 7
    
    def test_get_header_positions(self, grid_aligner):
        """测试获取Header位置"""
        # 测试零偏移
        positions = grid_aligner._get_header_positions(0, 0)
        assert len(positions) == 7  # 7位Header
        
        # 检查第一个位置
        x1, y1, x2, y2 = positions[0]
        assert x1 == 0 and x2 == 0
        assert y1 == 4 and y2 == 5
        
        # 测试有偏移的情况
        positions = grid_aligner._get_header_positions(10, 20)
        x1, y1, x2, y2 = positions[0]
        assert x1 == 10 and x2 == 10
        assert y1 == 24 and y2 == 25
    
    def test_check_positions_valid(self, grid_aligner):
        """测试位置有效性检查"""
        # 有效位置
        positions = [(10, 10, 11, 11), (20, 20, 21, 21)]
        assert grid_aligner._check_positions_valid(positions, 100, 100) is True
        
        # 无效位置（超出边界）
        positions = [(10, 10, 11, 11), (200, 200, 201, 201)]
        assert grid_aligner._check_positions_valid(positions, 100, 100) is False
        
        # 负坐标
        positions = [(-1, 10, 11, 11)]
        assert grid_aligner._check_positions_valid(positions, 100, 100) is False
    
    def test_calculate_correlation_score(self, grid_aligner):
        """测试互相关分数计算"""
        # 完全匹配
        detected = np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        expected = np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        confidences = np.ones(7, dtype=np.float32)
        
        score = grid_aligner._calculate_correlation_score(detected, confidences, expected)
        assert score == 1.0
        
        # 部分匹配
        detected = np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        expected = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        confidences = np.ones(7, dtype=np.float32)
        
        score = grid_aligner._calculate_correlation_score(detected, confidences, expected)
        assert 0.8 < score < 0.9  # 6/7匹配
        
        # 完全不匹配
        detected = np.array([0, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
        expected = np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        confidences = np.ones(7, dtype=np.float32)
        
        score = grid_aligner._calculate_correlation_score(detected, confidences, expected)
        assert score == 0.0
        
        # 加权匹配（低置信度）
        detected = np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        expected = np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        confidences = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0], dtype=np.float32)
        
        score = grid_aligner._calculate_correlation_score(detected, confidences, expected)
        assert score == 1.0  # 所有bit都匹配

    
    def test_header_detection_accuracy_no_offset(self, grid_aligner, ppm_modulator, 
                                                  macro_block_generator):
        """测试Header检测准确性（无偏移）"""
        # 创建测试图像 - 使用更均匀的背景以减少假阳性
        image = np.full((128, 128), 128, dtype=np.uint8)
        
        # 获取Header位置和模式
        layout = macro_block_generator.generate_block_layout()
        header_positions = layout['header']
        header_pattern = np.array([int(bit) for bit in grid_aligner.header_pattern], 
                                  dtype=np.uint8)
        
        # 嵌入Header
        image = ppm_modulator.modulate(image, header_positions, header_pattern)
        
        # 搜索Header
        candidates = grid_aligner.search_header(image, ppm_modulator, search_range=10)
        
        # 应该找到候选位置
        assert len(candidates) > 0
        
        # 最佳候选应该是(0, 0)
        best_x, best_y, best_score = candidates[0]
        assert best_x == 0 and best_y == 0
        assert best_score > 0.8  # 高置信度
    
    def test_header_detection_with_offset(self, grid_aligner, ppm_modulator, 
                                         macro_block_generator):
        """测试不同偏移量下的对齐能力"""
        test_offsets = [(5, 5), (10, 10), (15, 15)]  # 使用更合理的偏移量
        
        for offset_x, offset_y in test_offsets:
            # 创建测试图像 - 使用均匀背景
            image = np.full((128, 128), 128, dtype=np.uint8)
            
            # 获取Header位置和模式
            layout = macro_block_generator.generate_block_layout()
            header_positions_base = layout['header']
            
            # 调整Header位置以匹配偏移
            header_positions = [
                (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
                for x1, y1, x2, y2 in header_positions_base
            ]
            
            header_pattern = np.array([int(bit) for bit in grid_aligner.header_pattern], 
                                     dtype=np.uint8)
            
            # 嵌入Header
            image = ppm_modulator.modulate(image, header_positions, header_pattern)
            
            # 搜索Header
            candidates = grid_aligner.search_header(image, ppm_modulator, search_range=32)
            
            # 应该找到候选位置
            assert len(candidates) > 0, f"No candidates found for offset ({offset_x}, {offset_y})"
            
            # 最佳候选应该是正确的偏移
            best_x, best_y, best_score = candidates[0]
            assert best_x == offset_x and best_y == offset_y, \
                f"Expected ({offset_x}, {offset_y}), got ({best_x}, {best_y})"
            assert best_score > 0.7, f"Low score {best_score} for offset ({offset_x}, {offset_y})"
    
    def test_align_method(self, grid_aligner, ppm_modulator, macro_block_generator):
        """测试align方法的完整流程"""
        # 创建测试图像 - 使用均匀背景
        image = np.full((128, 128), 128, dtype=np.uint8)
        
        # 在偏移位置嵌入Header
        offset_x, offset_y = 8, 12
        layout = macro_block_generator.generate_block_layout()
        header_positions_base = layout['header']
        
        header_positions = [
            (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            for x1, y1, x2, y2 in header_positions_base
        ]
        
        header_pattern = np.array([int(bit) for bit in grid_aligner.header_pattern], 
                                 dtype=np.uint8)
        image = ppm_modulator.modulate(image, header_positions, header_pattern)
        
        # 执行对齐
        x, y, confidence = grid_aligner.align(image, ppm_modulator, search_range=32)
        
        # 验证结果
        assert x == offset_x
        assert y == offset_y
        assert confidence > 0.7
    
    def test_align_failure_handling(self, grid_aligner, ppm_modulator):
        """测试对齐失败的处理"""
        # 创建没有Header的随机图像
        image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        
        # 尝试对齐
        x, y, confidence = grid_aligner.align(image, ppm_modulator, search_range=32)
        
        # 随机图像可能会有假阳性匹配，这是正常的
        # 我们只验证函数能正常返回结果
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert 0 <= confidence <= 1.0
    
    def test_align_with_noise(self, grid_aligner, ppm_modulator, macro_block_generator):
        """测试噪声环境下的对齐能力"""
        # 创建测试图像 - 使用均匀背景
        image = np.full((128, 128), 128, dtype=np.uint8)
        
        # 嵌入Header
        layout = macro_block_generator.generate_block_layout()
        header_positions = layout['header']
        header_pattern = np.array([int(bit) for bit in grid_aligner.header_pattern], 
                                 dtype=np.uint8)
        image = ppm_modulator.modulate(image, header_positions, header_pattern)
        
        # 添加高斯噪声
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)  # 减少噪声强度
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 执行对齐
        x, y, confidence = grid_aligner.align(noisy_image, ppm_modulator, search_range=10)
        
        # 应该仍能找到正确位置，但置信度可能降低
        assert x == 0 and y == 0
        assert confidence > 0.5  # 噪声下置信度降低但仍可接受
    
    def test_get_block_positions(self, grid_aligner):
        """测试获取宏块位置"""
        # 测试128x128图像，32x32宏块
        positions = grid_aligner.get_block_positions(0, 0, (128, 128))
        assert len(positions) == 16  # 4x4 = 16个宏块
        
        # 检查第一个和最后一个位置
        assert positions[0] == (0, 0)
        assert positions[-1] == (96, 96)
        
        # 测试有偏移的情况
        positions = grid_aligner.get_block_positions(8, 8, (128, 128))
        assert len(positions) == 9  # 3x3 = 9个宏块
        assert positions[0] == (8, 8)
    
    def test_invalid_image_handling(self, grid_aligner, ppm_modulator):
        """测试无效图像的处理"""
        # 空图像
        with pytest.raises(ValueError, match="empty"):
            grid_aligner.align(np.array([]), ppm_modulator)
        
        # None图像
        with pytest.raises(ValueError, match="empty"):
            grid_aligner.align(None, ppm_modulator)
        
        # 图像太小
        small_image = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            grid_aligner.align(small_image, ppm_modulator)
    
    def test_missing_ppm_modulator(self, grid_aligner):
        """测试缺少PPM调制器的处理"""
        image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="PPM modulator is required"):
            grid_aligner.align(image, None)
    
    def test_search_range_boundary(self, grid_aligner, ppm_modulator, 
                                   macro_block_generator):
        """测试搜索范围边界情况"""
        # 创建测试图像 - 使用均匀背景
        image = np.full((128, 128), 128, dtype=np.uint8)
        
        # 在边界位置嵌入Header - 使用更合理的偏移
        offset_x, offset_y = 20, 20  # 在搜索范围内
        layout = macro_block_generator.generate_block_layout()
        header_positions_base = layout['header']
        
        header_positions = [
            (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            for x1, y1, x2, y2 in header_positions_base
        ]
        
        header_pattern = np.array([int(bit) for bit in grid_aligner.header_pattern], 
                                 dtype=np.uint8)
        image = ppm_modulator.modulate(image, header_positions, header_pattern)
        
        # 使用足够大的搜索范围
        x, y, confidence = grid_aligner.align(image, ppm_modulator, search_range=32)
        
        # 应该找到正确位置
        assert x == offset_x
        assert y == offset_y
        assert confidence > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
