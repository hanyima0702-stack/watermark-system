"""
宏块生成器单元测试

测试MacroBlockGenerator类的功能，包括：
- 布局生成的正确性
- 同步信号位置
- Header非对称性
"""

import pytest
import numpy as np
from engines.image.embedding.macro_block import MacroBlockGenerator


class TestMacroBlockGenerator:
    """测试宏块生成器"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.generator = MacroBlockGenerator()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.generator.BLOCK_SIZE == 32
        assert self.generator.HEADER_PATTERN == "1110010"
        assert len(self.generator.HEADER_PATTERN) == 7
    
    def test_generate_block_layout_structure(self):
        """测试布局生成的结构正确性"""
        layout = self.generator.generate_block_layout()
        
        # 验证返回的字典包含所有必需的键
        assert 'sync' in layout
        assert 'header' in layout
        assert 'payload' in layout
        
        # 验证数据类型
        assert isinstance(layout['sync'], list)
        assert isinstance(layout['header'], list)
        assert isinstance(layout['payload'], list)
    
    def test_sync_positions(self):
        """测试同步信号位置的正确性"""
        layout = self.generator.generate_block_layout()
        sync_positions = layout['sync']
        
        # 应该有4个同步信号位置（4个角点）
        assert len(sync_positions) == 4
        
        # 验证同步信号位置在角落附近
        expected_positions = [
            (2, 2),      # 左上
            (2, 29),     # 右上
            (29, 2),     # 左下
            (29, 29)     # 右下
        ]
        
        for pos in expected_positions:
            assert pos in sync_positions, f"Expected sync position {pos} not found"
        
        # 验证所有位置都在宏块范围内
        for x, y in sync_positions:
            assert 0 <= x < 32
            assert 0 <= y < 32
    
    def test_header_layout(self):
        """测试Header区域布局"""
        layout = self.generator.generate_block_layout()
        header_pairs = layout['header']
        
        # Header应该有8位（对应HEADER_PATTERN的7位 + 1位用于扩展）
        # 实际上HEADER_PATTERN是7位，但我们可能需要8位对齐
        # 根据设计，Header区域应该有足够的像素对
        assert len(header_pairs) >= 7, "Header should have at least 7 bit positions"
        
        # 验证每个Header位置是4元组(x1, y1, x2, y2)
        for pair in header_pairs:
            assert len(pair) == 4
            x1, y1, x2, y2 = pair
            
            # 验证位置在宏块范围内
            assert 0 <= x1 < 32
            assert 0 <= y1 < 32
            assert 0 <= x2 < 32
            assert 0 <= y2 < 32
            
            # 验证像素对是相邻的
            assert abs(x1 - x2) + abs(y1 - y2) == 1, "Pixel pairs should be adjacent"
    
    def test_payload_layout(self):
        """测试Payload区域布局"""
        layout = self.generator.generate_block_layout()
        payload_pairs = layout['payload']
        
        # Payload应该有128位
        assert len(payload_pairs) == 128, "Payload should have exactly 128 bit positions"
        
        # 验证每个Payload位置是4元组
        for pair in payload_pairs:
            assert len(pair) == 4
            x1, y1, x2, y2 = pair
            
            # 验证位置在宏块范围内
            assert 0 <= x1 < 32
            assert 0 <= y1 < 32
            assert 0 <= x2 < 32
            assert 0 <= y2 < 32
    
    def test_no_position_overlap(self):
        """测试不同区域之间没有位置重叠"""
        layout = self.generator.generate_block_layout()
        
        # 收集所有使用的位置
        used_positions = set()
        
        # 添加同步信号位置
        for pos in layout['sync']:
            assert pos not in used_positions, f"Sync position {pos} is duplicated"
            used_positions.add(pos)
        
        # 添加Header位置
        for x1, y1, x2, y2 in layout['header']:
            assert (x1, y1) not in used_positions, f"Header position ({x1}, {y1}) overlaps"
            assert (x2, y2) not in used_positions, f"Header position ({x2}, {y2}) overlaps"
            used_positions.add((x1, y1))
            used_positions.add((x2, y2))
        
        # 添加Payload位置
        for x1, y1, x2, y2 in layout['payload']:
            # Payload位置不应该与sync或header重叠
            # 但payload内部的像素对可能共享位置（这是正常的）
            pass
    
    def test_create_sync_pattern(self):
        """测试同步信号模式生成"""
        sync_pattern = self.generator.create_sync_pattern()
        
        # 验证形状
        assert sync_pattern.shape == (32, 32)
        
        # 验证数据类型
        assert sync_pattern.dtype == np.uint8
        
        # 验证只有4个位置为1（同步信号位置）
        assert np.sum(sync_pattern) == 4
        
        # 验证同步信号位置
        layout = self.generator.generate_block_layout()
        for x, y in layout['sync']:
            assert sync_pattern[x, y] == 1, f"Sync position ({x}, {y}) should be 1"
        
        # 验证其他位置为0
        for x in range(32):
            for y in range(32):
                if (x, y) not in layout['sync']:
                    assert sync_pattern[x, y] == 0
    
    def test_header_pattern_asymmetry(self):
        """测试Header模式的非对称性"""
        header_pattern = self.generator.get_header_pattern()
        
        # 验证Header模式是非对称的（不是回文）
        assert header_pattern != header_pattern[::-1], \
            "Header pattern should be asymmetric for orientation detection"
        
        # 验证Header模式长度
        assert len(header_pattern) == 7
        
        # 验证Header模式只包含0和1
        for bit in header_pattern:
            assert bit in ['0', '1']
    
    def test_layout_caching(self):
        """测试布局缓存机制"""
        # 第一次调用
        layout1 = self.generator.generate_block_layout()
        
        # 第二次调用应该返回相同的对象（缓存）
        layout2 = self.generator.generate_block_layout()
        
        assert layout1 is layout2, "Layout should be cached"
    
    def test_get_block_size(self):
        """测试获取宏块大小"""
        block_size = self.generator.get_block_size()
        assert block_size == 32
    
    def test_sufficient_space_for_data(self):
        """测试宏块空间足够容纳所有数据"""
        layout = self.generator.generate_block_layout()
        
        # 计算总共使用的像素数
        sync_pixels = len(layout['sync'])
        header_pixels = len(layout['header']) * 2  # 每个bit用2个像素
        payload_pixels = len(layout['payload']) * 2  # 每个bit用2个像素
        
        total_pixels = sync_pixels + header_pixels + payload_pixels
        
        # 32×32 = 1024像素，应该足够
        assert total_pixels <= 1024, \
            f"Total pixels used ({total_pixels}) exceeds block size (1024)"
    
    def test_header_in_first_row(self):
        """测试Header位置在第一行"""
        layout = self.generator.generate_block_layout()
        header_pairs = layout['header']
        
        # 验证所有Header位置都在第一行（x=0）
        for x1, y1, x2, y2 in header_pairs:
            assert x1 == 0, "Header should be in the first row"
            assert x2 == 0, "Header should be in the first row"
    
    def test_payload_avoids_first_row(self):
        """测试Payload避开第一行"""
        layout = self.generator.generate_block_layout()
        payload_pairs = layout['payload']
        
        # 验证Payload不在第一行
        for x1, y1, x2, y2 in payload_pairs:
            assert x1 != 0, "Payload should not be in the first row"
            assert x2 != 0, "Payload should not be in the first row"
    
    def test_payload_avoids_sync_positions(self):
        """测试Payload避开同步信号位置"""
        layout = self.generator.generate_block_layout()
        sync_positions = set(layout['sync'])
        payload_pairs = layout['payload']
        
        # 验证Payload位置不与同步信号重叠
        for x1, y1, x2, y2 in payload_pairs:
            assert (x1, y1) not in sync_positions, \
                f"Payload position ({x1}, {y1}) overlaps with sync signal"
            assert (x2, y2) not in sync_positions, \
                f"Payload position ({x2}, {y2}) overlaps with sync signal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
