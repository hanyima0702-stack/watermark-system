"""
图像嵌入器单元测试
"""

import pytest
import numpy as np
import cv2
from engines.image.embedding.image_embedder import ImageEmbedder
from engines.image.embedding.macro_block import MacroBlockGenerator
from engines.image.embedding.ppm_modulator import PPMModulator


class TestImageEmbedder:
    """测试ImageEmbedder类"""
    
    @pytest.fixture
    def block_generator(self):
        """创建宏块生成器"""
        return MacroBlockGenerator()
    
    @pytest.fixture
    def modulator(self):
        """创建PPM调制器"""
        return PPMModulator(strength=10)
    
    @pytest.fixture
    def embedder(self, block_generator, modulator):
        """创建图像嵌入器"""
        return ImageEmbedder(block_generator, modulator)
    
    @pytest.fixture
    def watermark_data(self):
        """创建128位测试水印数据"""
        return np.random.randint(0, 2, 128, dtype=np.uint8)
    
    def test_calculate_block_count_exact_fit(self, embedder):
        """测试宏块数量计算 - 完全匹配"""
        # 64×64图像应该容纳2×2个宏块
        rows, cols = embedder.calculate_block_count((64, 64))
        assert rows == 2
        assert cols == 2
    
    def test_calculate_block_count_with_remainder(self, embedder):
        """测试宏块数量计算 - 有余数"""
        # 100×100图像应该容纳3×3个宏块（余数被忽略）
        rows, cols = embedder.calculate_block_count((100, 100))
        assert rows == 3
        assert cols == 3
    
    def test_calculate_block_count_small_image(self, embedder):
        """测试宏块数量计算 - 小图像"""
        # 32×32图像应该容纳1×1个宏块
        rows, cols = embedder.calculate_block_count((32, 32))
        assert rows == 1
        assert cols == 1
    
    def test_calculate_block_count_rectangular(self, embedder):
        """测试宏块数量计算 - 矩形图像"""
        # 64×128图像应该容纳2×4个宏块
        rows, cols = embedder.calculate_block_count((64, 128))
        assert rows == 2
        assert cols == 4
    
    def test_embed_invalid_watermark_length(self, embedder):
        """测试嵌入 - 水印数据长度错误"""
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        invalid_data = np.random.randint(0, 2, 64, dtype=np.uint8)  # 只有64位
        
        with pytest.raises(ValueError, match="水印数据必须是128位"):
            embedder.embed(image, invalid_data)
    
    def test_embed_image_too_small(self, embedder, watermark_data):
        """测试嵌入 - 图像尺寸过小"""
        small_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="图像尺寸.*过小"):
            embedder.embed(small_image, watermark_data)
    
    def test_embed_basic_functionality(self, embedder, watermark_data):
        """测试嵌入 - 基本功能"""
        # 创建64×64的测试图像
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        # 嵌入水印
        watermarked = embedder.embed(image, watermark_data)
        
        # 验证输出
        assert watermarked.shape == image.shape
        assert watermarked.dtype == np.uint8
        
        # 验证图像已被修改（但不应该完全相同）
        assert not np.array_equal(image, watermarked)
    
    def test_embed_preserves_image_dimensions(self, embedder, watermark_data):
        """测试嵌入 - 保持图像尺寸"""
        test_sizes = [(64, 64), (128, 128), (100, 150), (256, 192)]
        
        for height, width in test_sizes:
            image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            watermarked = embedder.embed(image, watermark_data)
            
            assert watermarked.shape == (height, width, 3)
    
    def test_embed_color_space_conversion(self, embedder, watermark_data):
        """测试嵌入 - 色彩空间转换正确性"""
        # 创建纯色图像以便验证
        image = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
        
        # 嵌入水印
        watermarked = embedder.embed(image, watermark_data)
        
        # 验证输出是BGR格式（3通道）
        assert watermarked.shape[2] == 3
        
        # 验证像素值在有效范围内
        assert np.all(watermarked >= 0)
        assert np.all(watermarked <= 255)
    
    def test_embed_multiple_blocks(self, embedder, watermark_data):
        """测试嵌入 - 多个宏块"""
        # 创建128×128图像，应该容纳4×4=16个宏块
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        # 嵌入水印
        watermarked = embedder.embed(image, watermark_data)
        
        # 验证输出
        assert watermarked.shape == image.shape
        
        # 计算宏块数量
        rows, cols = embedder.calculate_block_count((128, 128))
        assert rows == 4
        assert cols == 4
    
    def test_embed_modifies_y_channel_only(self, embedder, watermark_data):
        """测试嵌入 - 主要修改Y通道"""
        # 创建测试图像
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        # 转换到YUV
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # 嵌入水印
        watermarked = embedder.embed(image, watermark_data)
        watermarked_yuv = cv2.cvtColor(watermarked, cv2.COLOR_BGR2YUV)
        
        # Y通道应该有变化
        y_diff = np.sum(np.abs(image_yuv[:, :, 0].astype(int) - 
                               watermarked_yuv[:, :, 0].astype(int)))
        assert y_diff > 0, "Y通道应该被修改"
    
    def test_embed_deterministic(self, embedder, watermark_data):
        """测试嵌入 - 确定性（相同输入产生相同输出）"""
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        # 两次嵌入相同的水印
        watermarked1 = embedder.embed(image.copy(), watermark_data)
        watermarked2 = embedder.embed(image.copy(), watermark_data)
        
        # 结果应该完全相同
        assert np.array_equal(watermarked1, watermarked2)
    
    def test_embed_different_watermarks(self, embedder):
        """测试嵌入 - 不同水印产生不同结果"""
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        # 创建两个不同的水印
        watermark1 = np.zeros(128, dtype=np.uint8)
        watermark2 = np.ones(128, dtype=np.uint8)
        
        # 嵌入不同的水印
        watermarked1 = embedder.embed(image.copy(), watermark1)
        watermarked2 = embedder.embed(image.copy(), watermark2)
        
        # 结果应该不同
        assert not np.array_equal(watermarked1, watermarked2)
    
    def test_embed_large_image(self, embedder, watermark_data):
        """测试嵌入 - 大图像"""
        # 创建1920×1080图像
        image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        
        # 嵌入水印
        watermarked = embedder.embed(image, watermark_data)
        
        # 验证输出
        assert watermarked.shape == image.shape
        
        # 计算宏块数量
        rows, cols = embedder.calculate_block_count((1080, 1920))
        assert rows == 33  # 1080 // 32
        assert cols == 60  # 1920 // 32
    
    def test_embed_preserves_pixel_range(self, embedder, watermark_data):
        """测试嵌入 - 保持像素值范围"""
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        # 嵌入水印
        watermarked = embedder.embed(image, watermark_data)
        
        # 验证所有像素值在[0, 255]范围内
        assert np.all(watermarked >= 0)
        assert np.all(watermarked <= 255)
        assert watermarked.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
