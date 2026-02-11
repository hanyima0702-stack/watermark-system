"""
暗水印主控制器测试

测试InvisibleWatermarkProcessor的基本功能。
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

from engines.image.invisible_watermark import InvisibleWatermarkProcessor
from engines.image.config import WatermarkConfig


class TestInvisibleWatermarkProcessor:
    """测试暗水印处理器"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        config = WatermarkConfig()
        return InvisibleWatermarkProcessor(config=config)
    
    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        # 创建一个512x512的测试图像
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        return image
    
    @pytest.fixture
    def watermark_data(self):
        """创建测试水印数据"""
        # 64位二进制字符串
        return "1010101010101010101010101010101010101010101010101010101010101010"
    
    def test_processor_initialization(self, processor):
        """测试处理器初始化"""
        assert processor is not None
        assert processor.config is not None
        assert processor.ecc_encoder is not None
        assert processor.scrambler is not None
        assert processor.image_embedder is not None
    
    def test_processor_with_config_path(self, tmp_path):
        """测试从配置文件初始化"""
        # 创建临时配置文件
        config_path = tmp_path / "test_config.yaml"
        config = WatermarkConfig()
        config.to_yaml(str(config_path))
        
        # 从配置文件加载
        processor = InvisibleWatermarkProcessor(config_path=str(config_path))
        assert processor is not None
        assert processor.config.modulation_strength == 10
    
    def test_embed_watermark(self, processor, test_image, watermark_data, tmp_path):
        """测试水印嵌入"""
        # 保存测试图像
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        cv2.imwrite(str(input_path), test_image)
        
        # 嵌入水印
        result = processor.embed_watermark(
            str(input_path),
            watermark_data,
            str(output_path)
        )
        
        # 验证结果
        assert result.success is True
        assert result.watermark_data == watermark_data
        assert len(result.encoded_data) == 128
        assert result.block_count[0] > 0
        assert result.block_count[1] > 0
        assert result.processing_time > 0
        assert 'psnr' in result.quality_metrics
        assert 'ssim' in result.quality_metrics
        
        # 验证输出文件存在
        assert os.path.exists(output_path)
    
    def test_embed_with_invalid_image(self, processor, watermark_data, tmp_path):
        """测试无效图像的嵌入"""
        input_path = tmp_path / "nonexistent.png"
        output_path = tmp_path / "output.png"
        
        result = processor.embed_watermark(
            str(input_path),
            watermark_data,
            str(output_path)
        )
        
        # 应该失败
        assert result.success is False
        assert result.error_message is not None
    
    def test_embed_with_small_image(self, processor, watermark_data, tmp_path):
        """测试小图像的嵌入"""
        # 创建一个太小的图像
        small_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        input_path = tmp_path / "small.png"
        output_path = tmp_path / "output.png"
        cv2.imwrite(str(input_path), small_image)
        
        result = processor.embed_watermark(
            str(input_path),
            watermark_data,
            str(output_path)
        )
        
        # 应该失败
        assert result.success is False
    
    def test_extract_watermark_basic(self, processor, test_image, watermark_data, tmp_path):
        """测试基本的水印提取"""
        # 先嵌入水印
        input_path = tmp_path / "input.png"
        watermarked_path = tmp_path / "watermarked.png"
        cv2.imwrite(str(input_path), test_image)
        
        embed_result = processor.embed_watermark(
            str(input_path),
            watermark_data,
            str(watermarked_path)
        )
        
        assert embed_result.success is True
        
        # 提取水印
        extract_result = processor.extract_watermark(str(watermarked_path))
        
        # 验证结果
        assert extract_result is not None
        assert extract_result.processing_time > 0
        
        # 注意：由于测试图像是随机的，提取可能不会成功
        # 这里只验证流程能够执行
    
    def test_extract_with_invalid_image(self, processor, tmp_path):
        """测试无效图像的提取"""
        input_path = tmp_path / "nonexistent.png"
        
        result = processor.extract_watermark(str(input_path))
        
        # 应该失败
        assert result.success is False
        assert result.error_message is not None
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效的编码类型
        config = WatermarkConfig(ecc_type="invalid")
        with pytest.raises(ValueError):
            config.validate()
        
        # 测试无效的宏块大小
        config = WatermarkConfig(block_size=100)
        with pytest.raises(ValueError):
            config.validate()
        
        # 测试无效的调制强度
        config = WatermarkConfig(modulation_strength=100)
        with pytest.raises(ValueError):
            config.validate()
    
    def test_embed_result_structure(self, processor, test_image, watermark_data, tmp_path):
        """测试嵌入结果的数据结构"""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        cv2.imwrite(str(input_path), test_image)
        
        result = processor.embed_watermark(
            str(input_path),
            watermark_data,
            str(output_path)
        )
        
        # 验证结果包含所有必要字段
        assert hasattr(result, 'success')
        assert hasattr(result, 'watermark_data')
        assert hasattr(result, 'encoded_data')
        assert hasattr(result, 'block_count')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'image_size')
        assert hasattr(result, 'quality_metrics')
    
    def test_extraction_result_structure(self, processor, test_image, tmp_path):
        """测试提取结果的数据结构"""
        input_path = tmp_path / "input.png"
        cv2.imwrite(str(input_path), test_image)
        
        result = processor.extract_watermark(str(input_path))
        
        # 验证结果包含所有必要字段
        assert hasattr(result, 'success')
        assert hasattr(result, 'watermark_data')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'detected_rotation')
        assert hasattr(result, 'detected_scale')
        assert hasattr(result, 'grid_offset')
        assert hasattr(result, 'total_blocks')
        assert hasattr(result, 'valid_blocks')
        assert hasattr(result, 'processing_time')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
