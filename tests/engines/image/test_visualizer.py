"""
可视化工具测试
"""

import pytest
import numpy as np
import cv2
import os
import tempfile
from engines.image.utils.visualizer import (
    visualize_fft_spectrum,
    visualize_block_grid,
    visualize_confidence_heatmap,
    visualize_comparison,
    visualize_bit_distribution,
    create_visualization_report
)


class TestFFTVisualization:
    """测试FFT频谱可视化"""
    
    def test_visualize_fft_basic(self):
        """测试基本FFT可视化"""
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = visualize_fft_spectrum(image)
        
        assert result is not None
        assert len(result.shape) == 3
        assert result.shape[2] == 3
    
    def test_visualize_fft_with_peaks(self):
        """测试带峰值标记的FFT可视化"""
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        peaks = [(64, 64), (32, 32), (96, 96)]
        
        result = visualize_fft_spectrum(image, peaks=peaks)
        
        assert result is not None
        assert len(result.shape) == 3
    
    def test_visualize_fft_save(self):
        """测试保存FFT频谱"""
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            result = visualize_fft_spectrum(image, output_path=output_path)
            assert os.path.exists(output_path)
            assert result is not None
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_visualize_fft_grayscale(self):
        """测试灰度图FFT可视化"""
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = visualize_fft_spectrum(image)
        
        assert result is not None
        assert len(result.shape) == 3


class TestBlockGridVisualization:
    """测试宏块网格可视化"""
    
    def test_visualize_grid_basic(self):
        """测试基本网格可视化"""
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = visualize_block_grid(image, block_size=32)
        
        assert result is not None
        assert result.shape == image.shape
    
    def test_visualize_grid_with_valid_blocks(self):
        """测试标记有效宏块"""
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        valid_blocks = [(0, 0), (1, 1), (2, 2)]
        
        result = visualize_block_grid(image, block_size=32, valid_blocks=valid_blocks)
        
        assert result is not None
        assert result.shape == image.shape
    
    def test_visualize_grid_save(self):
        """测试保存网格图"""
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            result = visualize_block_grid(image, output_path=output_path)
            assert os.path.exists(output_path)
            assert result is not None
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_visualize_grid_different_sizes(self):
        """测试不同宏块大小"""
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        for block_size in [16, 32, 64]:
            result = visualize_block_grid(image, block_size=block_size)
            assert result is not None


class TestConfidenceHeatmap:
    """测试置信度热力图"""
    
    def test_heatmap_2d(self):
        """测试2D置信度图"""
        confidence_map = np.random.rand(4, 4)
        result = visualize_confidence_heatmap(confidence_map)
        
        assert result is not None
        assert len(result.shape) == 3
    
    def test_heatmap_3d(self):
        """测试3D置信度图"""
        confidence_map = np.random.rand(4, 4, 128)
        result = visualize_confidence_heatmap(confidence_map)
        
        assert result is not None
        assert len(result.shape) == 3
    
    def test_heatmap_save(self):
        """测试保存热力图"""
        confidence_map = np.random.rand(4, 4)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            result = visualize_confidence_heatmap(confidence_map, output_path=output_path)
            assert os.path.exists(output_path)
            assert result is not None
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_heatmap_zero_confidence(self):
        """测试全零置信度"""
        confidence_map = np.zeros((4, 4))
        result = visualize_confidence_heatmap(confidence_map)
        
        assert result is not None


class TestComparisonVisualization:
    """测试对比图可视化"""
    
    def test_comparison_basic(self):
        """测试基本对比"""
        original = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        watermarked = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        result = visualize_comparison(original, watermarked)
        
        assert result is not None
        assert len(result.shape) == 3
    
    def test_comparison_with_titles(self):
        """测试带标题的对比"""
        original = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        watermarked = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        result = visualize_comparison(original, watermarked, 
                                     titles=('原图', '水印图'))
        
        assert result is not None
    
    def test_comparison_with_metrics(self):
        """测试带质量指标的对比"""
        original = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        watermarked = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        metrics = {'PSNR': 45.5, 'SSIM': 0.98}
        
        result = visualize_comparison(original, watermarked, metrics=metrics)
        
        assert result is not None
    
    def test_comparison_save(self):
        """测试保存对比图"""
        original = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        watermarked = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            result = visualize_comparison(original, watermarked, 
                                         output_path=output_path)
            assert os.path.exists(output_path)
            assert result is not None
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_comparison_grayscale(self):
        """测试灰度图对比"""
        original = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        watermarked = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        
        result = visualize_comparison(original, watermarked)
        
        assert result is not None


class TestBitDistribution:
    """测试bit分布可视化"""
    
    def test_bit_distribution_basic(self):
        """测试基本bit分布"""
        bits = np.random.randint(0, 2, 128)
        result = visualize_bit_distribution(bits)
        
        assert result is not None
        assert len(result.shape) == 3
    
    def test_bit_distribution_all_zeros(self):
        """测试全0分布"""
        bits = np.zeros(128, dtype=int)
        result = visualize_bit_distribution(bits)
        
        assert result is not None
    
    def test_bit_distribution_all_ones(self):
        """测试全1分布"""
        bits = np.ones(128, dtype=int)
        result = visualize_bit_distribution(bits)
        
        assert result is not None
    
    def test_bit_distribution_save(self):
        """测试保存分布图"""
        bits = np.random.randint(0, 2, 128)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            result = visualize_bit_distribution(bits, output_path=output_path)
            assert os.path.exists(output_path)
            assert result is not None
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestVisualizationReport:
    """测试综合可视化报告"""
    
    def test_create_report(self):
        """测试创建报告"""
        images = {
            'Image 1': np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            'Image 2': np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            'Image 3': np.random.randint(0, 256, (128, 128), dtype=np.uint8),
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            create_visualization_report(images, output_path)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_create_report_single_image(self):
        """测试单图报告"""
        images = {
            'Single Image': np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            create_visualization_report(images, output_path)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
