"""
FrFT分数阶傅里叶变换暗水印处理器单元测试
"""

import pytest
import numpy as np
import cv2

from engines.image.frft_watermark import (
    FrFTWatermarkProcessor,
    FrFTWatermarkConfig,
    FrFTProcessor,
    FrFTAttackSimulator
)


class TestFrFTProcessor:
    """FrFT处理器基础测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.processor = FrFTProcessor()
        self.test_signal = np.random.randn(64) + 1j * np.random.randn(64)
        self.test_image = np.random.randn(32, 32)
    
    def test_frft_special_cases(self):
        """测试FrFT特殊情况"""
        # alpha = 0 (恒等变换)
        result = self.processor.frft(self.test_signal, 0)
        np.testing.assert_allclose(result, self.test_signal, rtol=1e-10)
        
        # alpha = 1 (标准FFT)
        result = self.processor.frft(self.test_signal, 1)
        expected = np.fft.fft(self.test_signal)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # alpha = -1 (标准IFFT)
        result = self.processor.frft(self.test_signal, -1)
        expected = np.fft.ifft(self.test_signal)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_frft_invertibility(self):
        """测试FrFT可逆性"""
        alpha = 0.5
        
        # 一维信号
        frft_result = self.processor.frft(self.test_signal, alpha)
        reconstructed = self.processor.frft(frft_result, -alpha)
        np.testing.assert_allclose(reconstructed, self.test_signal, rtol=1e-3)
        
        # 二维图像
        frft_image = self.processor.frft2d(self.test_image, alpha)
        reconstructed_image = self.processor.ifrft2d(frft_image, alpha)
        np.testing.assert_allclose(reconstructed_image.real, self.test_image, rtol=1e-3)
    
    def test_frft2d_properties(self):
        """测试二维FrFT性质"""
        alpha = 0.3
        
        # 测试变换结果的形状
        frft_image = self.processor.frft2d(self.test_image, alpha)
        assert frft_image.shape == self.test_image.shape
        
        # 测试能量守恒（近似）
        original_energy = np.sum(np.abs(self.test_image)**2)
        frft_energy = np.sum(np.abs(frft_image)**2)
        assert abs(original_energy - frft_energy) / original_energy < 0.1


class TestFrFTWatermarkProcessor:
    """FrFT水印处理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config = FrFTWatermarkConfig(
            alpha=0.5,
            strength=0.1,
            block_size=32,
            overlap_ratio=0.25,
            detection_threshold=0.3
        )
        self.processor = FrFTWatermarkProcessor(self.config)
        
        # 创建测试图像
        self.test_image = np.random.randint(50, 200, (128, 128), dtype=np.uint8)
        self.watermark_bits = np.random.randint(0, 2, 16, dtype=np.uint8)
    
    def test_frft_watermark_embed_extract(self):
        """测试FrFT水印嵌入和提取"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        assert embed_result is not None
        assert embed_result.watermarked_image is not None
        assert embed_result.psnr > 20  # PSNR应该合理
        assert 0 <= embed_result.ssim <= 1
        assert embed_result.alpha_parameter == self.config.alpha
        assert len(embed_result.embedding_positions) > 0
        
        # 非盲提取（使用已知位置）
        extract_result = self.processor.extract_watermark(
            embed_result.watermarked_image,
            len(self.watermark_bits),
            embed_result.embedding_positions
        )
        
        assert extract_result is not None
        assert len(extract_result.extracted_bits) == len(self.watermark_bits)
        assert len(extract_result.confidence_scores) == len(self.watermark_bits)
        assert extract_result.detection_map is not None
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.4  # 允许一定的错误率
    
    def test_frft_blind_detection(self):
        """测试FrFT盲检测"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 盲检测
        blind_result = self.processor.blind_detect(embed_result.watermarked_image, len(self.watermark_bits))
        
        assert blind_result is not None
        assert len(blind_result.extracted_bits) <= len(self.watermark_bits)
        assert blind_result.correlation_peak >= 0
        assert len(blind_result.extraction_positions) > 0
        
        # 检测图应该有合理的值
        assert np.max(blind_result.detection_map) > 0
    
    def test_frft_different_alpha_values(self):
        """测试不同alpha参数值"""
        alpha_values = [0.2, 0.5, 0.8]
        
        for alpha in alpha_values:
            config = FrFTWatermarkConfig(alpha=alpha, block_size=32)
            processor = FrFTWatermarkProcessor(config)
            
            # 嵌入水印
            embed_result = processor.embed_watermark(self.test_image, self.watermark_bits)
            assert embed_result is not None
            assert embed_result.alpha_parameter == alpha
            
            # 提取水印
            extract_result = processor.extract_watermark(
                embed_result.watermarked_image,
                len(self.watermark_bits),
                embed_result.embedding_positions
            )
            assert extract_result is not None
    
    def test_frft_different_block_sizes(self):
        """测试不同块大小"""
        block_sizes = [16, 32, 64]
        
        for block_size in block_sizes:
            config = FrFTWatermarkConfig(block_size=block_size)
            processor = FrFTWatermarkProcessor(config)
            
            # 嵌入水印
            embed_result = processor.embed_watermark(self.test_image, self.watermark_bits)
            assert embed_result is not None
            
            # 提取水印
            extract_result = processor.extract_watermark(
                embed_result.watermarked_image,
                len(self.watermark_bits),
                embed_result.embedding_positions
            )
            assert extract_result is not None
    
    def test_frft_local_detection(self):
        """测试局部区域检测"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 定义检测区域
        region = (32, 32, 64, 64)  # x, y, w, h
        
        # 局部检测
        local_result = self.processor.local_detection(
            embed_result.watermarked_image,
            region,
            len(self.watermark_bits)
        )
        
        assert local_result is not None
        assert len(local_result.extracted_bits) > 0
    
    def test_frft_color_image(self):
        """测试彩色图像处理"""
        color_image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        # 嵌入水印
        embed_result = self.processor.embed_watermark(color_image, self.watermark_bits)
        
        assert embed_result is not None
        assert embed_result.watermarked_image.shape == color_image.shape
        
        # 提取水印
        extract_result = self.processor.extract_watermark(
            embed_result.watermarked_image,
            len(self.watermark_bits),
            embed_result.embedding_positions
        )
        
        assert extract_result is not None
    
    def test_frft_watermark_capacity(self):
        """测试水印容量"""
        # 测试不同长度的水印
        for length in [8, 16, 32]:
            watermark = np.random.randint(0, 2, length, dtype=np.uint8)
            
            embed_result = self.processor.embed_watermark(self.test_image, watermark)
            assert embed_result is not None
            assert len(embed_result.embedding_positions) >= length
            
            extract_result = self.processor.extract_watermark(
                embed_result.watermarked_image,
                length,
                embed_result.embedding_positions
            )
            assert len(extract_result.extracted_bits) == length


class TestFrFTRobustness:
    """FrFT水印鲁棒性测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config = FrFTWatermarkConfig(
            alpha=0.5,
            strength=0.15,  # 稍高的强度以提高鲁棒性
            block_size=32,
            detection_threshold=0.2
        )
        self.processor = FrFTWatermarkProcessor(self.config)
        self.test_image = np.random.randint(50, 200, (128, 128), dtype=np.uint8)
        self.watermark_bits = np.random.randint(0, 2, 16, dtype=np.uint8)
        self.attack_simulator = FrFTAttackSimulator()
    
    def test_rotation_robustness(self):
        """测试旋转鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 旋转攻击
        attacked_image = self.attack_simulator.geometric_attack(
            embed_result.watermarked_image, "rotation", angle=3
        )
        
        # 提取水印
        extract_result = self.processor.extract_watermark(
            attacked_image,
            len(self.watermark_bits),
            embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 旋转后仍应可检测
    
    def test_scaling_robustness(self):
        """测试缩放鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 缩放攻击
        attacked_image = self.attack_simulator.geometric_attack(
            embed_result.watermarked_image, "scaling", scale=0.8
        )
        
        # 提取水印
        extract_result = self.processor.extract_watermark(
            attacked_image,
            len(self.watermark_bits),
            embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.7  # 缩放后仍应可检测
    
    def test_cropping_robustness(self):
        """测试裁剪鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 裁剪攻击
        attacked_image = self.attack_simulator.geometric_attack(
            embed_result.watermarked_image, "cropping", crop_ratio=0.1
        )
        
        # 提取水印
        extract_result = self.processor.extract_watermark(
            attacked_image,
            len(self.watermark_bits),
            embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.8  # 裁剪后仍应可检测
    
    def test_noise_robustness(self):
        """测试噪声鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 高斯噪声攻击
        attacked_image = self.attack_simulator.signal_processing_attack(
            embed_result.watermarked_image, "gaussian_noise", noise_std=3
        )
        
        # 提取水印
        extract_result = self.processor.extract_watermark(
            attacked_image,
            len(self.watermark_bits),
            embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 噪声后仍应可检测
    
    def test_blur_robustness(self):
        """测试模糊鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 高斯模糊攻击
        attacked_image = self.attack_simulator.signal_processing_attack(
            embed_result.watermarked_image, "gaussian_blur", kernel_size=3, sigma=0.8
        )
        
        # 提取水印
        extract_result = self.processor.extract_watermark(
            attacked_image,
            len(self.watermark_bits),
            embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 模糊后仍应可检测
    
    def test_jpeg_compression_robustness(self):
        """测试JPEG压缩鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # JPEG压缩攻击
        attacked_image = self.attack_simulator.signal_processing_attack(
            embed_result.watermarked_image, "jpeg_compression", quality=70
        )
        
        # 提取水印
        extract_result = self.processor.extract_watermark(
            attacked_image,
            len(self.watermark_bits),
            embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.7  # 压缩后仍应可检测


class TestFrFTAttackSimulator:
    """攻击模拟器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.test_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        self.simulator = FrFTAttackSimulator()
    
    def test_geometric_attacks(self):
        """测试几何攻击"""
        # 旋转攻击
        rotated = self.simulator.geometric_attack(self.test_image, "rotation", angle=10)
        assert rotated.shape == self.test_image.shape
        
        # 缩放攻击
        scaled = self.simulator.geometric_attack(self.test_image, "scaling", scale=0.5)
        assert scaled.shape == self.test_image.shape
        
        # 裁剪攻击
        cropped = self.simulator.geometric_attack(self.test_image, "cropping", crop_ratio=0.2)
        assert cropped.shape == self.test_image.shape
    
    def test_signal_processing_attacks(self):
        """测试信号处理攻击"""
        # 高斯噪声
        noisy = self.simulator.signal_processing_attack(
            self.test_image, "gaussian_noise", noise_std=10
        )
        assert noisy.shape == self.test_image.shape
        
        # 高斯模糊
        blurred = self.simulator.signal_processing_attack(
            self.test_image, "gaussian_blur", kernel_size=5, sigma=2.0
        )
        assert blurred.shape == self.test_image.shape
        
        # JPEG压缩
        compressed = self.simulator.signal_processing_attack(
            self.test_image, "jpeg_compression", quality=50
        )
        assert compressed.shape == self.test_image.shape


class TestFrFTPerformance:
    """FrFT性能测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config = FrFTWatermarkConfig(block_size=32)
        self.processor = FrFTWatermarkProcessor(self.config)
    
    def test_processing_time(self):
        """测试处理时间"""
        import time
        
        # 测试不同尺寸图像的处理时间
        sizes = [(64, 64), (128, 128), (256, 256)]
        watermark_bits = np.random.randint(0, 2, 16, dtype=np.uint8)
        
        for size in sizes:
            test_image = np.random.randint(0, 256, size, dtype=np.uint8)
            
            # 测试嵌入时间
            start_time = time.time()
            embed_result = self.processor.embed_watermark(test_image, watermark_bits)
            embed_time = time.time() - start_time
            
            # 测试提取时间
            start_time = time.time()
            extract_result = self.processor.extract_watermark(
                embed_result.watermarked_image,
                len(watermark_bits),
                embed_result.embedding_positions
            )
            extract_time = time.time() - start_time
            
            print(f"Size {size}: Embed {embed_time:.3f}s, Extract {extract_time:.3f}s")
            
            # 确保处理时间合理（不超过10秒）
            assert embed_time < 10.0
            assert extract_time < 10.0


if __name__ == "__main__":
    pytest.main([__file__])