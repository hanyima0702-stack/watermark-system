"""
频域暗水印处理器单元测试
"""

import pytest
import numpy as np
import cv2
from PIL import Image

from engines.image.frequency_watermark import (
    FrequencyWatermarkProcessor,
    FrequencyWatermarkConfig,
    FrequencyMethod,
    WaveletType,
    DCTWatermark,
    DWTWatermark
)


class TestFrequencyWatermarkProcessor:
    """频域水印处理器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建测试图像
        self.test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        self.gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_RGB2GRAY)
        
        # 创建测试水印
        self.watermark_bits = np.random.randint(0, 2, 64, dtype=np.uint8)
        
        # 创建配置
        self.dct_config = FrequencyWatermarkConfig(
            method=FrequencyMethod.DCT,
            strength=0.1,
            block_size=8,
            frequency_band="mid"
        )
        
        self.dwt_config = FrequencyWatermarkConfig(
            method=FrequencyMethod.DWT,
            strength=0.05,
            wavelet_type=WaveletType.HAAR,
            decomposition_levels=3,
            frequency_band="mid"
        )
    
    def test_dct_watermark_embed_extract(self):
        """测试DCT水印嵌入和提取"""
        processor = FrequencyWatermarkProcessor(self.dct_config)
        
        # 嵌入水印
        embed_result = processor.embed_watermark(self.gray_image, self.watermark_bits)
        
        assert embed_result is not None
        assert embed_result.watermarked_image is not None
        assert embed_result.psnr > 30  # PSNR应该大于30dB
        assert 0 <= embed_result.ssim <= 1  # SSIM在0-1之间
        
        # 提取水印
        extract_result = processor.extract_watermark(
            embed_result.watermarked_image, 
            len(self.watermark_bits)
        )
        
        assert extract_result is not None
        assert len(extract_result.extracted_bits) == len(self.watermark_bits)
        assert extract_result.confidence_score >= 0
        assert extract_result.extraction_method == "DCT"
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.3  # 比特错误率应该小于30%
    
    def test_dwt_watermark_embed_extract(self):
        """测试DWT水印嵌入和提取"""
        processor = FrequencyWatermarkProcessor(self.dwt_config)
        
        # 嵌入水印
        embed_result = processor.embed_watermark(self.gray_image, self.watermark_bits)
        
        assert embed_result is not None
        assert embed_result.watermarked_image is not None
        assert embed_result.psnr > 25  # DWT的PSNR可能稍低
        assert 0 <= embed_result.ssim <= 1
        
        # 提取水印
        extract_result = processor.extract_watermark(
            embed_result.watermarked_image, 
            len(self.watermark_bits)
        )
        
        assert extract_result is not None
        assert len(extract_result.extracted_bits) == len(self.watermark_bits)
        assert extract_result.confidence_score >= 0
        assert extract_result.extraction_method == "DWT"
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5  # DWT的错误率可能稍高
    
    def test_dct_different_frequency_bands(self):
        """测试DCT不同频带的水印"""
        frequency_bands = ["low", "mid", "high"]
        
        for band in frequency_bands:
            config = FrequencyWatermarkConfig(
                method=FrequencyMethod.DCT,
                strength=0.1,
                frequency_band=band
            )
            
            processor = FrequencyWatermarkProcessor(config)
            
            # 嵌入水印
            embed_result = processor.embed_watermark(self.gray_image, self.watermark_bits)
            assert embed_result is not None
            
            # 提取水印
            extract_result = processor.extract_watermark(
                embed_result.watermarked_image, 
                len(self.watermark_bits)
            )
            assert extract_result is not None
            assert len(extract_result.extracted_bits) == len(self.watermark_bits)
    
    def test_dwt_different_wavelets(self):
        """测试DWT不同小波类型"""
        wavelets = [WaveletType.HAAR, WaveletType.DAUBECHIES, WaveletType.BIORTHOGONAL]
        
        for wavelet in wavelets:
            config = FrequencyWatermarkConfig(
                method=FrequencyMethod.DWT,
                strength=0.05,
                wavelet_type=wavelet
            )
            
            processor = FrequencyWatermarkProcessor(config)
            
            # 嵌入水印
            embed_result = processor.embed_watermark(self.gray_image, self.watermark_bits)
            assert embed_result is not None
            
            # 提取水印
            extract_result = processor.extract_watermark(
                embed_result.watermarked_image, 
                len(self.watermark_bits)
            )
            assert extract_result is not None
    
    def test_color_image_processing(self):
        """测试彩色图像处理"""
        processor = FrequencyWatermarkProcessor(self.dct_config)
        
        # 嵌入水印到彩色图像
        embed_result = processor.embed_watermark(self.test_image, self.watermark_bits)
        
        assert embed_result is not None
        assert embed_result.watermarked_image.shape == self.test_image.shape
        
        # 提取水印
        extract_result = processor.extract_watermark(
            embed_result.watermarked_image, 
            len(self.watermark_bits)
        )
        
        assert extract_result is not None
        assert len(extract_result.extracted_bits) == len(self.watermark_bits)
    
    def test_robustness_jpeg_compression(self):
        """测试JPEG压缩鲁棒性"""
        processor = FrequencyWatermarkProcessor(self.dct_config)
        
        # 嵌入水印
        embed_result = processor.embed_watermark(self.gray_image, self.watermark_bits)
        
        # 模拟JPEG压缩
        # 将图像编码为JPEG再解码
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, encoded_img = cv2.imencode('.jpg', embed_result.watermarked_image, encode_param)
        compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
        
        # 从压缩后的图像提取水印
        extract_result = processor.extract_watermark(compressed_img, len(self.watermark_bits))
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 压缩后错误率应该仍然可接受
    
    def test_robustness_noise(self):
        """测试噪声鲁棒性"""
        processor = FrequencyWatermarkProcessor(self.dct_config)
        
        # 嵌入水印
        embed_result = processor.embed_watermark(self.gray_image, self.watermark_bits)
        
        # 添加高斯噪声
        noise = np.random.normal(0, 5, embed_result.watermarked_image.shape)
        noisy_image = np.clip(embed_result.watermarked_image + noise, 0, 255).astype(np.uint8)
        
        # 从噪声图像提取水印
        extract_result = processor.extract_watermark(noisy_image, len(self.watermark_bits))
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5  # 噪声后错误率应该仍然可接受
    
    def test_robustness_scaling(self):
        """测试缩放鲁棒性"""
        processor = FrequencyWatermarkProcessor(self.dct_config)
        
        # 嵌入水印
        embed_result = processor.embed_watermark(self.gray_image, self.watermark_bits)
        
        # 缩放图像
        h, w = embed_result.watermarked_image.shape
        scaled_img = cv2.resize(embed_result.watermarked_image, (w//2, h//2))
        restored_img = cv2.resize(scaled_img, (w, h))
        
        # 从缩放后的图像提取水印
        extract_result = processor.extract_watermark(restored_img, len(self.watermark_bits))
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5  # 缩放后错误率可能较高但仍应可检测


class TestDCTWatermark:
    """DCT水印处理器专项测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config = FrequencyWatermarkConfig(
            method=FrequencyMethod.DCT,
            strength=0.1,
            block_size=8
        )
        self.processor = DCTWatermark(self.config)
        self.test_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        self.watermark_bits = np.random.randint(0, 2, 32, dtype=np.uint8)
    
    def test_dct_block_processing(self):
        """测试DCT块处理"""
        # 测试图像尺寸不是8的倍数的情况
        odd_image = np.random.randint(0, 256, (127, 129), dtype=np.uint8)
        
        embed_result = self.processor.embed(odd_image, self.watermark_bits)
        assert embed_result is not None
        assert embed_result.watermarked_image.shape == odd_image.shape
        
        extract_result = self.processor.extract(
            embed_result.watermarked_image, 
            len(self.watermark_bits)
        )
        assert extract_result is not None
    
    def test_dct_quality_metrics(self):
        """测试DCT质量指标计算"""
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # PSNR应该是合理的值
        assert 20 < embed_result.psnr < 60
        
        # SSIM应该接近1
        assert 0.8 < embed_result.ssim <= 1.0
        
        # 元数据应该包含必要信息
        assert embed_result.metadata["method"] == "DCT"
        assert embed_result.metadata["block_size"] == 8
        assert embed_result.metadata["watermark_length"] == len(self.watermark_bits)


class TestDWTWatermark:
    """DWT水印处理器专项测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config = FrequencyWatermarkConfig(
            method=FrequencyMethod.DWT,
            strength=0.05,
            wavelet_type=WaveletType.HAAR,
            decomposition_levels=2
        )
        self.processor = DWTWatermark(self.config)
        self.test_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        self.watermark_bits = np.random.randint(0, 2, 32, dtype=np.uint8)
    
    def test_dwt_decomposition_levels(self):
        """测试不同分解级数"""
        for levels in [1, 2, 3, 4]:
            config = FrequencyWatermarkConfig(
                method=FrequencyMethod.DWT,
                wavelet_type=WaveletType.HAAR,
                decomposition_levels=levels
            )
            processor = DWTWatermark(config)
            
            embed_result = processor.embed(self.test_image, self.watermark_bits)
            assert embed_result is not None
            
            extract_result = processor.extract(
                embed_result.watermarked_image, 
                len(self.watermark_bits)
            )
            assert extract_result is not None
    
    def test_dwt_watermark_capacity(self):
        """测试DWT水印容量"""
        # 测试不同长度的水印
        for length in [16, 32, 64, 128]:
            watermark = np.random.randint(0, 2, length, dtype=np.uint8)
            
            try:
                embed_result = self.processor.embed(self.test_image, watermark)
                assert embed_result is not None
                
                extract_result = self.processor.extract(
                    embed_result.watermarked_image, 
                    length
                )
                assert len(extract_result.extracted_bits) == length
            except ValueError:
                # 如果水印太长，应该抛出异常
                assert length > 1000  # 只有很长的水印才应该失败


class TestRobustnessAttacks:
    """鲁棒性攻击测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.dct_config = FrequencyWatermarkConfig(
            method=FrequencyMethod.DCT,
            strength=0.15,  # 稍高的强度以提高鲁棒性
            robustness_level="high"
        )
        self.processor = FrequencyWatermarkProcessor(self.dct_config)
        self.test_image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        self.watermark_bits = np.random.randint(0, 2, 64, dtype=np.uint8)
    
    def test_rotation_attack(self):
        """测试旋转攻击"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 旋转攻击
        center = (embed_result.watermarked_image.shape[1]//2, embed_result.watermarked_image.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)  # 旋转5度
        rotated_img = cv2.warpAffine(embed_result.watermarked_image, rotation_matrix, 
                                   embed_result.watermarked_image.shape[::-1])
        
        # 反向旋转
        rotation_matrix_inv = cv2.getRotationMatrix2D(center, -5, 1.0)
        restored_img = cv2.warpAffine(rotated_img, rotation_matrix_inv, 
                                    rotated_img.shape[::-1])
        
        # 提取水印
        extract_result = self.processor.extract_watermark(restored_img, len(self.watermark_bits))
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5  # 旋转攻击后仍应可检测
    
    def test_cropping_attack(self):
        """测试裁剪攻击"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 裁剪攻击（裁剪掉10%的边缘）
        h, w = embed_result.watermarked_image.shape
        crop_size = int(0.1 * min(h, w))
        cropped_img = embed_result.watermarked_image[crop_size:h-crop_size, crop_size:w-crop_size]
        
        # 恢复到原始尺寸
        restored_img = cv2.resize(cropped_img, (w, h))
        
        # 提取水印
        extract_result = self.processor.extract_watermark(restored_img, len(self.watermark_bits))
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5  # 裁剪攻击后仍应可检测
    
    def test_filtering_attack(self):
        """测试滤波攻击"""
        # 嵌入水印
        embed_result = self.processor.embed_watermark(self.test_image, self.watermark_bits)
        
        # 高斯滤波攻击
        filtered_img = cv2.GaussianBlur(embed_result.watermarked_image, (3, 3), 1.0)
        
        # 提取水印
        extract_result = self.processor.extract_watermark(filtered_img, len(self.watermark_bits))
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 滤波攻击后仍应可检测


if __name__ == "__main__":
    pytest.main([__file__])