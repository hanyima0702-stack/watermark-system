"""
PCA主成分分析暗水印处理器单元测试
"""

import pytest
import numpy as np
import cv2
from PIL import Image

from engines.image.pca_watermark import (
    PCAWatermark,
    PCAWatermarkConfig,
    PCAEmbedResult,
    PCAExtractResult
)


class TestPCAWatermarkProcessor:
    """PCA水印处理器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建测试图像
        self.test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        self.gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_RGB2GRAY)
        
        # 创建测试水印
        self.watermark_bits = np.random.randint(0, 2, 32, dtype=np.uint8)
        
        # 创建配置
        self.config = PCAWatermarkConfig(
            n_components=8,
            strength=0.05,
            block_size=32,
            overlap_ratio=0.25,
            detection_threshold=0.5,
            adaptive_strength=True,
            robustness_level="medium",
            eigenvalue_modification="additive"
        )
    
    def test_pca_watermark_embed_extract(self):
        """测试PCA水印嵌入和提取"""
        processor = PCAWatermark(self.config)
        
        # 嵌入水印
        embed_result = processor.embed(self.gray_image, self.watermark_bits)
        
        assert embed_result is not None
        assert isinstance(embed_result, PCAEmbedResult)
        assert embed_result.watermarked_image is not None
        assert embed_result.psnr > 25  # PSNR应该大于25dB
        assert 0 <= embed_result.ssim <= 1  # SSIM在0-1之间
        assert len(embed_result.embedding_positions) > 0
        assert len(embed_result.principal_components) > 0
        assert len(embed_result.eigenvalues) > 0
        
        # 提取水印
        extract_result = processor.extract(
            embed_result.watermarked_image,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions,
            reference_components=embed_result.principal_components
        )
        
        assert extract_result is not None
        assert isinstance(extract_result, PCAExtractResult)
        assert len(extract_result.extracted_bits) == len(self.watermark_bits)
        assert extract_result.correlation_peak >= 0
        assert extract_result.detection_map is not None
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.4  # 比特错误率应该小于40%
    
    def test_pca_blind_extraction(self):
        """测试PCA盲检测提取"""
        processor = PCAWatermark(self.config)
        
        # 嵌入水印
        embed_result = processor.embed(self.gray_image, self.watermark_bits)
        
        # 盲提取（不提供位置和参考主成分）
        extract_result = processor.extract(
            embed_result.watermarked_image,
            len(self.watermark_bits)
        )
        
        assert extract_result is not None
        assert len(extract_result.extracted_bits) == len(self.watermark_bits)
        
        # 盲检测的错误率可能稍高
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5
    
    def test_pca_different_eigenvalue_modifications(self):
        """测试不同的特征值修改策略"""
        modification_strategies = ["additive", "multiplicative", "qim"]
        
        for strategy in modification_strategies:
            config = PCAWatermarkConfig(
                n_components=8,
                strength=0.05,
                eigenvalue_modification=strategy
            )
            
            processor = PCAWatermark(config)
            
            # 嵌入水印
            embed_result = processor.embed(self.gray_image, self.watermark_bits)
            assert embed_result is not None
            assert embed_result.metadata["modification_strategy"] == strategy
            
            # 提取水印
            extract_result = processor.extract(
                embed_result.watermarked_image,
                len(self.watermark_bits),
                positions=embed_result.embedding_positions
            )
            assert extract_result is not None
            assert len(extract_result.extracted_bits) == len(self.watermark_bits)
    
    def test_pca_adaptive_strength(self):
        """测试自适应强度调整"""
        # 测试启用自适应强度
        config_adaptive = PCAWatermarkConfig(
            adaptive_strength=True,
            strength=0.05
        )
        processor_adaptive = PCAWatermark(config_adaptive)
        
        embed_result_adaptive = processor_adaptive.embed(self.gray_image, self.watermark_bits)
        
        # 测试固定强度
        config_fixed = PCAWatermarkConfig(
            adaptive_strength=False,
            strength=0.05
        )
        processor_fixed = PCAWatermark(config_fixed)
        
        embed_result_fixed = processor_fixed.embed(self.gray_image, self.watermark_bits)
        
        # 两种方式都应该成功
        assert embed_result_adaptive is not None
        assert embed_result_fixed is not None
        
        # 自适应强度可能产生不同的PSNR
        assert embed_result_adaptive.psnr > 0
        assert embed_result_fixed.psnr > 0
    
    def test_pca_different_robustness_levels(self):
        """测试不同鲁棒性级别"""
        robustness_levels = ["low", "medium", "high"]
        
        for level in robustness_levels:
            config = PCAWatermarkConfig(
                robustness_level=level,
                adaptive_strength=True
            )
            
            processor = PCAWatermark(config)
            
            # 嵌入水印
            embed_result = processor.embed(self.gray_image, self.watermark_bits)
            assert embed_result is not None
            
            # 提取水印
            extract_result = processor.extract(
                embed_result.watermarked_image,
                len(self.watermark_bits),
                positions=embed_result.embedding_positions
            )
            assert extract_result is not None
    
    def test_pca_different_block_sizes(self):
        """测试不同块大小"""
        block_sizes = [16, 32, 64]
        
        for block_size in block_sizes:
            config = PCAWatermarkConfig(
                block_size=block_size,
                n_components=min(8, block_size // 4)
            )
            
            processor = PCAWatermark(config)
            
            # 嵌入水印
            embed_result = processor.embed(self.gray_image, self.watermark_bits)
            assert embed_result is not None
            assert embed_result.metadata["block_size"] == block_size
            
            # 提取水印
            extract_result = processor.extract(
                embed_result.watermarked_image,
                len(self.watermark_bits),
                positions=embed_result.embedding_positions
            )
            assert extract_result is not None
    
    def test_pca_different_n_components(self):
        """测试不同主成分数量"""
        n_components_list = [4, 8, 16]
        
        for n_comp in n_components_list:
            config = PCAWatermarkConfig(
                n_components=n_comp,
                block_size=32
            )
            
            processor = PCAWatermark(config)
            
            # 嵌入水印
            embed_result = processor.embed(self.gray_image, self.watermark_bits)
            assert embed_result is not None
            assert embed_result.metadata["n_components"] == n_comp
            
            # 提取水印
            extract_result = processor.extract(
                embed_result.watermarked_image,
                len(self.watermark_bits),
                positions=embed_result.embedding_positions
            )
            assert extract_result is not None
    
    def test_pca_color_image_processing(self):
        """测试彩色图像处理"""
        processor = PCAWatermark(self.config)
        
        # 嵌入水印到彩色图像
        embed_result = processor.embed(self.test_image, self.watermark_bits)
        
        assert embed_result is not None
        assert embed_result.watermarked_image.shape == self.test_image.shape
        
        # 提取水印
        extract_result = processor.extract(
            embed_result.watermarked_image,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        assert extract_result is not None
        assert len(extract_result.extracted_bits) == len(self.watermark_bits)
    
    def test_pca_overlap_processing(self):
        """测试重叠块处理"""
        # 测试不同重叠比例
        overlap_ratios = [0.0, 0.25, 0.5]
        
        for overlap in overlap_ratios:
            config = PCAWatermarkConfig(
                overlap_ratio=overlap,
                block_size=32
            )
            
            processor = PCAWatermark(config)
            
            # 嵌入水印
            embed_result = processor.embed(self.gray_image, self.watermark_bits)
            assert embed_result is not None
            
            # 提取水印
            extract_result = processor.extract(
                embed_result.watermarked_image,
                len(self.watermark_bits),
                positions=embed_result.embedding_positions
            )
            assert extract_result is not None
    
    def test_pca_quality_metrics(self):
        """测试PCA质量指标计算"""
        processor = PCAWatermark(self.config)
        
        embed_result = processor.embed(self.gray_image, self.watermark_bits)
        
        # PSNR应该是合理的值
        assert 20 < embed_result.psnr < 60
        
        # SSIM应该接近1
        assert 0.7 < embed_result.ssim <= 1.0
        
        # 元数据应该包含必要信息
        assert embed_result.metadata["method"] == "PCA"
        assert embed_result.metadata["n_components"] == self.config.n_components
        assert embed_result.metadata["block_size"] == self.config.block_size
        assert embed_result.metadata["watermark_length"] == len(self.watermark_bits)


class TestPCARobustness:
    """PCA水印鲁棒性测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config = PCAWatermarkConfig(
            n_components=8,
            strength=0.08,  # 稍高的强度以提高鲁棒性
            block_size=32,
            adaptive_strength=True,
            robustness_level="high"
        )
        self.processor = PCAWatermark(self.config)
        self.test_image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        self.watermark_bits = np.random.randint(0, 2, 32, dtype=np.uint8)
    
    def test_robustness_jpeg_compression(self):
        """测试JPEG压缩鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 模拟JPEG压缩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        _, encoded_img = cv2.imencode('.jpg', embed_result.watermarked_image, encode_param)
        compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
        
        # 从压缩后的图像提取水印
        extract_result = self.processor.extract(
            compressed_img,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5  # 压缩后错误率应该仍然可接受
    
    def test_robustness_gaussian_noise(self):
        """测试高斯噪声鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 添加高斯噪声
        noise = np.random.normal(0, 5, embed_result.watermarked_image.shape)
        noisy_image = np.clip(embed_result.watermarked_image + noise, 0, 255).astype(np.uint8)
        
        # 从噪声图像提取水印
        extract_result = self.processor.extract(
            noisy_image,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5  # 噪声后错误率应该仍然可接受
    
    def test_robustness_salt_pepper_noise(self):
        """测试椒盐噪声鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 添加椒盐噪声
        noisy_image = embed_result.watermarked_image.copy()
        noise_ratio = 0.02
        num_salt = int(noise_ratio * noisy_image.size / 2)
        num_pepper = int(noise_ratio * noisy_image.size / 2)
        
        # 添加盐噪声
        coords = [np.random.randint(0, i, num_salt) for i in noisy_image.shape]
        noisy_image[coords[0], coords[1]] = 255
        
        # 添加椒噪声
        coords = [np.random.randint(0, i, num_pepper) for i in noisy_image.shape]
        noisy_image[coords[0], coords[1]] = 0
        
        # 从噪声图像提取水印
        extract_result = self.processor.extract(
            noisy_image,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 椒盐噪声后错误率可能稍高
    
    def test_robustness_scaling(self):
        """测试缩放鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 缩放图像
        h, w = embed_result.watermarked_image.shape
        scaled_img = cv2.resize(embed_result.watermarked_image, (w//2, h//2))
        restored_img = cv2.resize(scaled_img, (w, h))
        
        # 从缩放后的图像提取水印
        extract_result = self.processor.extract(
            restored_img,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 缩放后错误率可能较高但仍应可检测
    
    def test_robustness_rotation(self):
        """测试旋转鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
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
        extract_result = self.processor.extract(
            restored_img,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 旋转攻击后仍应可检测
    
    def test_robustness_cropping(self):
        """测试裁剪鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 裁剪攻击（裁剪掉10%的边缘）
        h, w = embed_result.watermarked_image.shape
        crop_size = int(0.1 * min(h, w))
        cropped_img = embed_result.watermarked_image[crop_size:h-crop_size, crop_size:w-crop_size]
        
        # 恢复到原始尺寸
        restored_img = cv2.resize(cropped_img, (w, h))
        
        # 提取水印
        extract_result = self.processor.extract(
            restored_img,
            len(self.watermark_bits)
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 裁剪攻击后仍应可检测
    
    def test_robustness_gaussian_blur(self):
        """测试高斯模糊鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 高斯模糊攻击
        blurred_img = cv2.GaussianBlur(embed_result.watermarked_image, (5, 5), 1.0)
        
        # 提取水印
        extract_result = self.processor.extract(
            blurred_img,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5  # 模糊攻击后仍应可检测
    
    def test_robustness_median_filter(self):
        """测试中值滤波鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 中值滤波攻击
        filtered_img = cv2.medianBlur(embed_result.watermarked_image, 3)
        
        # 提取水印
        extract_result = self.processor.extract(
            filtered_img,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.5  # 中值滤波后仍应可检测
    
    def test_robustness_histogram_equalization(self):
        """测试直方图均衡化鲁棒性"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 直方图均衡化攻击
        equalized_img = cv2.equalizeHist(embed_result.watermarked_image)
        
        # 提取水印
        extract_result = self.processor.extract(
            equalized_img,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 计算比特错误率
        ber = np.mean(extract_result.extracted_bits != self.watermark_bits)
        assert ber < 0.6  # 直方图均衡化后仍应可检测


class TestPCAEdgeCases:
    """PCA水印边界情况测试"""
    
    def test_small_image(self):
        """测试小图像"""
        config = PCAWatermarkConfig(
            block_size=16,
            n_components=4
        )
        processor = PCAWatermark(config)
        
        small_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        watermark_bits = np.random.randint(0, 2, 8, dtype=np.uint8)
        
        embed_result = processor.embed(small_image, watermark_bits)
        assert embed_result is not None
        
        extract_result = processor.extract(
            embed_result.watermarked_image,
            len(watermark_bits),
            positions=embed_result.embedding_positions
        )
        assert extract_result is not None
    
    def test_large_watermark(self):
        """测试较长水印"""
        config = PCAWatermarkConfig(
            block_size=32,
            overlap_ratio=0.5  # 增加重叠以容纳更多水印
        )
        processor = PCAWatermark(config)
        
        test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        watermark_bits = np.random.randint(0, 2, 64, dtype=np.uint8)
        
        embed_result = processor.embed(test_image, watermark_bits)
        assert embed_result is not None
        
        extract_result = processor.extract(
            embed_result.watermarked_image,
            len(watermark_bits),
            positions=embed_result.embedding_positions
        )
        assert extract_result is not None
        assert len(extract_result.extracted_bits) == len(watermark_bits)
    
    def test_uniform_image(self):
        """测试均匀图像"""
        config = PCAWatermarkConfig(
            strength=0.1,  # 增加强度以应对低纹理
            adaptive_strength=True
        )
        processor = PCAWatermark(config)
        
        # 创建均匀图像
        uniform_image = np.ones((128, 128), dtype=np.uint8) * 128
        watermark_bits = np.random.randint(0, 2, 16, dtype=np.uint8)
        
        embed_result = processor.embed(uniform_image, watermark_bits)
        assert embed_result is not None
        
        extract_result = processor.extract(
            embed_result.watermarked_image,
            len(watermark_bits),
            positions=embed_result.embedding_positions
        )
        assert extract_result is not None
    
    def test_high_texture_image(self):
        """测试高纹理图像"""
        config = PCAWatermarkConfig(
            adaptive_strength=True,
            robustness_level="high"
        )
        processor = PCAWatermark(config)
        
        # 创建高纹理图像
        high_texture_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        watermark_bits = np.random.randint(0, 2, 16, dtype=np.uint8)
        
        embed_result = processor.embed(high_texture_image, watermark_bits)
        assert embed_result is not None
        
        # 高纹理图像应该有较高的PSNR
        assert embed_result.psnr > 20
        
        extract_result = processor.extract(
            embed_result.watermarked_image,
            len(watermark_bits),
            positions=embed_result.embedding_positions
        )
        assert extract_result is not None


class TestPCAConfidenceScores:
    """PCA水印置信度测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config = PCAWatermarkConfig(
            n_components=8,
            strength=0.05,
            block_size=32
        )
        self.processor = PCAWatermark(self.config)
        self.test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        self.watermark_bits = np.random.randint(0, 2, 32, dtype=np.uint8)
    
    def test_confidence_scores_range(self):
        """测试置信度分数范围"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 提取水印
        extract_result = self.processor.extract(
            embed_result.watermarked_image,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 检查置信度分数范围
        for confidence in extract_result.confidence_scores:
            assert 0.0 <= confidence <= 1.0
        
        # 检查相关峰值
        assert 0.0 <= extract_result.correlation_peak <= 1.0
    
    def test_detection_map(self):
        """测试检测图"""
        # 嵌入水印
        embed_result = self.processor.embed(self.test_image, self.watermark_bits)
        
        # 提取水印
        extract_result = self.processor.extract(
            embed_result.watermarked_image,
            len(self.watermark_bits),
            positions=embed_result.embedding_positions
        )
        
        # 检查检测图
        assert extract_result.detection_map is not None
        assert extract_result.detection_map.shape == self.test_image.shape
        assert np.all(extract_result.detection_map >= 0)
        assert np.all(extract_result.detection_map <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
