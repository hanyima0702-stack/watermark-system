"""
图像处理工具测试
"""

import pytest
import numpy as np
import cv2
from engines.image.utils.image_utils import (
    bgr_to_yuv,
    yuv_to_bgr,
    calculate_psnr,
    calculate_ssim,
    preprocess_image,
    normalize_image,
    denoise_image,
    clip_image,
    get_y_channel,
    set_y_channel
)


class TestColorSpaceConversion:
    """测试色彩空间转换"""
    
    def test_bgr_to_yuv_basic(self):
        """测试BGR到YUV的基本转换"""
        # 创建测试图像
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # 转换
        yuv = bgr_to_yuv(image)
        
        # 验证
        assert yuv.shape == image.shape
        assert yuv.dtype == np.uint8
    
    def test_yuv_to_bgr_basic(self):
        """测试YUV到BGR的基本转换"""
        # 创建测试图像
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # 转换
        bgr = yuv_to_bgr(image)
        
        # 验证
        assert bgr.shape == image.shape
        assert bgr.dtype == np.uint8
    
    def test_color_space_round_trip(self):
        """测试色彩空间往返转换"""
        # 创建测试图像
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # 往返转换
        yuv = bgr_to_yuv(original)
        recovered = yuv_to_bgr(yuv)
        
        # 验证（允许较大误差，因为色彩空间转换会有精度损失）
        diff = np.abs(original.astype(int) - recovered.astype(int))
        assert np.mean(diff) < 5  # 平均误差应该较小
    
    def test_bgr_to_yuv_invalid_input(self):
        """测试无效输入"""
        with pytest.raises(ValueError):
            bgr_to_yuv(np.array([]))
        
        with pytest.raises(ValueError):
            bgr_to_yuv(np.random.randint(0, 256, (100, 100), dtype=np.uint8))
    
    def test_get_y_channel(self):
        """测试获取Y通道"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        y_channel = get_y_channel(image)
        
        assert y_channel.shape == (100, 100)
        assert y_channel.dtype == np.uint8
    
    def test_set_y_channel(self):
        """测试设置Y通道"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        new_y = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = set_y_channel(image, new_y)
        
        assert result.shape == image.shape
        # 验证Y通道已更新（允许色彩空间转换误差）
        result_y = get_y_channel(result)
        assert np.mean(np.abs(result_y.astype(int) - new_y.astype(int))) < 10


class TestImageQuality:
    """测试图像质量评估"""
    
    def test_psnr_identical_images(self):
        """测试相同图像的PSNR"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        psnr = calculate_psnr(image, image)
        
        assert psnr == float('inf')
    
    def test_psnr_different_images(self):
        """测试不同图像的PSNR"""
        image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        psnr = calculate_psnr(image1, image2)
        
        assert psnr > 0
        assert psnr < 100  # 随机图像的PSNR通常较低
    
    def test_psnr_slight_difference(self):
        """测试轻微差异的PSNR"""
        image1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image2 = image1.copy()
        image2[50, 50, 0] += 10  # 轻微修改
        
        psnr = calculate_psnr(image1, image2)
        
        assert psnr > 40  # 轻微差异应该有较高的PSNR
    
    def test_psnr_size_mismatch(self):
        """测试尺寸不匹配"""
        image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            calculate_psnr(image1, image2)
    
    def test_ssim_identical_images(self):
        """测试相同图像的SSIM"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        ssim = calculate_ssim(image, image)
        
        assert ssim > 0.99  # 相同图像SSIM应接近1
    
    def test_ssim_different_images(self):
        """测试不同图像的SSIM"""
        image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        ssim = calculate_ssim(image1, image2)
        
        assert 0 <= ssim <= 1
    
    def test_ssim_slight_difference(self):
        """测试轻微差异的SSIM"""
        image1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image2 = image1.copy().astype(np.int16)
        image2 += np.random.randint(-5, 6, image2.shape, dtype=np.int16)
        image2 = np.clip(image2, 0, 255).astype(np.uint8)
        
        ssim = calculate_ssim(image1, image2)
        
        assert ssim > 0.8  # 轻微差异应该有较高的SSIM


class TestImagePreprocessing:
    """测试图像预处理"""
    
    def test_preprocess_no_change(self):
        """测试不做任何处理"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        processed = preprocess_image(image)
        
        assert np.array_equal(processed, image)
    
    def test_preprocess_resize(self):
        """测试调整尺寸"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        processed = preprocess_image(image, target_size=(50, 50))
        
        assert processed.shape == (50, 50, 3)
    
    def test_preprocess_normalize(self):
        """测试归一化"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        processed = preprocess_image(image, normalize=True)
        
        assert processed.dtype == float
        assert np.min(processed) >= 0
        assert np.max(processed) <= 1
    
    def test_normalize_image(self):
        """测试图像归一化"""
        image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        normalized = normalize_image(image)
        
        assert normalized.dtype == np.uint8
        assert np.min(normalized) == 0
        assert np.max(normalized) == 255
    
    def test_normalize_constant_image(self):
        """测试常数图像归一化"""
        image = np.ones((100, 100), dtype=np.uint8) * 128
        normalized = normalize_image(image)
        
        assert np.all(normalized == 0)
    
    def test_denoise_gaussian(self):
        """测试高斯去噪"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        denoised = denoise_image(image, method='gaussian', strength=3)
        
        assert denoised.shape == image.shape
    
    def test_denoise_median(self):
        """测试中值去噪"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        denoised = denoise_image(image, method='median', strength=2)
        
        assert denoised.shape == image.shape
    
    def test_denoise_bilateral(self):
        """测试双边去噪"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        denoised = denoise_image(image, method='bilateral', strength=3)
        
        assert denoised.shape == image.shape
    
    def test_denoise_invalid_method(self):
        """测试无效去噪方法"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            denoise_image(image, method='invalid')
    
    def test_clip_image(self):
        """测试图像裁剪"""
        image = np.array([[-10, 50, 300]], dtype=float)
        clipped = clip_image(image, 0, 255)
        
        assert clipped[0, 0] == 0
        assert clipped[0, 1] == 50
        assert clipped[0, 2] == 255


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
