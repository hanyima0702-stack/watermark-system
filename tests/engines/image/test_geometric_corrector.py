"""
几何校正器单元测试

测试GeometricCorrector类的功能。
"""

import pytest
import numpy as np
import cv2
from engines.image.extraction.geometric_corrector import GeometricCorrector


class TestGeometricCorrector:
    """几何校正器测试类"""
    
    @pytest.fixture
    def corrector(self):
        """创建几何校正器实例"""
        return GeometricCorrector()
    
    @pytest.fixture
    def test_image(self):
        """创建测试图像：带有明显特征的图像"""
        # 创建一个512x512的测试图像，包含一些几何特征
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 绘制一个矩形
        cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), 2)
        
        # 绘制对角线
        cv2.line(image, (100, 100), (400, 400), (255, 0, 0), 2)
        
        # 绘制一个圆
        cv2.circle(image, (256, 256), 50, (0, 255, 0), 2)
        
        # 添加一些文字标记
        cv2.putText(image, "TOP", (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image
    
    def test_create_transform_matrix_no_transform(self, corrector):
        """测试无变换时的变换矩阵"""
        center = (256, 256)
        matrix = corrector.create_transform_matrix(
            rotation=0.0,
            scale=1.0,
            center=center
        )
        
        # 验证矩阵形状
        assert matrix.shape == (2, 3)
        
        # 无变换时应该接近单位矩阵
        expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(matrix, expected, decimal=5)
    
    def test_create_transform_matrix_rotation_only(self, corrector):
        """测试仅旋转的变换矩阵"""
        center = (256, 256)
        rotation = 45.0  # 检测到顺时针45度
        
        matrix = corrector.create_transform_matrix(
            rotation=rotation,
            scale=1.0,
            center=center
        )
        
        # 验证矩阵形状
        assert matrix.shape == (2, 3)
        
        # 验证矩阵不是单位矩阵
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        assert not np.allclose(matrix, identity)
    
    def test_create_transform_matrix_scale_only(self, corrector):
        """测试仅缩放的变换矩阵"""
        center = (256, 256)
        scale = 1.5  # 检测到放大1.5倍
        
        matrix = corrector.create_transform_matrix(
            rotation=0.0,
            scale=scale,
            center=center
        )
        
        # 验证矩阵形状
        assert matrix.shape == (2, 3)
        
        # 缩放矩阵的对角元素应该是1/scale
        expected_scale = 1.0 / scale
        assert abs(matrix[0, 0] - expected_scale) < 0.001
        assert abs(matrix[1, 1] - expected_scale) < 0.001
    
    def test_correct_no_transform(self, corrector, test_image):
        """测试无变换时的校正"""
        corrected = corrector.correct(
            image=test_image,
            rotation=0.0,
            scale=1.0
        )
        
        # 验证输出形状
        assert corrected.shape == test_image.shape
        
        # 无变换时图像应该基本不变
        diff = np.abs(corrected.astype(float) - test_image.astype(float))
        mean_diff = np.mean(diff)
        assert mean_diff < 1.0  # 允许微小的插值误差
    
    def test_correct_rotation_accuracy(self, corrector, test_image):
        """测试旋转校正的准确性"""
        # 先旋转图像
        center = (test_image.shape[1] // 2, test_image.shape[0] // 2)
        rotation_angle = 15.0  # 顺时针旋转15度
        
        # 创建旋转矩阵并应用
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(test_image, M, (test_image.shape[1], test_image.shape[0]))
        
        # 使用校正器校正回来
        corrected = corrector.correct(
            image=rotated,
            rotation=rotation_angle,  # 告诉校正器检测到的旋转角度
            scale=1.0,
            center=center
        )
        
        # 验证校正后的图像与原图相似
        # 计算中心区域的相似度（边缘可能有插值误差）
        h, w = test_image.shape[:2]
        margin = 50
        original_center = test_image[margin:h-margin, margin:w-margin]
        corrected_center = corrected[margin:h-margin, margin:w-margin]
        
        # 计算PSNR
        mse = np.mean((original_center.astype(float) - corrected_center.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0 ** 2 / mse)
            assert psnr > 25  # PSNR应该大于25dB，表示校正效果良好
    
    def test_correct_scale_accuracy(self, corrector, test_image):
        """测试缩放校正的准确性"""
        # 先缩放图像
        scale_factor = 1.5
        h, w = test_image.shape[:2]
        center = (w // 2, h // 2)
        
        # 创建缩放矩阵并应用
        M = cv2.getRotationMatrix2D(center, 0, scale_factor)
        scaled = cv2.warpAffine(test_image, M, (w, h))
        
        # 使用校正器校正回来
        corrected = corrector.correct(
            image=scaled,
            rotation=0.0,
            scale=scale_factor,
            center=center
        )
        
        # 验证校正后的图像与原图相似
        # 计算中心区域的相似度
        margin = 80
        original_center = test_image[margin:h-margin, margin:w-margin]
        corrected_center = corrected[margin:h-margin, margin:w-margin]
        
        # 计算相似度
        mse = np.mean((original_center.astype(float) - corrected_center.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0 ** 2 / mse)
            assert psnr > 20  # PSNR应该大于20dB
    
    def test_correct_combined_transform(self, corrector, test_image):
        """测试组合变换（旋转+缩放）的校正"""
        rotation_angle = 30.0
        scale_factor = 1.2
        h, w = test_image.shape[:2]
        center = (w // 2, h // 2)
        
        # 应用组合变换
        M = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
        transformed = cv2.warpAffine(test_image, M, (w, h))
        
        # 使用校正器校正
        corrected = corrector.correct(
            image=transformed,
            rotation=rotation_angle,
            scale=scale_factor,
            center=center
        )
        
        # 验证校正效果
        margin = 100
        original_center = test_image[margin:h-margin, margin:w-margin]
        corrected_center = corrected[margin:h-margin, margin:w-margin]
        
        mse = np.mean((original_center.astype(float) - corrected_center.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0 ** 2 / mse)
            assert psnr > 18  # 组合变换的校正难度更高，降低阈值
    
    def test_correct_with_custom_center(self, corrector, test_image):
        """测试使用自定义旋转中心的校正"""
        rotation_angle = 20.0
        custom_center = (100, 100)  # 非图像中心
        
        # 应用旋转
        M = cv2.getRotationMatrix2D(custom_center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(test_image, M, (test_image.shape[1], test_image.shape[0]))
        
        # 使用相同的中心点校正
        corrected = corrector.correct(
            image=rotated,
            rotation=rotation_angle,
            scale=1.0,
            center=custom_center
        )
        
        # 验证输出形状
        assert corrected.shape == test_image.shape
    
    def test_correct_grayscale_image(self, corrector):
        """测试灰度图像的校正"""
        # 创建灰度测试图像
        gray_image = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(gray_image, (50, 50), (200, 200), 255, 2)
        cv2.circle(gray_image, (128, 128), 30, 255, 2)
        
        # 旋转
        center = (128, 128)
        M = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated = cv2.warpAffine(gray_image, M, (256, 256))
        
        # 校正
        corrected = corrector.correct(
            image=rotated,
            rotation=45.0,
            scale=1.0,
            center=center
        )
        
        # 验证输出
        assert corrected.shape == gray_image.shape
        assert len(corrected.shape) == 2  # 灰度图
    
    def test_correct_invalid_input(self, corrector):
        """测试无效输入的处理"""
        with pytest.raises(ValueError):
            corrector.correct(
                image=None,
                rotation=0.0,
                scale=1.0
            )
        
        with pytest.raises(ValueError):
            corrector.correct(
                image=np.array([]),
                rotation=0.0,
                scale=1.0
            )
    
    def test_correct_large_rotation(self, corrector, test_image):
        """测试大角度旋转的校正"""
        # 测试90度旋转
        rotation_angle = 90.0
        center = (test_image.shape[1] // 2, test_image.shape[0] // 2)
        
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(test_image, M, (test_image.shape[1], test_image.shape[0]))
        
        corrected = corrector.correct(
            image=rotated,
            rotation=rotation_angle,
            scale=1.0,
            center=center
        )
        
        # 验证输出形状
        assert corrected.shape == test_image.shape
    
    def test_correct_extreme_scale(self, corrector, test_image):
        """测试极端缩放的校正"""
        # 测试缩小到50%
        scale_factor = 0.5
        center = (test_image.shape[1] // 2, test_image.shape[0] // 2)
        
        M = cv2.getRotationMatrix2D(center, 0, scale_factor)
        scaled = cv2.warpAffine(test_image, M, (test_image.shape[1], test_image.shape[0]))
        
        corrected = corrector.correct(
            image=scaled,
            rotation=0.0,
            scale=scale_factor,
            center=center
        )
        
        # 验证输出形状
        assert corrected.shape == test_image.shape
    
    def test_correct_negative_rotation(self, corrector, test_image):
        """测试负角度旋转的校正"""
        rotation_angle = -25.0  # 逆时针旋转
        center = (test_image.shape[1] // 2, test_image.shape[0] // 2)
        
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(test_image, M, (test_image.shape[1], test_image.shape[0]))
        
        corrected = corrector.correct(
            image=rotated,
            rotation=rotation_angle,
            scale=1.0,
            center=center
        )
        
        # 验证输出形状
        assert corrected.shape == test_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
