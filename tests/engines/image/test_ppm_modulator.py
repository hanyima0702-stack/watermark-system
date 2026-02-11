"""
PPM调制器单元测试
"""

import pytest
import numpy as np
import cv2
from engines.image.embedding.ppm_modulator import PPMModulator


class TestPPMModulator:
    """PPM调制器测试类"""
    
    def test_initialization(self):
        """测试初始化"""
        modulator = PPMModulator(strength=10)
        assert modulator.strength == 10
        
        modulator_custom = PPMModulator(strength=15)
        assert modulator_custom.strength == 15
    
    def test_modulate_single_bit_one(self):
        """测试调制单个bit=1"""
        modulator = PPMModulator(strength=10)
        
        # 创建100x100的灰度图像，初始值为128
        image = np.full((100, 100), 128, dtype=np.uint8)
        
        # 定义一个像素对位置
        positions = [(10, 10, 20, 20)]
        bits = np.array([1])
        
        # 调制
        modulated = modulator.modulate(image, positions, bits)
        
        # 验证: bit=1时，A增加，B减少
        assert modulated[10, 10] == 128 + 10  # A增加
        assert modulated[20, 20] == 128 - 10  # B减少
    
    def test_modulate_single_bit_zero(self):
        """测试调制单个bit=0"""
        modulator = PPMModulator(strength=10)
        
        image = np.full((100, 100), 128, dtype=np.uint8)
        positions = [(10, 10, 20, 20)]
        bits = np.array([0])
        
        modulated = modulator.modulate(image, positions, bits)
        
        # 验证: bit=0时，A减少，B增加
        assert modulated[10, 10] == 128 - 10  # A减少
        assert modulated[20, 20] == 128 + 10  # B增加
    
    def test_modulate_multiple_bits(self):
        """测试调制多个bits"""
        modulator = PPMModulator(strength=10)
        
        image = np.full((100, 100), 128, dtype=np.uint8)
        positions = [
            (10, 10, 20, 20),
            (30, 30, 40, 40),
            (50, 50, 60, 60)
        ]
        bits = np.array([1, 0, 1])
        
        modulated = modulator.modulate(image, positions, bits)
        
        # 验证第一个bit=1
        assert modulated[10, 10] == 138
        assert modulated[20, 20] == 118
        
        # 验证第二个bit=0
        assert modulated[30, 30] == 118
        assert modulated[40, 40] == 138
        
        # 验证第三个bit=1
        assert modulated[50, 50] == 138
        assert modulated[60, 60] == 118
    
    def test_modulate_clipping(self):
        """测试亮度值裁剪到[0, 255]范围"""
        modulator = PPMModulator(strength=20)
        
        # 创建接近边界的图像
        image = np.full((100, 100), 250, dtype=np.uint8)
        positions = [(10, 10, 20, 20)]
        bits = np.array([1])
        
        modulated = modulator.modulate(image, positions, bits)
        
        # 验证: 250+20=270应该被裁剪到255
        assert modulated[10, 10] == 255
        assert modulated[20, 20] == 230
        
        # 测试下边界
        image = np.full((100, 100), 10, dtype=np.uint8)
        modulated = modulator.modulate(image, positions, bits)
        
        # 验证: 10-20=-10应该被裁剪到0
        assert modulated[20, 20] == 0
    
    def test_demodulate_single_bit(self):
        """测试解调单个bit"""
        modulator = PPMModulator(strength=10)
        
        # 创建已调制的图像
        image = np.full((100, 100), 128, dtype=np.uint8)
        positions = [(10, 10, 20, 20)]
        bits_original = np.array([1])
        
        modulated = modulator.modulate(image, positions, bits_original)
        
        # 解调
        bits_decoded, confidences = modulator.demodulate(modulated, positions)
        
        # 验证解调结果
        assert bits_decoded[0] == 1
        assert confidences[0] > 0.5  # 置信度应该较高
    
    def test_demodulate_multiple_bits(self):
        """测试解调多个bits"""
        modulator = PPMModulator(strength=10)
        
        image = np.full((100, 100), 128, dtype=np.uint8)
        positions = [
            (10, 10, 20, 20),
            (30, 30, 40, 40),
            (50, 50, 60, 60),
            (70, 70, 80, 80)
        ]
        bits_original = np.array([1, 0, 1, 0])
        
        modulated = modulator.modulate(image, positions, bits_original)
        bits_decoded, confidences = modulator.demodulate(modulated, positions)
        
        # 验证所有bits都正确解调
        np.testing.assert_array_equal(bits_decoded, bits_original)
        
        # 验证置信度都较高
        assert np.all(confidences > 0.5)
    
    def test_modulate_demodulate_roundtrip(self):
        """测试调制-解调往返的准确性"""
        modulator = PPMModulator(strength=10)
        
        image = np.full((100, 100), 128, dtype=np.uint8)
        
        # 生成随机bits
        np.random.seed(42)
        bits_original = np.random.randint(0, 2, size=20)
        
        # 生成随机位置
        positions = []
        for i in range(20):
            x1, y1 = i * 4, i * 4
            x2, y2 = i * 4 + 2, i * 4 + 2
            positions.append((x1, y1, x2, y2))
        
        # 调制
        modulated = modulator.modulate(image, positions, bits_original)
        
        # 解调
        bits_decoded, confidences = modulator.demodulate(modulated, positions)
        
        # 验证完全一致
        np.testing.assert_array_equal(bits_decoded, bits_original)
        assert np.all(confidences > 0.8)
    
    def test_different_strength_parameters(self):
        """测试不同强度参数的效果"""
        image = np.full((100, 100), 128, dtype=np.uint8)
        positions = [(10, 10, 20, 20)]
        bits = np.array([1])
        
        # 测试不同强度
        for strength in [5, 10, 15, 20]:
            modulator = PPMModulator(strength=strength)
            modulated = modulator.modulate(image, positions, bits)
            bits_decoded, confidences = modulator.demodulate(modulated, positions)
            
            # 验证解调正确
            assert bits_decoded[0] == 1
            
            # 强度越大，置信度应该越高（在不裁剪的情况下）
            if strength <= 15:
                expected_confidence = min(2 * strength / (2 * strength), 1.0)
                assert abs(confidences[0] - expected_confidence) < 0.1
    
    def test_yuv_color_space_conversion(self):
        """测试YUV色彩空间转换"""
        modulator = PPMModulator(strength=10)
        
        # 创建彩色图像 (BGR格式)
        bgr_image = np.full((100, 100, 3), [100, 150, 200], dtype=np.uint8)
        
        # 转换到YUV
        yuv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV)
        y_channel = yuv_image[:, :, 0]
        
        # 在Y通道上调制
        positions = [(10, 10, 20, 20)]
        bits = np.array([1])
        
        y_modulated = modulator.modulate(y_channel, positions, bits)
        
        # 验证调制效果
        assert y_modulated[10, 10] > y_channel[10, 10]
        assert y_modulated[20, 20] < y_channel[20, 20]
        
        # 将调制后的Y通道放回YUV图像
        yuv_modulated = yuv_image.copy()
        yuv_modulated[:, :, 0] = y_modulated
        
        # 转换回BGR
        bgr_modulated = cv2.cvtColor(yuv_modulated, cv2.COLOR_YUV2BGR)
        
        # 验证图像仍然有效
        assert bgr_modulated.shape == bgr_image.shape
        assert bgr_modulated.dtype == np.uint8
    
    def test_boundary_pixel_handling(self):
        """测试边界像素处理"""
        modulator = PPMModulator(strength=10)
        
        image = np.full((50, 50), 128, dtype=np.uint8)
        
        # 测试边界位置
        positions = [
            (0, 0, 1, 1),      # 左上角
            (49, 0, 48, 1),    # 右上角
            (0, 49, 1, 48),    # 左下角
            (49, 49, 48, 48),  # 右下角
        ]
        bits = np.array([1, 0, 1, 0])
        
        # 调制应该成功
        modulated = modulator.modulate(image, positions, bits)
        
        # 解调应该成功
        bits_decoded, confidences = modulator.demodulate(modulated, positions)
        
        # 验证结果
        np.testing.assert_array_equal(bits_decoded, bits)
    
    def test_invalid_positions(self):
        """测试无效位置的处理"""
        modulator = PPMModulator(strength=10)
        
        image = np.full((50, 50), 128, dtype=np.uint8)
        
        # 包含超出边界的位置
        positions = [
            (10, 10, 20, 20),  # 有效
            (60, 60, 70, 70),  # 无效：超出边界
            (30, 30, 40, 40),  # 有效
        ]
        bits = np.array([1, 0, 1])
        
        # 调制应该跳过无效位置
        modulated = modulator.modulate(image, positions, bits)
        
        # 验证有效位置被调制
        assert modulated[10, 10] != 128
        assert modulated[30, 30] != 128
        
        # 解调
        bits_decoded, confidences = modulator.demodulate(modulated, positions)
        
        # 无效位置的置信度应该为0
        assert confidences[1] == 0.0
    
    def test_mismatched_positions_and_bits(self):
        """测试位置和bits数量不匹配的情况"""
        modulator = PPMModulator(strength=10)
        
        image = np.full((100, 100), 128, dtype=np.uint8)
        positions = [(10, 10, 20, 20)]
        bits = np.array([1, 0])  # 数量不匹配
        
        # 应该抛出ValueError
        with pytest.raises(ValueError, match="位置数量.*与bit数量.*不匹配"):
            modulator.modulate(image, positions, bits)
    
    def test_confidence_calculation(self):
        """测试置信度计算"""
        modulator = PPMModulator(strength=10)
        
        # 创建具有不同差值的图像
        image = np.full((100, 100), 128, dtype=np.uint8)
        
        # 手动设置像素对，模拟不同的信号强度
        # 置信度 = |diff| / (2 * strength) = |diff| / 20
        test_cases = [
            ((10, 10, 20, 20), 20, 1.0),   # 理想情况: diff=20, confidence=20/20=1.0
            ((30, 30, 40, 40), 10, 0.5),   # 弱信号: diff=10, confidence=10/20=0.5
            ((50, 50, 60, 60), 4, 0.2),    # 更弱: diff=4, confidence=4/20=0.2
        ]
        
        for (x1, y1, x2, y2), diff, expected_conf in test_cases:
            test_image = image.copy()
            test_image[y1, x1] = 128 + diff
            test_image[y2, x2] = 128
            
            bits, confidences = modulator.demodulate(test_image, [(x1, y1, x2, y2)])
            
            assert abs(confidences[0] - expected_conf) < 0.01
    
    def test_zero_difference_handling(self):
        """测试像素对差值为0的情况"""
        modulator = PPMModulator(strength=10)
        
        # 创建像素对差值为0的图像
        image = np.full((100, 100), 128, dtype=np.uint8)
        positions = [(10, 10, 20, 20)]
        
        # 解调
        bits, confidences = modulator.demodulate(image, positions)
        
        # 差值为0时，bit应该为0，置信度为0
        assert bits[0] == 0
        assert confidences[0] == 0.0
    
    def test_noise_robustness(self):
        """测试对噪声的鲁棒性"""
        modulator = PPMModulator(strength=15)
        
        image = np.full((100, 100), 128, dtype=np.uint8)
        positions = [(i*5, i*5, i*5+2, i*5+2) for i in range(10)]
        bits_original = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        
        # 调制
        modulated = modulator.modulate(image, positions, bits_original)
        
        # 添加轻微噪声
        noise = np.random.normal(0, 3, modulated.shape).astype(np.int16)
        noisy_image = np.clip(modulated.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 解调
        bits_decoded, confidences = modulator.demodulate(noisy_image, positions)
        
        # 在轻微噪声下，大部分bits应该仍然正确
        accuracy = np.sum(bits_decoded == bits_original) / len(bits_original)
        assert accuracy >= 0.8  # 至少80%准确率
