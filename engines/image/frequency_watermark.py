"""
频域暗水印处理器 - 实现基于DCT和DWT的频域水印嵌入与提取
支持抗压缩、抗噪声的鲁棒性水印算法
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import pywt
from scipy.fftpack import dct, idct
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FrequencyMethod(Enum):
    """频域变换方法枚举"""
    DCT = "dct"
    DWT = "dwt"


class WaveletType(Enum):
    """小波类型枚举"""
    HAAR = "haar"
    DAUBECHIES = "db4"
    BIORTHOGONAL = "bior2.2"
    COIFLETS = "coif2"


@dataclass
class FrequencyWatermarkConfig:
    """频域水印配置"""
    method: FrequencyMethod = FrequencyMethod.DCT
    strength: float = 0.1
    block_size: int = 8
    wavelet_type: WaveletType = WaveletType.HAAR
    decomposition_levels: int = 3
    frequency_band: str = "mid"  # low, mid, high
    robustness_level: str = "medium"  # low, medium, high


@dataclass
class WatermarkEmbedResult:
    """水印嵌入结果"""
    watermarked_image: np.ndarray
    psnr: float
    ssim: float
    embedding_strength: float
    metadata: Dict[str, Any]


@dataclass
class WatermarkExtractResult:
    """水印提取结果"""
    extracted_bits: np.ndarray
    confidence_score: float
    correlation: float
    extraction_method: str
    metadata: Dict[str, Any]


class DCTWatermark:
    """基于DCT变换的频域水印处理器"""
    
    def __init__(self, config: FrequencyWatermarkConfig):
        self.config = config
        self.block_size = config.block_size
        
    def embed(self, image: np.ndarray, watermark_bits: np.ndarray) -> WatermarkEmbedResult:
        """
        嵌入DCT频域水印
        
        Args:
            image: 原始图像 (H, W, C) 或 (H, W)
            watermark_bits: 水印比特序列
            
        Returns:
            水印嵌入结果
        """
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image.copy()
            
            # 确保图像尺寸是块大小的倍数
            h, w = gray_image.shape
            h_pad = (self.block_size - h % self.block_size) % self.block_size
            w_pad = (self.block_size - w % self.block_size) % self.block_size
            
            if h_pad > 0 or w_pad > 0:
                gray_image = np.pad(gray_image, ((0, h_pad), (0, w_pad)), mode='reflect')
            
            # 转换为浮点数
            float_image = gray_image.astype(np.float32)
            
            # 分块DCT变换和水印嵌入
            watermarked = self._embed_dct_blocks(float_image, watermark_bits)
            
            # 裁剪回原始尺寸
            if h_pad > 0 or w_pad > 0:
                watermarked = watermarked[:h, :w]
            
            # 转换回uint8
            watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
            
            # 如果原图是彩色的，需要重新组合
            if len(image.shape) == 3:
                result_image = image.copy()
                result_image[:, :, 0] = watermarked  # 只在Y通道嵌入水印
            else:
                result_image = watermarked
            
            # 计算质量指标
            psnr = self._calculate_psnr(gray_image[:h, :w], watermarked)
            ssim = self._calculate_ssim(gray_image[:h, :w], watermarked)
            
            return WatermarkEmbedResult(
                watermarked_image=result_image,
                psnr=psnr,
                ssim=ssim,
                embedding_strength=self.config.strength,
                metadata={
                    "method": "DCT",
                    "block_size": self.block_size,
                    "watermark_length": len(watermark_bits)
                }
            )
            
        except Exception as e:
            logger.error(f"DCT watermark embedding failed: {e}")
            raise
    
    def extract(self, watermarked_image: np.ndarray, watermark_length: int) -> WatermarkExtractResult:
        """
        提取DCT频域水印
        
        Args:
            watermarked_image: 含水印图像
            watermark_length: 水印长度
            
        Returns:
            水印提取结果
        """
        try:
            # 转换为灰度图像
            if len(watermarked_image.shape) == 3:
                gray_image = cv2.cvtColor(watermarked_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = watermarked_image.copy()
            
            # 确保图像尺寸是块大小的倍数
            h, w = gray_image.shape
            h_pad = (self.block_size - h % self.block_size) % self.block_size
            w_pad = (self.block_size - w % self.block_size) % self.block_size
            
            if h_pad > 0 or w_pad > 0:
                gray_image = np.pad(gray_image, ((0, h_pad), (0, w_pad)), mode='reflect')
            
            # 转换为浮点数
            float_image = gray_image.astype(np.float32)
            
            # 分块DCT变换和水印提取
            extracted_bits, correlation = self._extract_dct_blocks(float_image, watermark_length)
            
            # 计算置信度
            confidence = self._calculate_confidence(correlation)
            
            return WatermarkExtractResult(
                extracted_bits=extracted_bits,
                confidence_score=confidence,
                correlation=correlation,
                extraction_method="DCT",
                metadata={
                    "block_size": self.block_size,
                    "extracted_length": len(extracted_bits)
                }
            )
            
        except Exception as e:
            logger.error(f"DCT watermark extraction failed: {e}")
            raise
    
    def _embed_dct_blocks(self, image: np.ndarray, watermark_bits: np.ndarray) -> np.ndarray:
        """在DCT块中嵌入水印"""
        h, w = image.shape
        watermarked = image.copy()
        
        bit_index = 0
        total_bits = len(watermark_bits)
        
        # 遍历所有8x8块
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                if bit_index >= total_bits:
                    break
                
                # 提取块
                block = image[i:i+self.block_size, j:j+self.block_size]
                
                # DCT变换
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # 在中频系数中嵌入水印
                if self.config.frequency_band == "mid":
                    # 选择中频系数位置 (通常是(3,4)或(4,3))
                    coeff_pos = [(2, 3), (3, 2), (1, 4), (4, 1)]
                elif self.config.frequency_band == "low":
                    coeff_pos = [(1, 1), (1, 2), (2, 1)]
                else:  # high
                    coeff_pos = [(5, 6), (6, 5), (4, 7), (7, 4)]
                
                # 嵌入一个比特
                if bit_index < total_bits:
                    bit = watermark_bits[bit_index]
                    pos = coeff_pos[bit_index % len(coeff_pos)]
                    
                    if pos[0] < self.block_size and pos[1] < self.block_size:
                        # 量化索引调制 (QIM)
                        delta = self.config.strength * 50  # 量化步长
                        coeff = dct_block[pos[0], pos[1]]
                        
                        if bit == 1:
                            # 调制到奇数量化区间
                            dct_block[pos[0], pos[1]] = delta * (2 * np.floor(coeff / (2 * delta)) + 1)
                        else:
                            # 调制到偶数量化区间
                            dct_block[pos[0], pos[1]] = delta * (2 * np.floor(coeff / (2 * delta)))
                    
                    bit_index += 1
                
                # 逆DCT变换
                reconstructed_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                watermarked[i:i+self.block_size, j:j+self.block_size] = reconstructed_block
                
                if bit_index >= total_bits:
                    break
        
        return watermarked
    
    def _extract_dct_blocks(self, image: np.ndarray, watermark_length: int) -> Tuple[np.ndarray, float]:
        """从DCT块中提取水印"""
        h, w = image.shape
        extracted_bits = []
        correlations = []
        
        bit_index = 0
        
        # 遍历所有8x8块
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                if bit_index >= watermark_length:
                    break
                
                # 提取块
                block = image[i:i+self.block_size, j:j+self.block_size]
                
                # DCT变换
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # 选择系数位置
                if self.config.frequency_band == "mid":
                    coeff_pos = [(2, 3), (3, 2), (1, 4), (4, 1)]
                elif self.config.frequency_band == "low":
                    coeff_pos = [(1, 1), (1, 2), (2, 1)]
                else:  # high
                    coeff_pos = [(5, 6), (6, 5), (4, 7), (7, 4)]
                
                # 提取一个比特
                if bit_index < watermark_length:
                    pos = coeff_pos[bit_index % len(coeff_pos)]
                    
                    if pos[0] < self.block_size and pos[1] < self.block_size:
                        delta = self.config.strength * 50
                        coeff = dct_block[pos[0], pos[1]]
                        
                        # 量化索引解调
                        quantized = np.round(coeff / delta)
                        bit = int(quantized % 2)
                        extracted_bits.append(bit)
                        
                        # 计算相关性
                        correlation = abs(coeff % (2 * delta) - delta) / delta
                        correlations.append(correlation)
                    
                    bit_index += 1
                
                if bit_index >= watermark_length:
                    break
        
        # 计算平均相关性
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return np.array(extracted_bits), avg_correlation


class DWTWatermark:
    """基于DWT小波变换的频域水印处理器"""
    
    def __init__(self, config: FrequencyWatermarkConfig):
        self.config = config
        self.wavelet = config.wavelet_type.value
        self.levels = config.decomposition_levels
        
    def embed(self, image: np.ndarray, watermark_bits: np.ndarray) -> WatermarkEmbedResult:
        """
        嵌入DWT频域水印
        
        Args:
            image: 原始图像
            watermark_bits: 水印比特序列
            
        Returns:
            水印嵌入结果
        """
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image.copy()
            
            # 转换为浮点数
            float_image = gray_image.astype(np.float32) / 255.0
            
            # 多级小波分解
            coeffs = pywt.wavedec2(float_image, self.wavelet, level=self.levels)
            
            # 在指定频带嵌入水印
            modified_coeffs = self._embed_dwt_coeffs(coeffs, watermark_bits)
            
            # 小波重构
            watermarked = pywt.waverec2(modified_coeffs, self.wavelet)
            
            # 转换回uint8
            watermarked = np.clip(watermarked * 255, 0, 255).astype(np.uint8)
            
            # 如果原图是彩色的，需要重新组合
            if len(image.shape) == 3:
                result_image = image.copy()
                result_image[:, :, 0] = watermarked
            else:
                result_image = watermarked
            
            # 计算质量指标
            psnr = self._calculate_psnr(gray_image, watermarked)
            ssim = self._calculate_ssim(gray_image, watermarked)
            
            return WatermarkEmbedResult(
                watermarked_image=result_image,
                psnr=psnr,
                ssim=ssim,
                embedding_strength=self.config.strength,
                metadata={
                    "method": "DWT",
                    "wavelet": self.wavelet,
                    "levels": self.levels,
                    "watermark_length": len(watermark_bits)
                }
            )
            
        except Exception as e:
            logger.error(f"DWT watermark embedding failed: {e}")
            raise
    
    def extract(self, watermarked_image: np.ndarray, watermark_length: int) -> WatermarkExtractResult:
        """
        提取DWT频域水印
        
        Args:
            watermarked_image: 含水印图像
            watermark_length: 水印长度
            
        Returns:
            水印提取结果
        """
        try:
            # 转换为灰度图像
            if len(watermarked_image.shape) == 3:
                gray_image = cv2.cvtColor(watermarked_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = watermarked_image.copy()
            
            # 转换为浮点数
            float_image = gray_image.astype(np.float32) / 255.0
            
            # 多级小波分解
            coeffs = pywt.wavedec2(float_image, self.wavelet, level=self.levels)
            
            # 从指定频带提取水印
            extracted_bits, correlation = self._extract_dwt_coeffs(coeffs, watermark_length)
            
            # 计算置信度
            confidence = self._calculate_confidence(correlation)
            
            return WatermarkExtractResult(
                extracted_bits=extracted_bits,
                confidence_score=confidence,
                correlation=correlation,
                extraction_method="DWT",
                metadata={
                    "wavelet": self.wavelet,
                    "levels": self.levels,
                    "extracted_length": len(extracted_bits)
                }
            )
            
        except Exception as e:
            logger.error(f"DWT watermark extraction failed: {e}")
            raise
    
    def _embed_dwt_coeffs(self, coeffs, watermark_bits: np.ndarray):
        """在小波系数中嵌入水印"""
        modified_coeffs = list(coeffs)
        
        # 选择嵌入的频带
        if self.config.frequency_band == "low":
            # 在低频系数中嵌入（LL子带）
            target_coeffs = modified_coeffs[0]
        elif self.config.frequency_band == "high":
            # 在高频系数中嵌入（HH子带）
            target_coeffs = modified_coeffs[1][2]  # HH of level 1
        else:  # mid
            # 在中频系数中嵌入（LH或HL子带）
            target_coeffs = modified_coeffs[1][0]  # LH of level 1
        
        # 嵌入水印比特
        h, w = target_coeffs.shape
        total_positions = h * w
        watermark_length = len(watermark_bits)
        
        if watermark_length > total_positions:
            raise ValueError(f"Watermark too long: {watermark_length} > {total_positions}")
        
        # 生成伪随机位置序列
        np.random.seed(42)  # 固定种子确保可重复性
        positions = np.random.choice(total_positions, watermark_length, replace=False)
        
        flat_coeffs = target_coeffs.flatten()
        
        for i, pos in enumerate(positions):
            bit = watermark_bits[i]
            coeff = flat_coeffs[pos]
            
            # 使用量化索引调制
            delta = self.config.strength
            if bit == 1:
                flat_coeffs[pos] = coeff + delta * np.sign(coeff)
            else:
                flat_coeffs[pos] = coeff - delta * np.sign(coeff)
        
        # 重新整形并更新系数
        if self.config.frequency_band == "low":
            modified_coeffs[0] = flat_coeffs.reshape(h, w)
        elif self.config.frequency_band == "high":
            modified_coeffs[1] = (modified_coeffs[1][0], modified_coeffs[1][1], flat_coeffs.reshape(h, w))
        else:  # mid
            modified_coeffs[1] = (flat_coeffs.reshape(h, w), modified_coeffs[1][1], modified_coeffs[1][2])
        
        return modified_coeffs
    
    def _extract_dwt_coeffs(self, coeffs, watermark_length: int) -> Tuple[np.ndarray, float]:
        """从小波系数中提取水印"""
        # 选择提取的频带
        if self.config.frequency_band == "low":
            target_coeffs = coeffs[0]
        elif self.config.frequency_band == "high":
            target_coeffs = coeffs[1][2]  # HH of level 1
        else:  # mid
            target_coeffs = coeffs[1][0]  # LH of level 1
        
        h, w = target_coeffs.shape
        
        # 生成相同的伪随机位置序列
        np.random.seed(42)
        total_positions = h * w
        positions = np.random.choice(total_positions, watermark_length, replace=False)
        
        flat_coeffs = target_coeffs.flatten()
        extracted_bits = []
        correlations = []
        
        for pos in positions:
            coeff = flat_coeffs[pos]
            
            # 基于系数符号和幅度提取比特
            if abs(coeff) > self.config.strength / 2:
                bit = 1 if coeff > 0 else 0
                correlation = abs(coeff) / (abs(coeff) + self.config.strength)
            else:
                bit = 0
                correlation = 0.5
            
            extracted_bits.append(bit)
            correlations.append(correlation)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return np.array(extracted_bits), avg_correlation
    
    def _calculate_psnr(self, original: np.ndarray, watermarked: np.ndarray) -> float:
        """计算PSNR"""
        mse = np.mean((original.astype(np.float32) - watermarked.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def _calculate_ssim(self, original: np.ndarray, watermarked: np.ndarray) -> float:
        """计算SSIM (简化版本)"""
        # 简化的SSIM计算
        mu1 = np.mean(original)
        mu2 = np.mean(watermarked)
        sigma1 = np.var(original)
        sigma2 = np.var(watermarked)
        sigma12 = np.mean((original - mu1) * (watermarked - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    def _calculate_confidence(self, correlation: float) -> float:
        """计算置信度"""
        # 基于相关性计算置信度
        return min(1.0, max(0.0, correlation))


class FrequencyWatermarkProcessor:
    """频域水印处理器统一接口"""
    
    def __init__(self, config: FrequencyWatermarkConfig):
        self.config = config
        
        if config.method == FrequencyMethod.DCT:
            self.processor = DCTWatermark(config)
        elif config.method == FrequencyMethod.DWT:
            self.processor = DWTWatermark(config)
        else:
            raise ValueError(f"Unsupported frequency method: {config.method}")
    
    def embed_watermark(self, image: np.ndarray, watermark_bits: np.ndarray) -> WatermarkEmbedResult:
        """嵌入频域水印"""
        return self.processor.embed(image, watermark_bits)
    
    def extract_watermark(self, watermarked_image: np.ndarray, watermark_length: int) -> WatermarkExtractResult:
        """提取频域水印"""
        return self.processor.extract(watermarked_image, watermark_length)
    
    def _calculate_psnr(self, original: np.ndarray, watermarked: np.ndarray) -> float:
        """计算PSNR"""
        mse = np.mean((original.astype(np.float32) - watermarked.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def _calculate_ssim(self, original: np.ndarray, watermarked: np.ndarray) -> float:
        """计算SSIM (简化版本)"""
        mu1 = np.mean(original)
        mu2 = np.mean(watermarked)
        sigma1 = np.var(original)
        sigma2 = np.var(watermarked)
        sigma12 = np.mean((original - mu1) * (watermarked - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    def _calculate_confidence(self, correlation: float) -> float:
        """计算置信度"""
        return min(1.0, max(0.0, correlation))


# 为了兼容性，添加一些辅助函数
DCTWatermark._calculate_psnr = FrequencyWatermarkProcessor._calculate_psnr
DCTWatermark._calculate_ssim = FrequencyWatermarkProcessor._calculate_ssim
DCTWatermark._calculate_confidence = FrequencyWatermarkProcessor._calculate_confidence