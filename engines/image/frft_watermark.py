"""
FrFT分数阶傅里叶变换暗水印处理器
实现基于分数阶傅里叶变换的高隐蔽性水印算法
支持盲检测提取和局部区域水印检测
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.special import factorial
from scipy.linalg import expm
import math

logger = logging.getLogger(__name__)


@dataclass
class FrFTWatermarkConfig:
    """FrFT水印配置"""
    alpha: float = 0.5  # 旋转角度参数 (0-1)
    strength: float = 0.1  # 嵌入强度
    block_size: int = 32  # 块大小
    overlap_ratio: float = 0.5  # 重叠比例
    detection_threshold: float = 0.6  # 检测阈值
    robustness_level: str = "high"  # 鲁棒性级别


@dataclass
class FrFTEmbedResult:
    """FrFT水印嵌入结果"""
    watermarked_image: np.ndarray
    psnr: float
    ssim: float
    embedding_positions: List[Tuple[int, int]]
    alpha_parameter: float
    metadata: Dict[str, Any]


@dataclass
class FrFTExtractResult:
    """FrFT水印提取结果"""
    extracted_bits: np.ndarray
    confidence_scores: List[float]
    detection_map: np.ndarray
    correlation_peak: float
    extraction_positions: List[Tuple[int, int]]
    metadata: Dict[str, Any]


class FrFTProcessor:
    """分数阶傅里叶变换处理器"""
    
    def __init__(self):
        self._kernel_cache = {}  # 缓存计算的核矩阵
    
    def frft(self, signal: np.ndarray, alpha: float) -> np.ndarray:
        """
        计算一维分数阶傅里叶变换
        
        Args:
            signal: 输入信号
            alpha: 旋转角度参数 (0-1)
            
        Returns:
            变换后的信号
        """
        N = len(signal)
        
        # 特殊情况处理
        if alpha == 0:
            return signal.copy()
        elif alpha == 1:
            return np.fft.fft(signal)
        elif alpha == -1:
            return np.fft.ifft(signal)
        elif alpha == 0.5:
            return self._chirp_z_transform(signal)
        
        # 一般情况的FrFT计算
        return self._general_frft(signal, alpha)
    
    def frft2d(self, image: np.ndarray, alpha: float) -> np.ndarray:
        """
        计算二维分数阶傅里叶变换
        
        Args:
            image: 输入图像
            alpha: 旋转角度参数
            
        Returns:
            变换后的图像
        """
        # 先对行进行FrFT
        rows_transformed = np.zeros_like(image, dtype=complex)
        for i in range(image.shape[0]):
            rows_transformed[i, :] = self.frft(image[i, :].astype(complex), alpha)
        
        # 再对列进行FrFT
        result = np.zeros_like(rows_transformed, dtype=complex)
        for j in range(image.shape[1]):
            result[:, j] = self.frft(rows_transformed[:, j], alpha)
        
        return result
    
    def ifrft2d(self, frft_image: np.ndarray, alpha: float) -> np.ndarray:
        """
        计算二维分数阶傅里叶逆变换
        
        Args:
            frft_image: FrFT域图像
            alpha: 旋转角度参数
            
        Returns:
            逆变换后的图像
        """
        return self.frft2d(frft_image, -alpha)
    
    def _general_frft(self, signal: np.ndarray, alpha: float) -> np.ndarray:
        """通用FrFT计算方法"""
        N = len(signal)
        
        # 构建FrFT核矩阵
        kernel = self._get_frft_kernel(N, alpha)
        
        # 执行变换
        return kernel @ signal
    
    def _get_frft_kernel(self, N: int, alpha: float) -> np.ndarray:
        """获取FrFT核矩阵（带缓存）"""
        cache_key = (N, alpha)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        
        # 计算核矩阵
        kernel = self._compute_frft_kernel(N, alpha)
        self._kernel_cache[cache_key] = kernel
        
        return kernel
    
    def _compute_frft_kernel(self, N: int, alpha: float) -> np.ndarray:
        """计算FrFT核矩阵"""
        # 使用Hermite函数基的FrFT实现
        phi = alpha * np.pi / 2
        
        if abs(np.sin(phi)) < 1e-10:
            # 当sin(phi)接近0时的特殊处理
            if abs(np.cos(phi) - 1) < 1e-10:
                return np.eye(N, dtype=complex)
            else:
                return np.flip(np.eye(N), axis=0).astype(complex)
        
        # 一般情况
        kernel = np.zeros((N, N), dtype=complex)
        
        # 简化的核矩阵计算
        for m in range(N):
            for n in range(N):
                # 使用近似的核函数
                arg = np.pi * (m - N//2) * (n - N//2) * np.sin(phi) / N
                kernel[m, n] = np.exp(1j * arg) / np.sqrt(N * abs(np.sin(phi)))
        
        return kernel
    
    def _chirp_z_transform(self, signal: np.ndarray) -> np.ndarray:
        """Chirp-Z变换（用于alpha=0.5的情况）"""
        N = len(signal)
        
        # 生成chirp信号
        n = np.arange(N)
        chirp = np.exp(-1j * np.pi * n**2 / N)
        
        # 卷积计算
        padded_signal = np.zeros(2*N, dtype=complex)
        padded_signal[:N] = signal * chirp
        
        # 使用FFT加速卷积
        kernel = np.exp(1j * np.pi * np.arange(2*N)**2 / N)
        result = np.fft.ifft(np.fft.fft(padded_signal) * np.fft.fft(kernel))
        
        return result[:N] * chirp


class FrFTWatermark:
    """基于FrFT的水印处理器"""
    
    def __init__(self, config: FrFTWatermarkConfig):
        self.config = config
        self.frft_processor = FrFTProcessor()
        
    def embed(self, image: np.ndarray, watermark_bits: np.ndarray) -> FrFTEmbedResult:
        """
        嵌入FrFT域水印
        
        Args:
            image: 原始图像
            watermark_bits: 水印比特序列
            
        Returns:
            FrFT水印嵌入结果
        """
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image.copy()
            
            # 转换为浮点数
            float_image = gray_image.astype(np.float32)
            
            # 分块处理
            watermarked, positions = self._embed_blocks(float_image, watermark_bits)
            
            # 转换回uint8
            watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
            
            # 如果原图是彩色的，需要重新组合
            if len(image.shape) == 3:
                result_image = image.copy()
                result_image[:, :, 0] = watermarked
            else:
                result_image = watermarked
            
            # 计算质量指标
            psnr = self._calculate_psnr(gray_image, watermarked)
            ssim = self._calculate_ssim(gray_image, watermarked)
            
            return FrFTEmbedResult(
                watermarked_image=result_image,
                psnr=psnr,
                ssim=ssim,
                embedding_positions=positions,
                alpha_parameter=self.config.alpha,
                metadata={
                    "method": "FrFT",
                    "alpha": self.config.alpha,
                    "block_size": self.config.block_size,
                    "watermark_length": len(watermark_bits),
                    "strength": self.config.strength
                }
            )
            
        except Exception as e:
            logger.error(f"FrFT watermark embedding failed: {e}")
            raise
    
    def extract(self, watermarked_image: np.ndarray, watermark_length: int, 
                positions: Optional[List[Tuple[int, int]]] = None) -> FrFTExtractResult:
        """
        提取FrFT域水印（支持盲检测）
        
        Args:
            watermarked_image: 含水印图像
            watermark_length: 水印长度
            positions: 嵌入位置（如果已知）
            
        Returns:
            FrFT水印提取结果
        """
        try:
            # 转换为灰度图像
            if len(watermarked_image.shape) == 3:
                gray_image = cv2.cvtColor(watermarked_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = watermarked_image.copy()
            
            # 转换为浮点数
            float_image = gray_image.astype(np.float32)
            
            if positions is not None:
                # 非盲检测
                extracted_bits, confidence_scores, detection_map = self._extract_known_positions(
                    float_image, watermark_length, positions
                )
                extraction_positions = positions
            else:
                # 盲检测
                extracted_bits, confidence_scores, detection_map, extraction_positions = self._blind_extract(
                    float_image, watermark_length
                )
            
            # 计算相关峰值
            correlation_peak = np.max(confidence_scores) if confidence_scores else 0.0
            
            return FrFTExtractResult(
                extracted_bits=extracted_bits,
                confidence_scores=confidence_scores,
                detection_map=detection_map,
                correlation_peak=correlation_peak,
                extraction_positions=extraction_positions,
                metadata={
                    "method": "FrFT",
                    "alpha": self.config.alpha,
                    "extracted_length": len(extracted_bits),
                    "detection_threshold": self.config.detection_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"FrFT watermark extraction failed: {e}")
            raise
    
    def _embed_blocks(self, image: np.ndarray, watermark_bits: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """分块嵌入水印"""
        h, w = image.shape
        block_size = self.config.block_size
        overlap = int(block_size * self.config.overlap_ratio)
        step = block_size - overlap
        
        watermarked = image.copy()
        positions = []
        bit_index = 0
        
        # 遍历所有块
        for i in range(0, h - block_size + 1, step):
            for j in range(0, w - block_size + 1, step):
                if bit_index >= len(watermark_bits):
                    break
                
                # 提取块
                block = image[i:i+block_size, j:j+block_size]
                
                # 在FrFT域嵌入水印
                watermarked_block = self._embed_single_block(block, watermark_bits[bit_index])
                
                # 更新图像（使用重叠区域的平均值）
                if overlap > 0:
                    # 处理重叠区域
                    watermarked[i:i+block_size, j:j+block_size] = (
                        watermarked[i:i+block_size, j:j+block_size] + watermarked_block
                    ) / 2
                else:
                    watermarked[i:i+block_size, j:j+block_size] = watermarked_block
                
                positions.append((i, j))
                bit_index += 1
                
                if bit_index >= len(watermark_bits):
                    break
        
        return watermarked, positions
    
    def _embed_single_block(self, block: np.ndarray, bit: int) -> np.ndarray:
        """在单个块中嵌入一个比特"""
        # FrFT变换
        frft_block = self.frft_processor.frft2d(block.astype(complex), self.config.alpha)
        
        # 选择嵌入位置（中频区域）
        h, w = frft_block.shape
        center_h, center_w = h // 2, w // 2
        
        # 定义嵌入区域
        embed_region = slice(center_h - 2, center_h + 3), slice(center_w - 2, center_w + 3)
        
        # 量化索引调制
        delta = self.config.strength * np.mean(np.abs(frft_block))
        
        if bit == 1:
            # 增强特定频率分量
            frft_block[embed_region] += delta * np.exp(1j * np.pi / 4)
        else:
            # 减弱特定频率分量
            frft_block[embed_region] -= delta * np.exp(1j * np.pi / 4)
        
        # 逆FrFT变换
        reconstructed = self.frft_processor.ifrft2d(frft_block, self.config.alpha)
        
        return np.real(reconstructed)
    
    def _extract_known_positions(self, image: np.ndarray, watermark_length: int, 
                                positions: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[float], np.ndarray]:
        """从已知位置提取水印"""
        block_size = self.config.block_size
        extracted_bits = []
        confidence_scores = []
        
        # 创建检测图
        detection_map = np.zeros(image.shape)
        
        for idx, (i, j) in enumerate(positions[:watermark_length]):
            # 提取块
            if i + block_size <= image.shape[0] and j + block_size <= image.shape[1]:
                block = image[i:i+block_size, j:j+block_size]
                
                # 提取比特和置信度
                bit, confidence = self._extract_single_block(block)
                
                extracted_bits.append(bit)
                confidence_scores.append(confidence)
                
                # 更新检测图
                detection_map[i:i+block_size, j:j+block_size] = confidence
        
        return np.array(extracted_bits), confidence_scores, detection_map
    
    def _blind_extract(self, image: np.ndarray, watermark_length: int) -> Tuple[np.ndarray, List[float], np.ndarray, List[Tuple[int, int]]]:
        """盲检测提取水印"""
        h, w = image.shape
        block_size = self.config.block_size
        overlap = int(block_size * self.config.overlap_ratio)
        step = block_size - overlap
        
        # 扫描所有可能的位置
        candidates = []
        
        for i in range(0, h - block_size + 1, step):
            for j in range(0, w - block_size + 1, step):
                block = image[i:i+block_size, j:j+block_size]
                bit, confidence = self._extract_single_block(block)
                
                if confidence > self.config.detection_threshold:
                    candidates.append({
                        'position': (i, j),
                        'bit': bit,
                        'confidence': confidence
                    })
        
        # 按置信度排序
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 选择最佳候选位置
        selected_candidates = candidates[:watermark_length]
        
        extracted_bits = [c['bit'] for c in selected_candidates]
        confidence_scores = [c['confidence'] for c in selected_candidates]
        positions = [c['position'] for c in selected_candidates]
        
        # 创建检测图
        detection_map = np.zeros(image.shape)
        for candidate in candidates:
            i, j = candidate['position']
            detection_map[i:i+block_size, j:j+block_size] = candidate['confidence']
        
        return np.array(extracted_bits), confidence_scores, detection_map, positions
    
    def _extract_single_block(self, block: np.ndarray) -> Tuple[int, float]:
        """从单个块中提取一个比特"""
        # FrFT变换
        frft_block = self.frft_processor.frft2d(block.astype(complex), self.config.alpha)
        
        # 选择检测区域
        h, w = frft_block.shape
        center_h, center_w = h // 2, w // 2
        detect_region = slice(center_h - 2, center_h + 3), slice(center_w - 2, center_w + 3)
        
        # 计算检测统计量
        region_coeffs = frft_block[detect_region]
        
        # 基于相位和幅度的检测
        phase_sum = np.sum(np.angle(region_coeffs))
        magnitude_mean = np.mean(np.abs(region_coeffs))
        
        # 判断比特值
        if phase_sum > 0:
            bit = 1
            confidence = min(1.0, abs(phase_sum) / (np.pi * region_coeffs.size))
        else:
            bit = 0
            confidence = min(1.0, abs(phase_sum) / (np.pi * region_coeffs.size))
        
        # 结合幅度信息调整置信度
        if magnitude_mean > 0:
            confidence *= min(1.0, magnitude_mean / np.mean(np.abs(frft_block)))
        
        return bit, confidence
    
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


class FrFTWatermarkProcessor:
    """FrFT水印处理器统一接口"""
    
    def __init__(self, config: FrFTWatermarkConfig):
        self.config = config
        self.processor = FrFTWatermark(config)
    
    def embed_watermark(self, image: np.ndarray, watermark_bits: np.ndarray) -> FrFTEmbedResult:
        """嵌入FrFT域水印"""
        return self.processor.embed(image, watermark_bits)
    
    def extract_watermark(self, watermarked_image: np.ndarray, watermark_length: int,
                         positions: Optional[List[Tuple[int, int]]] = None) -> FrFTExtractResult:
        """提取FrFT域水印"""
        return self.processor.extract(watermarked_image, watermark_length, positions)
    
    def blind_detect(self, image: np.ndarray, max_watermark_length: int = 100) -> FrFTExtractResult:
        """盲检测水印存在性"""
        return self.processor.extract(image, max_watermark_length, positions=None)
    
    def local_detection(self, image: np.ndarray, region: Tuple[int, int, int, int],
                       watermark_length: int) -> FrFTExtractResult:
        """局部区域水印检测"""
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        return self.processor.extract(roi, watermark_length, positions=None)


# 攻击测试工具
class FrFTAttackSimulator:
    """FrFT水印攻击模拟器"""
    
    @staticmethod
    def geometric_attack(image: np.ndarray, attack_type: str, **params) -> np.ndarray:
        """几何攻击模拟"""
        if attack_type == "rotation":
            angle = params.get("angle", 5)
            center = (image.shape[1]//2, image.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, image.shape[::-1])
        
        elif attack_type == "scaling":
            scale = params.get("scale", 0.8)
            h, w = image.shape
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (new_w, new_h))
            return cv2.resize(scaled, (w, h))
        
        elif attack_type == "cropping":
            crop_ratio = params.get("crop_ratio", 0.1)
            h, w = image.shape
            crop_size = int(min(h, w) * crop_ratio)
            cropped = image[crop_size:h-crop_size, crop_size:w-crop_size]
            return cv2.resize(cropped, (w, h))
        
        return image
    
    @staticmethod
    def signal_processing_attack(image: np.ndarray, attack_type: str, **params) -> np.ndarray:
        """信号处理攻击模拟"""
        if attack_type == "gaussian_noise":
            noise_std = params.get("noise_std", 5)
            noise = np.random.normal(0, noise_std, image.shape)
            return np.clip(image + noise, 0, 255).astype(np.uint8)
        
        elif attack_type == "gaussian_blur":
            kernel_size = params.get("kernel_size", 3)
            sigma = params.get("sigma", 1.0)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        elif attack_type == "jpeg_compression":
            quality = params.get("quality", 80)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded_img = cv2.imencode('.jpg', image, encode_param)
            return cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
        
        return image