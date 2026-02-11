"""
FFT分析器模块

该模块实现基于FFT的几何变换检测，用于识别图像的旋转和缩放。
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy import signal


class FFTAnalyzer:
    """
    FFT分析器
    
    负责检测图像的几何变换参数（旋转角度和缩放比例）。
    通过分析FFT频谱中的同步信号峰值位置来计算变换参数。
    """
    
    def __init__(self, block_size: int = 32, 
                 peak_threshold: float = 0.3):
        """
        初始化FFT分析器
        
        Args:
            block_size: 宏块大小（像素），默认32
            peak_threshold: 峰值检测阈值（相对于最大值的比例）
        """
        self.block_size = block_size
        self.peak_threshold = peak_threshold
        # 预期的频域峰值距离（基于宏块重复模式）
        # 对于32×32的宏块，频域中会在 freq = 1/32 处出现峰值
        self.expected_freq = 1.0 / block_size
    
    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """
        分析图像的几何变换
        
        对图像进行2D FFT变换，检测同步信号峰值，计算旋转角度和缩放比例。
        
        Args:
            image: 待分析图像（BGR或灰度）
        
        Returns:
            包含以下键的字典:
            - rotation: 旋转角度（度，逆时针为正）
            - scale: 缩放比例
            - confidence: 检测置信度（0-1）
            - peaks_found: 检测到的峰值数量
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 执行2D FFT
        fft_result = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft_result)
        
        # 计算幅度谱（对数变换以增强可视化）
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log1p(magnitude)
        
        # 检测同步信号峰值
        peaks = self.detect_sync_peaks(magnitude_log)
        
        if len(peaks) < 2:
            # 峰值不足，无法计算几何变换
            return {
                'rotation': 0.0,
                'scale': 1.0,
                'confidence': 0.0,
                'peaks_found': len(peaks)
            }
        
        # 计算旋转角度和缩放比例
        rotation, scale, confidence = self._calculate_transform_params(
            peaks, gray.shape
        )
        
        return {
            'rotation': rotation,
            'scale': scale,
            'confidence': confidence,
            'peaks_found': len(peaks)
        }
    
    def detect_sync_peaks(self, fft_magnitude: np.ndarray) -> List[Tuple[int, int]]:
        """
        检测FFT频谱中的同步信号峰值
        
        检测由宏块重复模式产生的频域峰值。
        
        Args:
            fft_magnitude: FFT幅度谱（对数变换后）
        
        Returns:
            峰值坐标列表 [(x, y), ...]，按强度降序排列
        """
        # 归一化幅度谱
        magnitude_norm = fft_magnitude / (np.max(fft_magnitude) + 1e-10)
        
        # 使用更小的阈值来检测峰值
        threshold = self.peak_threshold * np.max(magnitude_norm)
        
        # 使用形态学操作找到局部最大值
        # 创建一个结构元素用于局部最大值检测
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size))
        dilated = cv2.dilate(magnitude_norm.astype(np.float32), kernel)
        local_max = (magnitude_norm == dilated)
        
        # 应用阈值过滤
        peaks_mask = local_max & (magnitude_norm > threshold)
        
        # 获取峰值坐标
        peak_coords = np.argwhere(peaks_mask)
        
        if len(peak_coords) == 0:
            return []
        
        # 获取峰值强度
        peak_values = magnitude_norm[peak_coords[:, 0], peak_coords[:, 1]]
        
        # 按强度降序排序
        sorted_indices = np.argsort(peak_values)[::-1]
        sorted_peaks = peak_coords[sorted_indices]
        
        # 移除中心DC分量（通常是最强的峰值）
        center = np.array(fft_magnitude.shape) // 2
        filtered_peaks = []
        
        for peak in sorted_peaks:
            # 跳过距离中心太近的峰值（DC分量及其附近）
            distance_to_center = np.linalg.norm(peak - center)
            if distance_to_center > 3:  # 至少距离中心3个像素
                filtered_peaks.append(tuple(peak))
        
        # 返回前12个最强的峰值
        return filtered_peaks[:12]
    
    def _calculate_transform_params(self, peaks: List[Tuple[int, int]], 
                                    image_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        根据峰值位置计算旋转角度和缩放比例
        
        Args:
            peaks: 峰值坐标列表
            image_shape: 图像形状 (height, width)
        
        Returns:
            (旋转角度, 缩放比例, 置信度)
        """
        if len(peaks) < 2:
            return 0.0, 1.0, 0.0
        
        center = np.array(image_shape) / 2.0
        
        # 计算所有峰值相对于中心的向量
        vectors = []
        distances = []
        angles = []
        
        for peak in peaks:
            p = np.array(peak)
            vec = p - center
            dist = np.linalg.norm(vec)
            
            # 只考虑距离在合理范围内的峰值
            # 基于宏块大小，预期频域峰值距离中心约为 image_size / block_size
            expected_dist = min(image_shape) / (2.0 * self.block_size)
            
            if dist > 3:  # 排除太靠近中心的点
                vectors.append(vec)
                distances.append(dist)
                # 计算角度
                angle = np.arctan2(vec[0], vec[1])
                angles.append(angle)
        
        if len(vectors) == 0:
            return 0.0, 1.0, 0.0
        
        # 寻找对称的峰值对来确定旋转角度
        # 在频域中，旋转会导致所有峰值一起旋转
        rotation_deg = 0.0
        if len(angles) >= 2:
            # 使用最强的两个峰值的角度差来估计旋转
            # 如果图像旋转了θ度，频域也会旋转θ度
            angle_diffs = []
            for i in range(len(angles)):
                for j in range(i + 1, len(angles)):
                    # 检查是否是对称峰值（相差约180度）
                    angle_diff = abs(angles[i] - angles[j])
                    if abs(angle_diff - np.pi) < 0.5:  # 接近180度
                        # 这是一对对称峰值，使用它们的平均角度
                        avg_angle = (angles[i] + angles[j]) / 2.0
                        rotation_deg = np.degrees(avg_angle)
                        break
            
            # 如果没有找到对称峰值对，使用主要峰值的角度
            if rotation_deg == 0.0 and len(angles) > 0:
                # 使用距离最远的峰值（通常是最显著的）
                max_dist_idx = np.argmax(distances)
                rotation_deg = np.degrees(angles[max_dist_idx])
        
        # 归一化到[-45, 45]度范围
        while rotation_deg > 45:
            rotation_deg -= 90
        while rotation_deg < -45:
            rotation_deg += 90
        
        # 计算缩放比例
        # 基于峰值距离与预期距离的比值
        expected_dist = min(image_shape) / (2.0 * self.block_size)
        if len(distances) > 0:
            avg_distance = np.mean(distances)
            scale = avg_distance / expected_dist if expected_dist > 0 else 1.0
        else:
            scale = 1.0
        
        # 计算置信度
        # 基于峰值数量和距离的一致性
        confidence = min(1.0, len(peaks) / 4.0)  # 理想情况下有多个峰值
        
        # 如果距离变化太大，降低置信度
        if len(distances) > 1:
            dist_std = np.std(distances)
            dist_mean = np.mean(distances)
            if dist_mean > 0:
                cv = dist_std / dist_mean  # 变异系数
                if cv > 0.5:  # 如果变异系数大于0.5，降低置信度
                    confidence *= 0.5
        
        return rotation_deg, scale, confidence
    
    def get_fft_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        获取图像的FFT频谱（用于可视化）
        
        Args:
            image: 输入图像
        
        Returns:
            FFT幅度谱（对数变换后，归一化到0-255）
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 执行2D FFT
        fft_result = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft_result)
        
        # 计算幅度谱
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log1p(magnitude)
        
        # 归一化到0-255
        magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, 
                                      cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return magnitude_norm
