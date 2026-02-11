"""
图像处理工具函数
提供色彩空间转换、图像质量评估和预处理功能
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def bgr_to_yuv(image: np.ndarray) -> np.ndarray:
    """
    将BGR图像转换为YUV色彩空间
    
    Args:
        image: BGR格式图像 (H, W, 3)
        
    Returns:
        YUV格式图像 (H, W, 3)
        
    需求: 3.5 - 在YUV色彩空间的Y通道进行操作
    """
    if image is None or image.size == 0:
        raise ValueError("输入图像为空")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是3通道BGR图像")
    
    # 使用OpenCV进行色彩空间转换
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    return yuv_image


def yuv_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    将YUV图像转换为BGR色彩空间
    
    Args:
        image: YUV格式图像 (H, W, 3)
        
    Returns:
        BGR格式图像 (H, W, 3)
        
    需求: 3.5 - 在YUV色彩空间的Y通道进行操作
    """
    if image is None or image.size == 0:
        raise ValueError("输入图像为空")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是3通道YUV图像")
    
    # 使用OpenCV进行色彩空间转换
    bgr_image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    
    return bgr_image


def calculate_psnr(original: np.ndarray, modified: np.ndarray, 
                   max_value: float = 255.0) -> float:
    """
    计算两幅图像之间的峰值信噪比(PSNR)
    
    Args:
        original: 原始图像
        modified: 修改后的图像
        max_value: 像素最大值，默认255
        
    Returns:
        PSNR值(dB)，值越高表示质量越好
        
    需求: 6.6 - 图像质量评估
    """
    if original.shape != modified.shape:
        raise ValueError("两幅图像尺寸必须相同")
    
    # 计算均方误差(MSE)
    mse = np.mean((original.astype(float) - modified.astype(float)) ** 2)
    
    # 如果MSE为0，说明图像完全相同
    if mse == 0:
        return float('inf')
    
    # 计算PSNR
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    
    return psnr


def calculate_ssim(original: np.ndarray, modified: np.ndarray,
                   window_size: int = 11, k1: float = 0.01, 
                   k2: float = 0.03) -> float:
    """
    计算两幅图像之间的结构相似性指数(SSIM)
    
    Args:
        original: 原始图像
        modified: 修改后的图像
        window_size: 滑动窗口大小
        k1, k2: SSIM算法常数
        
    Returns:
        SSIM值，范围[0, 1]，值越高表示相似度越高
        
    需求: 6.6 - 图像质量评估
    """
    if original.shape != modified.shape:
        raise ValueError("两幅图像尺寸必须相同")
    
    # 转换为灰度图
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        modified_gray = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        modified_gray = modified
    
    # 转换为float
    img1 = original_gray.astype(float)
    img2 = modified_gray.astype(float)
    
    # 常数
    c1 = (k1 * 255) ** 2
    c2 = (k2 * 255) ** 2
    
    # 创建高斯窗口
    kernel = cv2.getGaussianKernel(window_size, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    # 计算均值
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(np.mean(ssim_map))


def preprocess_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None,
                    normalize: bool = False) -> np.ndarray:
    """
    图像预处理
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)，None表示不调整
        normalize: 是否归一化到[0, 1]
        
    Returns:
        预处理后的图像
        
    需求: 6.6 - 图像预处理
    """
    processed = image.copy()
    
    # 调整尺寸
    if target_size is not None:
        processed = cv2.resize(processed, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 归一化
    if normalize:
        processed = processed.astype(float) / 255.0
    
    return processed


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    归一化图像到[0, 255]范围
    
    Args:
        image: 输入图像
        
    Returns:
        归一化后的图像
    """
    if image.size == 0:
        return image
    
    # 获取最小值和最大值
    min_val = np.min(image)
    max_val = np.max(image)
    
    # 避免除零
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)
    
    # 归一化到[0, 255]
    normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return normalized


def denoise_image(image: np.ndarray, method: str = 'gaussian', 
                 strength: int = 5) -> np.ndarray:
    """
    图像去噪
    
    Args:
        image: 输入图像
        method: 去噪方法 ('gaussian', 'median', 'bilateral')
        strength: 去噪强度
        
    Returns:
        去噪后的图像
    """
    if method == 'gaussian':
        # 高斯模糊
        ksize = strength * 2 + 1
        denoised = cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == 'median':
        # 中值滤波
        ksize = strength * 2 + 1
        denoised = cv2.medianBlur(image, ksize)
    elif method == 'bilateral':
        # 双边滤波
        denoised = cv2.bilateralFilter(image, strength * 2 + 1, 
                                       strength * 10, strength * 10)
    else:
        raise ValueError(f"不支持的去噪方法: {method}")
    
    return denoised


def clip_image(image: np.ndarray, min_val: float = 0, 
               max_val: float = 255) -> np.ndarray:
    """
    裁剪图像像素值到指定范围
    
    Args:
        image: 输入图像
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        裁剪后的图像
    """
    return np.clip(image, min_val, max_val)


def get_y_channel(image: np.ndarray) -> np.ndarray:
    """
    获取图像的Y通道(亮度通道)
    
    Args:
        image: BGR格式图像
        
    Returns:
        Y通道图像
        
    需求: 3.5 - 在YUV色彩空间的Y通道进行操作
    """
    yuv = bgr_to_yuv(image)
    return yuv[:, :, 0]


def set_y_channel(image: np.ndarray, y_channel: np.ndarray) -> np.ndarray:
    """
    设置图像的Y通道
    
    Args:
        image: BGR格式图像
        y_channel: 新的Y通道
        
    Returns:
        更新后的BGR图像
        
    需求: 3.5 - 在YUV色彩空间的Y通道进行操作
    """
    yuv = bgr_to_yuv(image)
    yuv[:, :, 0] = y_channel
    return yuv_to_bgr(yuv)
