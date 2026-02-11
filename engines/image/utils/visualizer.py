"""
可视化工具
提供FFT频谱、宏块网格、置信度热力图等可视化功能
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


def visualize_fft_spectrum(image: np.ndarray, output_path: Optional[str] = None,
                          peaks: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """
    可视化FFT频谱
    
    Args:
        image: 输入图像
        output_path: 输出路径，None表示不保存
        peaks: 峰值位置列表 [(x, y), ...]
        
    Returns:
        频谱可视化图像
        
    需求: 10.6 - 支持可视化输出（FFT频谱图）
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 进行FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 归一化到[0, 255]
    magnitude_spectrum = ((magnitude_spectrum - magnitude_spectrum.min()) / 
                         (magnitude_spectrum.max() - magnitude_spectrum.min()) * 255)
    magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
    
    # 创建彩色图像用于标记峰值
    spectrum_color = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2BGR)
    
    # 标记峰值
    if peaks is not None:
        for x, y in peaks:
            cv2.circle(spectrum_color, (x, y), 5, (0, 0, 255), 2)
            cv2.circle(spectrum_color, (x, y), 2, (0, 255, 0), -1)
    
    # 保存图像
    if output_path is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(spectrum_color)
        plt.title('FFT Magnitude Spectrum')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return spectrum_color


def visualize_block_grid(image: np.ndarray, block_size: int = 32,
                        output_path: Optional[str] = None,
                        valid_blocks: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """
    可视化宏块网格
    
    Args:
        image: 输入图像
        block_size: 宏块大小
        output_path: 输出路径
        valid_blocks: 有效宏块位置列表 [(row, col), ...]
        
    Returns:
        带网格的图像
        
    需求: 10.6 - 支持可视化输出（宏块网格图）
    """
    result = image.copy()
    height, width = image.shape[:2]
    
    # 绘制网格线
    for i in range(0, height, block_size):
        cv2.line(result, (0, i), (width, i), (0, 255, 0), 1)
    for j in range(0, width, block_size):
        cv2.line(result, (j, 0), (j, height), (0, 255, 0), 1)
    
    # 标记有效宏块
    if valid_blocks is not None:
        for row, col in valid_blocks:
            y = row * block_size
            x = col * block_size
            # 绘制半透明矩形
            overlay = result.copy()
            cv2.rectangle(overlay, (x, y), (x + block_size, y + block_size),
                         (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
            # 绘制边框
            cv2.rectangle(result, (x, y), (x + block_size, y + block_size),
                         (0, 255, 0), 2)
    
    # 保存图像
    if output_path is not None:
        cv2.imwrite(output_path, result)
    
    return result


def visualize_confidence_heatmap(confidence_map: np.ndarray, 
                                 output_path: Optional[str] = None,
                                 block_size: int = 32) -> np.ndarray:
    """
    可视化置信度热力图
    
    Args:
        confidence_map: 置信度图 (rows, cols) 或 (rows, cols, bits)
        output_path: 输出路径
        block_size: 宏块大小（用于放大显示）
        
    Returns:
        热力图图像
        
    需求: 10.6 - 支持可视化输出（置信度热力图）
    """
    # 如果是3D数组，取平均值
    if len(confidence_map.shape) == 3:
        confidence_2d = np.mean(confidence_map, axis=2)
    else:
        confidence_2d = confidence_map
    
    # 归一化到[0, 1]
    if confidence_2d.max() > 0:
        confidence_normalized = confidence_2d / confidence_2d.max()
    else:
        confidence_normalized = confidence_2d
    
    # 使用matplotlib创建热力图
    plt.figure(figsize=(12, 10))
    plt.imshow(confidence_normalized, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Confidence')
    plt.title('Watermark Confidence Heatmap')
    plt.xlabel('Block Column')
    plt.ylabel('Block Row')
    
    # 添加数值标注
    if confidence_2d.shape[0] <= 20 and confidence_2d.shape[1] <= 20:
        for i in range(confidence_2d.shape[0]):
            for j in range(confidence_2d.shape[1]):
                text = plt.text(j, i, f'{confidence_2d[i, j]:.2f}',
                              ha="center", va="center", color="w", fontsize=8)
    
    plt.tight_layout()
    
    # 保存或转换为图像
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # 转换为numpy数组
    plt.gcf().canvas.draw()
    buf = plt.gcf().canvas.buffer_rgba()
    heatmap_img = np.asarray(buf)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGBA2BGR)
    
    plt.close()
    
    return heatmap_img


def visualize_comparison(original: np.ndarray, watermarked: np.ndarray,
                        output_path: Optional[str] = None,
                        titles: Optional[Tuple[str, str]] = None,
                        metrics: Optional[dict] = None) -> np.ndarray:
    """
    可视化对比图
    
    Args:
        original: 原始图像
        watermarked: 带水印图像
        output_path: 输出路径
        titles: 图像标题 (title1, title2)
        metrics: 质量指标字典 {'PSNR': value, 'SSIM': value}
        
    Returns:
        对比图像
        
    需求: 10.6 - 支持可视化输出（对比图生成）
    """
    if titles is None:
        titles = ('Original', 'Watermarked')
    
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 显示原始图像
    if len(original.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title(titles[0], fontsize=14)
    axes[0].axis('off')
    
    # 显示带水印图像
    if len(watermarked.shape) == 3:
        axes[1].imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
    else:
        axes[1].imshow(watermarked, cmap='gray')
    axes[1].set_title(titles[1], fontsize=14)
    axes[1].axis('off')
    
    # 添加质量指标
    if metrics is not None:
        metrics_text = '\n'.join([f'{k}: {v:.2f}' for k, v in metrics.items()])
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存或转换为图像
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # 转换为numpy数组
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    comparison_img = np.asarray(buf)
    comparison_img = cv2.cvtColor(comparison_img, cv2.COLOR_RGBA2BGR)
    
    plt.close()
    
    return comparison_img


def visualize_bit_distribution(bits: np.ndarray, output_path: Optional[str] = None) -> np.ndarray:
    """
    可视化bit分布
    
    Args:
        bits: bit数组
        output_path: 输出路径
        
    Returns:
        分布图像
    """
    plt.figure(figsize=(10, 6))
    
    # 统计0和1的数量
    unique, counts = np.unique(bits, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    plt.bar(['0', '1'], [count_dict.get(0, 0), count_dict.get(1, 0)])
    plt.title('Bit Distribution')
    plt.xlabel('Bit Value')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # 转换为numpy数组
    plt.gcf().canvas.draw()
    buf = plt.gcf().canvas.buffer_rgba()
    dist_img = np.asarray(buf)
    dist_img = cv2.cvtColor(dist_img, cv2.COLOR_RGBA2BGR)
    
    plt.close()
    
    return dist_img


def create_visualization_report(images: dict, output_path: str):
    """
    创建综合可视化报告
    
    Args:
        images: 图像字典 {'name': image, ...}
        output_path: 输出路径
    """
    n_images = len(images)
    cols = 2
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, img) in enumerate(images.items()):
        row = idx // cols
        col = idx % cols
        
        if len(img.shape) == 3:
            axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(name, fontsize=12)
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
