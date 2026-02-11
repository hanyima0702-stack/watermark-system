"""
性能指标计算模块
提供处理时间统计、错误率计算和置信度统计功能
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerformanceTimer:
    """
    性能计时器
    用于统计各个阶段的处理时间
    
    需求: 9.7, 9.8 - 性能测试
    """
    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed_time: float = 0.0
    sub_timers: Dict[str, 'PerformanceTimer'] = field(default_factory=dict)
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        return self
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        return False
    
    def add_sub_timer(self, name: str) -> 'PerformanceTimer':
        """
        添加子计时器
        
        Args:
            name: 子计时器名称
            
        Returns:
            子计时器对象
        """
        timer = PerformanceTimer(name)
        self.sub_timers[name] = timer
        return timer
    
    def get_report(self, indent: int = 0) -> str:
        """
        生成计时报告
        
        Args:
            indent: 缩进级别
            
        Returns:
            格式化的报告字符串
        """
        prefix = "  " * indent
        report = f"{prefix}{self.name}: {self.elapsed_time:.4f}s\n"
        
        for sub_timer in self.sub_timers.values():
            report += sub_timer.get_report(indent + 1)
        
        return report
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            包含计时信息的字典
        """
        result = {
            'name': self.name,
            'elapsed_time': self.elapsed_time,
            'start_time': self.start_time,
            'end_time': self.end_time
        }
        
        if self.sub_timers:
            result['sub_timers'] = {
                name: timer.to_dict() 
                for name, timer in self.sub_timers.items()
            }
        
        return result


def calculate_error_rate(original_bits: np.ndarray, 
                        extracted_bits: np.ndarray) -> Dict[str, float]:
    """
    计算错误率
    
    Args:
        original_bits: 原始bit数组
        extracted_bits: 提取的bit数组
        
    Returns:
        包含错误率统计的字典
        
    需求: 9.7, 9.8 - 错误率计算
    """
    if len(original_bits) != len(extracted_bits):
        raise ValueError("Bit arrays must have the same length")
    
    # 计算错误位数
    errors = np.sum(original_bits != extracted_bits)
    total = len(original_bits)
    
    # 计算错误率
    error_rate = errors / total if total > 0 else 0.0
    accuracy = 1.0 - error_rate
    
    # 计算位错误分布
    error_positions = np.where(original_bits != extracted_bits)[0]
    
    return {
        'total_bits': int(total),
        'error_bits': int(errors),
        'correct_bits': int(total - errors),
        'error_rate': float(error_rate),
        'accuracy': float(accuracy),
        'error_positions': error_positions.tolist()
    }


def calculate_confidence_stats(confidences: np.ndarray) -> Dict[str, float]:
    """
    计算置信度统计
    
    Args:
        confidences: 置信度数组
        
    Returns:
        包含置信度统计的字典
        
    需求: 9.7, 9.8 - 置信度统计
    """
    if confidences.size == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'q25': 0.0,
            'q75': 0.0
        }
    
    return {
        'mean': float(np.mean(confidences)),
        'std': float(np.std(confidences)),
        'min': float(np.min(confidences)),
        'max': float(np.max(confidences)),
        'median': float(np.median(confidences)),
        'q25': float(np.percentile(confidences, 25)),
        'q75': float(np.percentile(confidences, 75))
    }


def calculate_bit_error_rate(original: str, extracted: str) -> float:
    """
    计算比特错误率(BER)
    
    Args:
        original: 原始二进制字符串
        extracted: 提取的二进制字符串
        
    Returns:
        比特错误率
    """
    if len(original) != len(extracted):
        raise ValueError("Strings must have the same length")
    
    errors = sum(o != e for o, e in zip(original, extracted))
    return errors / len(original) if len(original) > 0 else 0.0


def calculate_throughput(data_size: int, elapsed_time: float) -> Dict[str, float]:
    """
    计算吞吐量
    
    Args:
        data_size: 数据大小(字节)
        elapsed_time: 处理时间(秒)
        
    Returns:
        吞吐量统计
    """
    if elapsed_time <= 0:
        return {
            'bytes_per_second': 0.0,
            'kb_per_second': 0.0,
            'mb_per_second': 0.0
        }
    
    bytes_per_second = data_size / elapsed_time
    
    return {
        'bytes_per_second': bytes_per_second,
        'kb_per_second': bytes_per_second / 1024,
        'mb_per_second': bytes_per_second / (1024 * 1024)
    }


def generate_performance_report(timer: PerformanceTimer,
                               error_stats: Optional[Dict] = None,
                               confidence_stats: Optional[Dict] = None,
                               additional_metrics: Optional[Dict] = None) -> Dict[str, Any]:
    """
    生成综合性能报告
    
    Args:
        timer: 性能计时器
        error_stats: 错误率统计
        confidence_stats: 置信度统计
        additional_metrics: 额外的指标
        
    Returns:
        综合性能报告
        
    需求: 9.7, 9.8 - 生成性能报告
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'timing': timer.to_dict(),
        'summary': {
            'total_time': timer.elapsed_time
        }
    }
    
    if error_stats is not None:
        report['error_stats'] = error_stats
        report['summary']['accuracy'] = error_stats.get('accuracy', 0.0)
    
    if confidence_stats is not None:
        report['confidence_stats'] = confidence_stats
        report['summary']['mean_confidence'] = confidence_stats.get('mean', 0.0)
    
    if additional_metrics is not None:
        report['additional_metrics'] = additional_metrics
    
    return report


class MetricsCollector:
    """
    指标收集器
    用于收集和聚合多次运行的指标
    """
    
    def __init__(self):
        """初始化指标收集器"""
        self.metrics: List[Dict[str, Any]] = []
    
    def add_metric(self, metric: Dict[str, Any]):
        """
        添加一个指标记录
        
        Args:
            metric: 指标字典
        """
        self.metrics.append(metric)
    
    def get_aggregated_stats(self) -> Dict[str, Any]:
        """
        获取聚合统计
        
        Returns:
            聚合后的统计信息
        """
        if not self.metrics:
            return {}
        
        # 提取所有数值型指标
        numeric_metrics = {}
        for metric in self.metrics:
            for key, value in metric.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # 计算统计量
        stats = {}
        for key, values in numeric_metrics.items():
            values_array = np.array(values)
            stats[key] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array))
            }
        
        stats['total_runs'] = len(self.metrics)
        
        return stats
    
    def clear(self):
        """清空收集的指标"""
        self.metrics.clear()


def calculate_snr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """
    计算信噪比(SNR)
    
    Args:
        original: 原始图像
        watermarked: 带水印图像
        
    Returns:
        SNR值(dB)
    """
    signal_power = np.mean(original.astype(float) ** 2)
    noise = watermarked.astype(float) - original.astype(float)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


def calculate_capacity(image_shape: tuple, block_size: int, 
                      bits_per_block: int) -> Dict[str, int]:
    """
    计算水印容量
    
    Args:
        image_shape: 图像尺寸 (height, width)
        block_size: 宏块大小
        bits_per_block: 每个宏块的bit数
        
    Returns:
        容量统计
    """
    height, width = image_shape[:2]
    
    blocks_h = height // block_size
    blocks_w = width // block_size
    total_blocks = blocks_h * blocks_w
    total_bits = total_blocks * bits_per_block
    
    return {
        'image_size': (height, width),
        'block_size': block_size,
        'blocks_horizontal': blocks_w,
        'blocks_vertical': blocks_h,
        'total_blocks': total_blocks,
        'bits_per_block': bits_per_block,
        'total_capacity_bits': total_bits,
        'total_capacity_bytes': total_bits // 8
    }
