"""
性能指标测试
"""

import pytest
import numpy as np
import time
from engines.image.utils.metrics import (
    PerformanceTimer,
    calculate_error_rate,
    calculate_confidence_stats,
    calculate_bit_error_rate,
    calculate_throughput,
    generate_performance_report,
    MetricsCollector,
    calculate_snr,
    calculate_capacity
)


class TestPerformanceTimer:
    """测试性能计时器"""
    
    def test_basic_timing(self):
        """测试基本计时"""
        timer = PerformanceTimer("test")
        timer.start()
        time.sleep(0.1)
        timer.stop()
        
        assert timer.elapsed_time >= 0.1
        assert timer.elapsed_time < 0.2
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with PerformanceTimer("test") as timer:
            time.sleep(0.1)
        
        assert timer.elapsed_time >= 0.1
    
    def test_sub_timers(self):
        """测试子计时器"""
        main_timer = PerformanceTimer("main")
        main_timer.start()
        
        sub1 = main_timer.add_sub_timer("sub1")
        sub1.start()
        time.sleep(0.05)
        sub1.stop()
        
        sub2 = main_timer.add_sub_timer("sub2")
        sub2.start()
        time.sleep(0.05)
        sub2.stop()
        
        main_timer.stop()
        
        assert len(main_timer.sub_timers) == 2
        assert main_timer.elapsed_time >= 0.1
    
    def test_get_report(self):
        """测试生成报告"""
        timer = PerformanceTimer("test")
        timer.start()
        time.sleep(0.1)
        timer.stop()
        
        report = timer.get_report()
        assert "test" in report
        assert "0." in report
    
    def test_to_dict(self):
        """测试转换为字典"""
        timer = PerformanceTimer("test")
        timer.start()
        time.sleep(0.1)
        timer.stop()
        
        data = timer.to_dict()
        assert data['name'] == 'test'
        assert 'elapsed_time' in data
        assert data['elapsed_time'] >= 0.1
    
    def test_timer_not_started(self):
        """测试未启动的计时器"""
        timer = PerformanceTimer("test")
        
        with pytest.raises(ValueError):
            timer.stop()


class TestErrorRate:
    """测试错误率计算"""
    
    def test_no_errors(self):
        """测试无错误"""
        original = np.array([1, 0, 1, 0, 1, 0])
        extracted = np.array([1, 0, 1, 0, 1, 0])
        
        stats = calculate_error_rate(original, extracted)
        
        assert stats['error_rate'] == 0.0
        assert stats['accuracy'] == 1.0
        assert stats['error_bits'] == 0
    
    def test_some_errors(self):
        """测试部分错误"""
        original = np.array([1, 0, 1, 0, 1, 0])
        extracted = np.array([1, 1, 1, 0, 0, 0])
        
        stats = calculate_error_rate(original, extracted)
        
        assert stats['error_bits'] == 2
        assert stats['error_rate'] == pytest.approx(2/6)
        assert stats['accuracy'] == pytest.approx(4/6)
    
    def test_all_errors(self):
        """测试全部错误"""
        original = np.array([1, 1, 1, 1])
        extracted = np.array([0, 0, 0, 0])
        
        stats = calculate_error_rate(original, extracted)
        
        assert stats['error_rate'] == 1.0
        assert stats['accuracy'] == 0.0
    
    def test_error_positions(self):
        """测试错误位置"""
        original = np.array([1, 0, 1, 0, 1, 0])
        extracted = np.array([1, 1, 1, 0, 0, 0])
        
        stats = calculate_error_rate(original, extracted)
        
        assert stats['error_positions'] == [1, 4]
    
    def test_length_mismatch(self):
        """测试长度不匹配"""
        original = np.array([1, 0, 1])
        extracted = np.array([1, 0])
        
        with pytest.raises(ValueError):
            calculate_error_rate(original, extracted)


class TestConfidenceStats:
    """测试置信度统计"""
    
    def test_basic_stats(self):
        """测试基本统计"""
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        
        stats = calculate_confidence_stats(confidences)
        
        assert stats['mean'] == pytest.approx(0.7)
        assert stats['min'] == 0.5
        assert stats['max'] == 0.9
        assert stats['median'] == 0.7
    
    def test_empty_array(self):
        """测试空数组"""
        confidences = np.array([])
        
        stats = calculate_confidence_stats(confidences)
        
        assert stats['mean'] == 0.0
        assert stats['min'] == 0.0
        assert stats['max'] == 0.0
    
    def test_single_value(self):
        """测试单个值"""
        confidences = np.array([0.8])
        
        stats = calculate_confidence_stats(confidences)
        
        assert stats['mean'] == 0.8
        assert stats['min'] == 0.8
        assert stats['max'] == 0.8
        assert stats['std'] == 0.0
    
    def test_quartiles(self):
        """测试四分位数"""
        confidences = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        stats = calculate_confidence_stats(confidences)
        
        assert 'q25' in stats
        assert 'q75' in stats
        assert stats['q25'] < stats['median'] < stats['q75']


class TestBitErrorRate:
    """测试比特错误率"""
    
    def test_identical_strings(self):
        """测试相同字符串"""
        ber = calculate_bit_error_rate("10101010", "10101010")
        assert ber == 0.0
    
    def test_different_strings(self):
        """测试不同字符串"""
        ber = calculate_bit_error_rate("10101010", "01010101")
        assert ber == 1.0
    
    def test_partial_errors(self):
        """测试部分错误"""
        ber = calculate_bit_error_rate("1111", "1101")
        assert ber == 0.25
    
    def test_length_mismatch(self):
        """测试长度不匹配"""
        with pytest.raises(ValueError):
            calculate_bit_error_rate("111", "11")


class TestThroughput:
    """测试吞吐量计算"""
    
    def test_basic_throughput(self):
        """测试基本吞吐量"""
        stats = calculate_throughput(1024 * 1024, 1.0)  # 1MB in 1 second
        
        assert stats['bytes_per_second'] == 1024 * 1024
        assert stats['kb_per_second'] == 1024
        assert stats['mb_per_second'] == 1.0
    
    def test_zero_time(self):
        """测试零时间"""
        stats = calculate_throughput(1024, 0.0)
        
        assert stats['bytes_per_second'] == 0.0
    
    def test_negative_time(self):
        """测试负时间"""
        stats = calculate_throughput(1024, -1.0)
        
        assert stats['bytes_per_second'] == 0.0


class TestPerformanceReport:
    """测试性能报告生成"""
    
    def test_basic_report(self):
        """测试基本报告"""
        timer = PerformanceTimer("test")
        timer.start()
        time.sleep(0.1)
        timer.stop()
        
        report = generate_performance_report(timer)
        
        assert 'timestamp' in report
        assert 'timing' in report
        assert 'summary' in report
        assert report['summary']['total_time'] >= 0.1
    
    def test_report_with_error_stats(self):
        """测试带错误统计的报告"""
        timer = PerformanceTimer("test")
        timer.start()
        timer.stop()
        
        error_stats = {'accuracy': 0.95, 'error_rate': 0.05}
        report = generate_performance_report(timer, error_stats=error_stats)
        
        assert 'error_stats' in report
        assert report['summary']['accuracy'] == 0.95
    
    def test_report_with_confidence_stats(self):
        """测试带置信度统计的报告"""
        timer = PerformanceTimer("test")
        timer.start()
        timer.stop()
        
        confidence_stats = {'mean': 0.85, 'std': 0.1}
        report = generate_performance_report(timer, confidence_stats=confidence_stats)
        
        assert 'confidence_stats' in report
        assert report['summary']['mean_confidence'] == 0.85
    
    def test_report_with_additional_metrics(self):
        """测试带额外指标的报告"""
        timer = PerformanceTimer("test")
        timer.start()
        timer.stop()
        
        additional = {'custom_metric': 42}
        report = generate_performance_report(timer, additional_metrics=additional)
        
        assert 'additional_metrics' in report
        assert report['additional_metrics']['custom_metric'] == 42


class TestMetricsCollector:
    """测试指标收集器"""
    
    def test_add_and_aggregate(self):
        """测试添加和聚合"""
        collector = MetricsCollector()
        
        collector.add_metric({'time': 1.0, 'accuracy': 0.9})
        collector.add_metric({'time': 2.0, 'accuracy': 0.95})
        collector.add_metric({'time': 1.5, 'accuracy': 0.92})
        
        stats = collector.get_aggregated_stats()
        
        assert stats['total_runs'] == 3
        assert 'time' in stats
        assert stats['time']['mean'] == pytest.approx(1.5)
        assert 'accuracy' in stats
        assert stats['accuracy']['mean'] == pytest.approx(0.923, rel=0.01)
    
    def test_empty_collector(self):
        """测试空收集器"""
        collector = MetricsCollector()
        stats = collector.get_aggregated_stats()
        
        assert stats == {}
    
    def test_clear(self):
        """测试清空"""
        collector = MetricsCollector()
        collector.add_metric({'time': 1.0})
        collector.clear()
        
        stats = collector.get_aggregated_stats()
        assert stats == {}


class TestSNR:
    """测试信噪比计算"""
    
    def test_identical_images(self):
        """测试相同图像"""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        snr = calculate_snr(image, image)
        
        assert snr == float('inf')
    
    def test_different_images(self):
        """测试不同图像"""
        original = np.ones((100, 100), dtype=np.uint8) * 128
        watermarked = original.copy()
        watermarked += 10
        
        snr = calculate_snr(original, watermarked)
        
        assert snr > 0
        assert snr < 100


class TestCapacity:
    """测试容量计算"""
    
    def test_basic_capacity(self):
        """测试基本容量"""
        stats = calculate_capacity((512, 512), 32, 128)
        
        assert stats['blocks_horizontal'] == 16
        assert stats['blocks_vertical'] == 16
        assert stats['total_blocks'] == 256
        assert stats['total_capacity_bits'] == 256 * 128
        assert stats['total_capacity_bytes'] == 256 * 128 // 8
    
    def test_different_sizes(self):
        """测试不同尺寸"""
        stats = calculate_capacity((1024, 768), 32, 64)
        
        assert stats['blocks_horizontal'] == 24
        assert stats['blocks_vertical'] == 32
        assert stats['total_blocks'] == 768


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
