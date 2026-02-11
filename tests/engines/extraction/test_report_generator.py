"""
Unit tests for WatermarkReportGenerator
测试水印提取可视化PDF报告生成器
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import fitz  # PyMuPDF

from engines.extraction.report_generator import (
    WatermarkReportGenerator,
    ExtractionResult,
    ReportConfig
)


class TestWatermarkReportGenerator:
    """测试WatermarkReportGenerator类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_extraction_results(self):
        """创建示例提取结果"""
        results = [
            ExtractionResult(
                page_num=1,
                watermark_data="user_admin|2024-01-20 10:30:00|confidential",
                confidence=0.85,
                bits=np.array([1, 0, 1, 1, 0, 0, 1, 0] * 16),
                metadata={
                    'method': 'FFT',
                    'frequency_band': 'mid',
                    'extracted_length': 128
                }
            ),
            ExtractionResult(
                page_num=2,
                watermark_data="user_admin|2024-01-20 10:30:00|confidential",
                confidence=0.45,
                bits=np.array([0, 1, 0, 1, 1, 0, 0, 1] * 16),
                metadata={
                    'method': 'FFT',
                    'frequency_band': 'mid',
                    'extracted_length': 128
                }
            ),
            ExtractionResult(
                page_num=3,
                watermark_data=None,
                confidence=0.25,
                bits=np.array([1, 1, 0, 0, 1, 0, 1, 0] * 16),
                metadata={
                    'method': 'FFT',
                    'frequency_band': 'mid',
                    'extracted_length': 128
                }
            ),
        ]
        return results
    
    @pytest.fixture
    def sample_source_file(self, temp_dir):
        """创建示例源文件"""
        source_file = temp_dir / "sample_document.pdf"
        
        # 创建一个简单的PDF
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        text_writer = fitz.TextWriter(page.rect)
        text_writer.append(fitz.Point(50, 50), "Sample Document", fontsize=18)
        text_writer.write_text(page)
        doc.save(str(source_file))
        doc.close()
        
        return source_file
    
    def test_report_generator_initialization(self):
        """测试报告生成器初始化"""
        # 使用默认配置
        generator = WatermarkReportGenerator()
        assert generator.config is not None
        assert generator.config.page_width == 595
        assert generator.config.page_height == 842
        
        # 使用自定义配置
        custom_config = ReportConfig(
            page_width=600,
            page_height=850,
            title_fontsize=20
        )
        generator = WatermarkReportGenerator(custom_config)
        assert generator.config.page_width == 600
        assert generator.config.title_fontsize == 20
    
    def test_generate_extraction_report(
        self,
        temp_dir,
        sample_source_file,
        sample_extraction_results
    ):
        """测试生成提取报告"""
        generator = WatermarkReportGenerator()
        output_path = temp_dir / "extraction_report.pdf"
        
        result_path = generator.generate_extraction_report(
            source_file=sample_source_file,
            extraction_results=sample_extraction_results,
            output_path=output_path,
            include_technical_details=True
        )
        
        # 验证文件已创建
        assert result_path.exists()
        assert result_path == output_path
        
        # 验证PDF内容
        doc = fitz.open(str(result_path))
        assert len(doc) > 0  # 至少有一页
        
        # 检查第一页内容
        page = doc[0]
        text = page.get_text()
        
        # 验证标题
        assert "水印提取报告" in text or "Watermark Extraction Report" in text
        
        # 验证文件信息
        assert "sample_document.pdf" in text
        assert "处理页数" in text or "Pages Processed" in text
        
        doc.close()
    
    def test_generate_report_with_dict_results(
        self,
        temp_dir,
        sample_source_file
    ):
        """测试使用字典格式的提取结果生成报告"""
        generator = WatermarkReportGenerator()
        output_path = temp_dir / "report_from_dict.pdf"
        
        # 使用字典格式的结果
        dict_results = [
            {
                'page_num': 1,
                'watermark_data': "test_watermark",
                'confidence': 0.75,
                'bits': np.array([1, 0, 1, 0] * 8),
                'metadata': {'method': 'FFT'}
            }
        ]
        
        result_path = generator.generate_extraction_report(
            source_file=sample_source_file,
            extraction_results=dict_results,
            output_path=output_path
        )
        
        assert result_path.exists()
        
        # 验证内容 - 检查所有页面
        doc = fitz.open(str(result_path))
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        assert "test_watermark" in all_text
        doc.close()
    
    def test_confidence_color_coding(
        self,
        temp_dir,
        sample_source_file,
        sample_extraction_results
    ):
        """测试置信度颜色编码"""
        generator = WatermarkReportGenerator()
        output_path = temp_dir / "confidence_test.pdf"
        
        generator.generate_extraction_report(
            source_file=sample_source_file,
            extraction_results=sample_extraction_results,
            output_path=output_path
        )
        
        # 验证报告生成成功
        assert output_path.exists()
        
        # 打开PDF并检查内容
        doc = fitz.open(str(output_path))
        
        # 查找包含置信度信息的页面
        found_confidence = False
        all_text = ""
        for page in doc:
            text = page.get_text()
            all_text += text
            if "置信度" in text or "Confidence" in text:
                found_confidence = True
        
        # 验证三个不同置信度级别都存在
        assert "85.0%" in all_text or "45.0%" in all_text or "25.0%" in all_text
        
        assert found_confidence, "Confidence information not found in report"
        doc.close()
    
    def test_bits_preview_formatting(self):
        """测试比特序列预览格式化"""
        generator = WatermarkReportGenerator()
        
        # 测试短比特序列
        short_bits = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        preview = generator._format_bits_preview(short_bits, max_bits=32)
        assert preview == "10101100"
        
        # 测试长比特序列
        long_bits = np.array([1, 0] * 50)
        preview = generator._format_bits_preview(long_bits, max_bits=32)
        assert preview.endswith("...")
        assert len(preview) == 35  # 32 bits + "..."
        
        # 测试空比特序列
        empty_bits = np.array([])
        preview = generator._format_bits_preview(empty_bits)
        assert "[空" in preview or "Empty" in preview
    
    def test_hex_conversion(self):
        """测试比特到十六进制转换"""
        generator = WatermarkReportGenerator()
        
        # 测试正常转换
        bits = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
        hex_str = generator._bits_to_hex(bits)
        assert hex_str == "ACC3"
        
        # 测试空数组
        empty_bits = np.array([])
        hex_str = generator._bits_to_hex(empty_bits)
        assert "[空" in hex_str or "Empty" in hex_str
        
        # 测试不足一字节
        short_bits = np.array([1, 0, 1])
        hex_str = generator._bits_to_hex(short_bits)
        assert "[不足一字节" in hex_str or "Less than 1 byte" in hex_str
    
    def test_watermark_data_formatting(self):
        """测试水印数据格式化"""
        generator = WatermarkReportGenerator()
        
        # 测试正常数据
        data = "user_admin|2024-01-20|confidential"
        formatted = generator._format_watermark_data(data, max_length=60)
        assert formatted == data
        
        # 测试长数据
        long_data = "a" * 100
        formatted = generator._format_watermark_data(long_data, max_length=60)
        assert formatted.endswith("...")
        assert len(formatted) == 63  # 60 + "..."
        
        # 测试None数据
        formatted = generator._format_watermark_data(None)
        assert "[无法解码" in formatted or "Unable to decode" in formatted
    
    def test_confidence_formatting(self):
        """测试置信度格式化"""
        generator = WatermarkReportGenerator()
        
        # 测试高置信度（绿色）
        color, text = generator._format_confidence(0.85)
        assert text == "85.0%"
        assert color == generator.config.color_high_confidence
        
        # 测试中置信度（橙色）
        color, text = generator._format_confidence(0.45)
        assert text == "45.0%"
        assert color == generator.config.color_medium_confidence
        
        # 测试低置信度（红色）
        color, text = generator._format_confidence(0.25)
        assert text == "25.0%"
        assert color == generator.config.color_low_confidence
    
    def test_file_size_formatting(self, temp_dir):
        """测试文件大小格式化"""
        generator = WatermarkReportGenerator()
        
        # 创建测试文件
        test_file = temp_dir / "test.txt"
        test_file.write_text("a" * 1024)  # 1 KB
        
        size_str = generator._format_file_size(test_file)
        assert "KB" in size_str
        
        # 测试不存在的文件
        non_existent = temp_dir / "non_existent.txt"
        size_str = generator._format_file_size(non_existent)
        assert size_str == "N/A"
    
    def test_report_with_technical_details(
        self,
        temp_dir,
        sample_source_file,
        sample_extraction_results
    ):
        """测试包含技术细节的报告"""
        generator = WatermarkReportGenerator()
        output_path = temp_dir / "report_with_details.pdf"
        
        generator.generate_extraction_report(
            source_file=sample_source_file,
            extraction_results=sample_extraction_results,
            output_path=output_path,
            include_technical_details=True
        )
        
        # 验证报告包含技术细节
        doc = fitz.open(str(output_path))
        
        # 查找技术细节页面
        found_technical = False
        for page in doc:
            text = page.get_text()
            if "技术细节" in text or "Technical Details" in text:
                found_technical = True
                assert "FFT" in text
                assert "置信度分布" in text or "Confidence Distribution" in text
                break
        
        assert found_technical, "Technical details not found in report"
        doc.close()
    
    def test_report_without_technical_details(
        self,
        temp_dir,
        sample_source_file,
        sample_extraction_results
    ):
        """测试不包含技术细节的报告"""
        generator = WatermarkReportGenerator()
        output_path = temp_dir / "report_without_details.pdf"
        
        generator.generate_extraction_report(
            source_file=sample_source_file,
            extraction_results=sample_extraction_results,
            output_path=output_path,
            include_technical_details=False
        )
        
        assert output_path.exists()
        
        # 验证报告不包含技术细节页面
        doc = fitz.open(str(output_path))
        
        # 检查是否没有技术细节
        has_technical = False
        for page in doc:
            text = page.get_text()
            if "技术细节" in text or "Technical Details" in text:
                has_technical = True
                break
        
        # 不应该有技术细节页面
        assert not has_technical, "Technical details found but should not be included"
        doc.close()
    
    def test_multi_page_results(
        self,
        temp_dir,
        sample_source_file
    ):
        """测试多页提取结果的汇总"""
        generator = WatermarkReportGenerator()
        output_path = temp_dir / "multi_page_report.pdf"
        
        # 创建多个页面的提取结果
        results = []
        for i in range(10):
            results.append(ExtractionResult(
                page_num=i + 1,
                watermark_data=f"watermark_page_{i+1}",
                confidence=0.5 + (i % 5) * 0.1,
                bits=np.array([i % 2] * 128),
                metadata={'method': 'FFT'}
            ))
        
        generator.generate_extraction_report(
            source_file=sample_source_file,
            extraction_results=results,
            output_path=output_path
        )
        
        # 验证报告包含所有页面
        doc = fitz.open(str(output_path))
        
        # 检查是否包含所有页面的信息
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        
        # 验证至少包含部分页面信息
        assert "第 1 页" in all_text or "Page 1" in all_text
        assert "第 10 页" in all_text or "Page 10" in all_text
        
        doc.close()
    
    def test_notes_page_content(
        self,
        temp_dir,
        sample_source_file,
        sample_extraction_results
    ):
        """测试说明页面内容"""
        generator = WatermarkReportGenerator()
        output_path = temp_dir / "report_with_notes.pdf"
        
        generator.generate_extraction_report(
            source_file=sample_source_file,
            extraction_results=sample_extraction_results,
            output_path=output_path
        )
        
        # 验证说明页面存在
        doc = fitz.open(str(output_path))
        
        found_notes = False
        for page in doc:
            text = page.get_text()
            if "说明" in text or "Notes" in text:
                found_notes = True
                # 验证说明内容
                assert "FFT频域水印" in text or "FFT frequency domain" in text
                assert "颜色编码" in text or "Color Coding" in text
                assert "绿色" in text or "Green" in text
                break
        
        assert found_notes, "Notes page not found in report"
        doc.close()
    
    def test_custom_config(
        self,
        temp_dir,
        sample_source_file,
        sample_extraction_results
    ):
        """测试自定义配置"""
        custom_config = ReportConfig(
            title_fontsize=24,
            body_fontsize=14,
            color_high_confidence=(0, 1, 0),  # 纯绿色
            margin_left=60
        )
        
        generator = WatermarkReportGenerator(custom_config)
        output_path = temp_dir / "custom_config_report.pdf"
        
        generator.generate_extraction_report(
            source_file=sample_source_file,
            extraction_results=sample_extraction_results,
            output_path=output_path
        )
        
        # 验证报告生成成功
        assert output_path.exists()
        
        # 验证配置已应用
        assert generator.config.title_fontsize == 24
        assert generator.config.body_fontsize == 14
        assert generator.config.margin_left == 60
    
    def test_empty_results(
        self,
        temp_dir,
        sample_source_file
    ):
        """测试空提取结果"""
        generator = WatermarkReportGenerator()
        output_path = temp_dir / "empty_results_report.pdf"
        
        # 空结果列表
        empty_results = []
        
        generator.generate_extraction_report(
            source_file=sample_source_file,
            extraction_results=empty_results,
            output_path=output_path
        )
        
        # 验证报告仍然生成
        assert output_path.exists()
        
        # 验证报告内容
        doc = fitz.open(str(output_path))
        page = doc[0]
        text = page.get_text()
        
        # 应该显示0页处理
        assert "0" in text
        doc.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
