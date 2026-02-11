"""
水印提取可视化PDF报告生成器
Watermark Extraction Visualization PDF Report Generator

实现水印提取结果的可视化展示，包括：
- 置信度颜色编码
- 比特序列预览
- 多页面提取结果汇总
- 技术细节和说明信息
"""

import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """单页水印提取结果"""
    page_num: int
    watermark_data: Optional[str]
    confidence: float
    bits: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class ReportConfig:
    """报告配置"""
    page_width: float = 595  # A4宽度（点）
    page_height: float = 842  # A4高度（点）
    margin_left: float = 50
    margin_right: float = 50
    margin_top: float = 50
    margin_bottom: float = 50
    title_fontsize: int = 18
    heading_fontsize: int = 14
    body_fontsize: int = 12
    small_fontsize: int = 9
    line_spacing: float = 20
    section_spacing: float = 30
    
    # 颜色配置
    color_high_confidence: tuple = (0, 0.5, 0)  # 绿色 >50%
    color_medium_confidence: tuple = (0.8, 0.4, 0)  # 橙色 30-50%
    color_low_confidence: tuple = (0.8, 0, 0)  # 红色 <30%
    color_title: tuple = (0, 0, 0)  # 黑色
    color_heading: tuple = (0, 0, 0.5)  # 深蓝色
    color_body: tuple = (0, 0, 0)  # 黑色
    color_note: tuple = (0.4, 0.4, 0.4)  # 灰色
    color_separator: tuple = (0, 0, 0)  # 黑色


class WatermarkReportGenerator:
    """水印提取可视化PDF报告生成器"""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        初始化报告生成器
        
        Args:
            config: 报告配置，如果为None则使用默认配置
        """
        self.config = config or ReportConfig()
        
    def generate_extraction_report(
        self,
        source_file: Union[str, Path],
        extraction_results: List[Union[ExtractionResult, Dict]],
        output_path: Union[str, Path],
        include_technical_details: bool = True
    ) -> Path:
        """
        生成水印提取的可视化PDF报告
        
        Args:
            source_file: 源文件路径
            extraction_results: 提取结果列表
            output_path: 输出报告路径
            include_technical_details: 是否包含技术细节
            
        Returns:
            生成的报告文件路径
            
        报告内容包括：
        1. 文档基本信息（文件名、提取时间、处理页数）
        2. 每页的提取结果：
           - 水印内容（文本形式）
           - 提取置信度（百分比和颜色编码）
           - 比特序列（前32位预览）
           - 十六进制表示
        3. 可视化元素：
           - 置信度颜色编码（绿色>50%，橙色30-50%，红色<30%）
           - 分隔线和格式化布局
        4. 说明信息和技术细节
        """
        try:
            source_file = Path(source_file)
            output_path = Path(output_path)
            
            # 转换字典为ExtractionResult对象
            results = self._normalize_results(extraction_results)
            
            # 创建PDF文档
            doc = fitz.open()
            
            # 添加报告页面
            self._add_title_page(doc, source_file, results)
            self._add_results_pages(doc, results)
            
            if include_technical_details:
                self._add_technical_details_page(doc, results)
            
            self._add_notes_page(doc)
            
            # 保存报告
            doc.save(str(output_path))
            doc.close()
            
            logger.info(f"Generated watermark extraction report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate extraction report: {e}")
            raise
    
    def _normalize_results(
        self,
        extraction_results: List[Union[ExtractionResult, Dict]]
    ) -> List[ExtractionResult]:
        """将字典格式的结果转换为ExtractionResult对象"""
        normalized = []
        
        for result in extraction_results:
            if isinstance(result, ExtractionResult):
                normalized.append(result)
            elif isinstance(result, dict):
                normalized.append(ExtractionResult(
                    page_num=result.get('page_num', 0),
                    watermark_data=result.get('watermark_data'),
                    confidence=result.get('confidence', 0.0),
                    bits=result.get('bits', np.array([])),
                    metadata=result.get('metadata', {})
                ))
            else:
                raise ValueError(f"Invalid result type: {type(result)}")
        
        return normalized
    
    def _add_title_page(
        self,
        doc: fitz.Document,
        source_file: Path,
        results: List[ExtractionResult]
    ):
        """添加标题页"""
        page = doc.new_page(
            width=self.config.page_width,
            height=self.config.page_height
        )
        page_rect = page.rect
        
        y_pos = self.config.margin_top
        
        # 标题
        self._add_text(
            page, page_rect,
            self.config.margin_left, y_pos,
            "水印提取报告",
            fontsize=self.config.title_fontsize,
            color=self.config.color_title
        )
        y_pos += self.config.title_fontsize + 5
        
        self._add_text(
            page, page_rect,
            self.config.margin_left, y_pos,
            "Watermark Extraction Report",
            fontsize=self.config.title_fontsize,
            color=self.config.color_title
        )
        y_pos += self.config.section_spacing
        
        # 分隔线
        self._add_separator(page, y_pos)
        y_pos += self.config.section_spacing
        
        # 文件信息
        info_items = [
            ("源文件 / Source File", source_file.name),
            ("提取时间 / Extraction Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            ("处理页数 / Pages Processed", str(len(results))),
            ("文件大小 / File Size", self._format_file_size(source_file)),
        ]
        
        for label, value in info_items:
            self._add_text(
                page, page_rect,
                self.config.margin_left, y_pos,
                f"{label}: {value}",
                fontsize=self.config.body_fontsize,
                color=self.config.color_body
            )
            y_pos += self.config.line_spacing
        
        y_pos += self.config.section_spacing
        
        # 汇总统计
        self._add_text(
            page, page_rect,
            self.config.margin_left, y_pos,
            "提取结果汇总 / Extraction Summary:",
            fontsize=self.config.heading_fontsize,
            color=self.config.color_heading
        )
        y_pos += self.config.section_spacing
        
        # 计算统计信息
        avg_confidence = np.mean([r.confidence for r in results]) if results else 0.0
        successful_extractions = sum(1 for r in results if r.watermark_data)
        
        summary_items = [
            ("平均置信度 / Average Confidence", f"{avg_confidence * 100:.1f}%"),
            ("成功提取 / Successful Extractions", f"{successful_extractions}/{len(results)}"),
            ("提取方法 / Extraction Method", results[0].metadata.get('method', 'FFT') if results else 'N/A'),
        ]
        
        for label, value in summary_items:
            self._add_text(
                page, page_rect,
                self.config.margin_left + 10, y_pos,
                f"• {label}: {value}",
                fontsize=self.config.body_fontsize,
                color=self.config.color_body
            )
            y_pos += self.config.line_spacing
    
    def _add_results_pages(
        self,
        doc: fitz.Document,
        results: List[ExtractionResult]
    ):
        """添加提取结果页面"""
        # 创建新页面
        page = doc.new_page(
            width=self.config.page_width,
            height=self.config.page_height
        )
        page_rect = page.rect
        
        y_pos = self.config.margin_top
        
        # 标题
        self._add_text(
            page, page_rect,
            self.config.margin_left, y_pos,
            "详细提取结果 / Detailed Extraction Results",
            fontsize=self.config.heading_fontsize,
            color=self.config.color_heading
        )
        y_pos += self.config.section_spacing
        
        # 遍历每页的提取结果
        for result in results:
            # 检查是否需要新页面
            if y_pos > self.config.page_height - self.config.margin_bottom - 150:
                page = doc.new_page(
                    width=self.config.page_width,
                    height=self.config.page_height
                )
                page_rect = page.rect
                y_pos = self.config.margin_top
            
            # 页码标题
            self._add_text(
                page, page_rect,
                self.config.margin_left + 10, y_pos,
                f"第 {result.page_num} 页 / Page {result.page_num}:",
                fontsize=self.config.body_fontsize,
                color=self.config.color_heading
            )
            y_pos += self.config.line_spacing + 5
            
            # 水印数据
            watermark_text = self._format_watermark_data(result.watermark_data)
            self._add_text(
                page, page_rect,
                self.config.margin_left + 20, y_pos,
                f"水印内容 / Watermark: {watermark_text}",
                fontsize=self.config.body_fontsize - 1,
                color=self.config.color_body
            )
            y_pos += self.config.line_spacing
            
            # 置信度（带颜色编码）
            confidence_color, confidence_text = self._format_confidence(result.confidence)
            self._add_text(
                page, page_rect,
                self.config.margin_left + 20, y_pos,
                f"置信度 / Confidence: {confidence_text}",
                fontsize=self.config.body_fontsize - 1,
                color=confidence_color
            )
            y_pos += self.config.line_spacing
            
            # 比特序列（前32位）
            bits_display = self._format_bits_preview(result.bits, max_bits=32)
            self._add_text(
                page, page_rect,
                self.config.margin_left + 20, y_pos,
                f"比特序列 / Bit Sequence: {bits_display}",
                fontsize=self.config.small_fontsize,
                color=self.config.color_note
            )
            y_pos += self.config.line_spacing
            
            # 十六进制表示
            hex_data = self._bits_to_hex(result.bits[:64])
            self._add_text(
                page, page_rect,
                self.config.margin_left + 20, y_pos,
                f"十六进制 / Hex: {hex_data}",
                fontsize=self.config.small_fontsize,
                color=self.config.color_note
            )
            y_pos += self.config.line_spacing + 10
            
            # 添加轻微分隔线
            self._add_light_separator(page, y_pos)
            y_pos += 15
    
    def _add_technical_details_page(
        self,
        doc: fitz.Document,
        results: List[ExtractionResult]
    ):
        """添加技术细节页面"""
        page = doc.new_page(
            width=self.config.page_width,
            height=self.config.page_height
        )
        page_rect = page.rect
        
        y_pos = self.config.margin_top
        
        # 标题
        self._add_text(
            page, page_rect,
            self.config.margin_left, y_pos,
            "技术细节 / Technical Details",
            fontsize=self.config.heading_fontsize,
            color=self.config.color_heading
        )
        y_pos += self.config.section_spacing
        
        # 提取方法信息
        if results:
            metadata = results[0].metadata
            
            details = [
                ("提取方法 / Method", metadata.get('method', 'FFT')),
                ("频域范围 / Frequency Band", metadata.get('frequency_band', 'mid')),
                ("水印长度 / Watermark Length", str(metadata.get('extracted_length', 'N/A'))),
            ]
            
            for label, value in details:
                self._add_text(
                    page, page_rect,
                    self.config.margin_left + 10, y_pos,
                    f"• {label}: {value}",
                    fontsize=self.config.body_fontsize,
                    color=self.config.color_body
                )
                y_pos += self.config.line_spacing
        
        y_pos += self.config.section_spacing
        
        # 置信度分布
        self._add_text(
            page, page_rect,
            self.config.margin_left, y_pos,
            "置信度分布 / Confidence Distribution:",
            fontsize=self.config.heading_fontsize,
            color=self.config.color_heading
        )
        y_pos += self.config.section_spacing
        
        # 统计置信度分布
        high_conf = sum(1 for r in results if r.confidence > 0.5)
        medium_conf = sum(1 for r in results if 0.3 <= r.confidence <= 0.5)
        low_conf = sum(1 for r in results if r.confidence < 0.3)
        
        distribution = [
            ("高置信度 / High (>50%)", high_conf, self.config.color_high_confidence),
            ("中置信度 / Medium (30-50%)", medium_conf, self.config.color_medium_confidence),
            ("低置信度 / Low (<30%)", low_conf, self.config.color_low_confidence),
        ]
        
        for label, count, color in distribution:
            self._add_text(
                page, page_rect,
                self.config.margin_left + 10, y_pos,
                f"• {label}: {count} 页",
                fontsize=self.config.body_fontsize,
                color=color
            )
            y_pos += self.config.line_spacing
    
    def _add_notes_page(self, doc: fitz.Document):
        """添加说明页面"""
        page = doc.new_page(
            width=self.config.page_width,
            height=self.config.page_height
        )
        page_rect = page.rect
        
        y_pos = self.config.margin_top
        
        # 分隔线
        self._add_separator(page, y_pos)
        y_pos += self.config.section_spacing
        
        # 标题
        self._add_text(
            page, page_rect,
            self.config.margin_left, y_pos,
            "说明 / Notes:",
            fontsize=self.config.heading_fontsize,
            color=self.config.color_heading
        )
        y_pos += self.config.section_spacing
        
        # 说明内容
        notes = [
            "• 本报告展示了从PDF文档中提取的FFT频域水印信息",
            "  This report shows FFT frequency domain watermark extracted from PDF",
            "",
            "• 置信度表示提取结果的可靠程度，受多种因素影响",
            "  Confidence indicates reliability, affected by various factors",
            "",
            "• 如果无法解码，将显示十六进制表示",
            "  Hex representation shown if decoding fails",
            "",
            "• 颜色编码说明 / Color Coding:",
            "  - 绿色 / Green: 高置信度 (>50%)",
            "  - 橙色 / Orange: 中置信度 (30-50%)",
            "  - 红色 / Red: 低置信度 (<30%)",
            "",
            "• 水印提取技术基于FFT频域变换",
            "  Watermark extraction based on FFT frequency domain transformation",
            "",
            "• 本报告可作为审计和证据保存使用",
            "  This report can be used for audit and evidence preservation",
        ]
        
        for note in notes:
            if note == "":
                y_pos += 5
                continue
                
            self._add_text(
                page, page_rect,
                self.config.margin_left + 10, y_pos,
                note,
                fontsize=self.config.small_fontsize,
                color=self.config.color_note
            )
            y_pos += self.config.line_spacing - 5
    
    # ========== 辅助方法 ==========
    
    def _add_text(
        self,
        page: fitz.Page,
        page_rect: fitz.Rect,
        x: float,
        y: float,
        text: str,
        fontsize: int = 12,
        color: tuple = (0, 0, 0)
    ):
        """添加文本到页面"""
        text_writer = fitz.TextWriter(page_rect)
        text_writer.append(fitz.Point(x, y), text, fontsize=fontsize)
        text_writer.write_text(page, color=color)
    
    def _add_separator(self, page: fitz.Page, y_pos: float, width: float = 1.0):
        """添加分隔线"""
        page.draw_line(
            fitz.Point(self.config.margin_left, y_pos),
            fitz.Point(self.config.page_width - self.config.margin_right, y_pos),
            color=self.config.color_separator,
            width=width
        )
    
    def _add_light_separator(self, page: fitz.Page, y_pos: float):
        """添加轻微分隔线"""
        page.draw_line(
            fitz.Point(self.config.margin_left + 20, y_pos),
            fitz.Point(self.config.page_width - self.config.margin_right - 20, y_pos),
            color=(0.7, 0.7, 0.7),
            width=0.5
        )
    
    def _format_confidence(self, confidence: float) -> tuple:
        """格式化置信度显示，返回(颜色, 文本)"""
        percent = confidence * 100
        
        if percent > 50:
            color = self.config.color_high_confidence
        elif percent > 30:
            color = self.config.color_medium_confidence
        else:
            color = self.config.color_low_confidence
        
        return color, f"{percent:.1f}%"
    
    def _format_watermark_data(self, watermark_data: Optional[str], max_length: int = 60) -> str:
        """格式化水印数据显示"""
        if not watermark_data:
            return "[无法解码 / Unable to decode]"
        
        if len(watermark_data) > max_length:
            return watermark_data[:max_length] + "..."
        
        return watermark_data
    
    def _format_bits_preview(self, bits: np.ndarray, max_bits: int = 32) -> str:
        """格式化比特序列预览"""
        if len(bits) == 0:
            return "[空 / Empty]"
        
        bits_str = ''.join(str(int(b)) for b in bits[:max_bits])
        
        if len(bits) > max_bits:
            bits_str += "..."
        
        return bits_str
    
    def _bits_to_hex(self, bits: np.ndarray) -> str:
        """将比特序列转换为十六进制字符串"""
        try:
            if len(bits) == 0:
                return "[空 / Empty]"
            
            num_bytes = len(bits) // 8
            if num_bytes == 0:
                return "[不足一字节 / Less than 1 byte]"
            
            byte_data = bytearray()
            for i in range(num_bytes):
                byte_val = 0
                for j in range(8):
                    bit_idx = i * 8 + j
                    if bit_idx < len(bits):
                        byte_val = (byte_val << 1) | int(bits[bit_idx])
                byte_data.append(byte_val)
            
            hex_str = byte_data.hex().upper()
            
            # 限制显示长度
            if len(hex_str) > 32:
                hex_str = hex_str[:32] + "..."
            
            return hex_str
            
        except Exception as e:
            logger.error(f"Failed to convert bits to hex: {e}")
            return "[转换失败 / Conversion failed]"
    
    def _format_file_size(self, file_path: Path) -> str:
        """格式化文件大小"""
        try:
            if not file_path.exists():
                return "N/A"
            
            size_bytes = file_path.stat().st_size
            
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
                
        except Exception:
            return "N/A"
