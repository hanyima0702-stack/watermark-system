"""
PDF document processor for watermark embedding and digital signatures.
Implements visible watermarks, invisible watermarks, and PKCS#7 digital signatures.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
import logging
import base64
import hashlib
from PIL import Image
# PDF processing libraries
import fitz  # PyMuPDF
import cv2

import io

from markdown_it.rules_inline import image

from .base_processor import DocumentProcessor, WatermarkConfig, DocumentProcessingError
from .base_processor import WatermarkEmbeddingError, DigitalSignatureError
from ..extraction.report_generator import WatermarkReportGenerator, ExtractionResult
import numpy as np
from ..image.image_processor import ImageProcessor


logger = logging.getLogger(__name__)


class PDFProcessor(DocumentProcessor):
    """Processor for PDF documents with watermark and signature capabilities."""
    
    def __init__(self,):
        """
        初始化PDF处理器

        """
        super().__init__()
        self.supported_formats = ['.pdf']
        
        # 初始化水印处理器
        self.imageProcessor = ImageProcessor()




    def add_visible_watermark(self, file_path: Path, watermark_config: WatermarkConfig,
                            user_id: str, timestamp: datetime) -> Path:
        """Add visible watermark to PDF document using overlay technique."""
        if not self.is_supported_format(file_path):
            raise DocumentProcessingError(f"Unsupported format: {file_path.suffix}")
            
        output_path = self.generate_output_path(file_path)
        
        try:
            visible_config = watermark_config.visible_watermark
            if not visible_config.get('enabled', False):
                # Just copy the file if watermark is disabled
                import shutil
                shutil.copy2(file_path, output_path)
                return output_path
                
            # Open PDF document
            pdf_doc = fitz.open(str(file_path))
            
            # Process each page
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                
                # Add watermark layers
                for layer in visible_config.get('layers', []):
                    if layer.get('type') == 'text':
                        self._add_pdf_text_watermark(page, layer, user_id, timestamp)
                    elif layer.get('type') == 'qrcode':
                        self._add_pdf_qrcode_watermark(page, layer, user_id, timestamp)
                        
            # Save watermarked PDF
            pdf_doc.save(str(output_path))
            pdf_doc.close()
            
            logger.info(f"Added visible watermark to PDF: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to add visible watermark to PDF: {e}")
            raise WatermarkEmbeddingError(f"Failed to add visible watermark: {e}")

    def add_invisible_watermark(self, file_path: Path, watermark_data: str , mode = "img") -> Path:
        """Add invisible watermark to PDF using selected method."""
        assert mode in ('img', 'str', 'bit'), "mode in ('img','str','bit')"

        if not self.is_supported_format(file_path):
            raise DocumentProcessingError(f"Unsupported format: {file_path.suffix}")

        output_path = self.generate_output_path(file_path, "_invisible")

        try:
            # 打开PDF文档
            pdf_doc = fitz.open(str(file_path))

            # 创建新的PDF文档
            output_doc = fitz.open()

            # 处理每一页
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]

                # 将PDF页面渲染为高分辨率图像（300 DPI）
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # 转换为numpy数组
                img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                img_array = img_data.reshape(pix.height, pix.width, pix.n)

                # RGB -> BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                #读取原图片
                self.imageProcessor.bwm_core.read_img_arr(img=img_array)

                #读取水印

                self.imageProcessor.read_wm(watermark_data,mode=mode)

                #嵌入水印

                embed_img =self.imageProcessor.embed()


                # 将含水印的图像转换回PDF页面
                new_page = output_doc.new_page(
                    width=page.rect.width,
                    height=page.rect.height
                )

                # 将图像插入新页面
                img_buffer = io.BytesIO()
                from PIL import Image

                # 四舍五入
                rounded_img = np.round(embed_img)

                #限制范围
                clipped_img = np.clip(rounded_img, 0, 255)

                #类型转换
                uint8_img = clipped_img.astype(np.uint8)

                #颜色空间转换 BGR -> RGB
                # if uint8_img.shape[2] == 3:  # 确保是彩色图片
                uint8_img = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2RGB)

                pil_img = Image.fromarray(uint8_img)
                pil_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                # 插入图像到页面
                img_rect = new_page.rect
                new_page.insert_image(img_rect, stream=img_buffer.getvalue())

                logger.info(f"{page_num}页处理完毕")

            # 保存输出PDF
            output_doc.save(str(output_path))
            output_doc.close()
            pdf_doc.close()

            logger.info(f"PDF: {output_path}水印添加完毕")

            return output_path

        except Exception as e:
            logger.error(f"Failed to add invisible watermark to PDF: {e}")
            raise WatermarkEmbeddingError(f"Failed to add invisible watermark: {e}")


    def add_digital_signature(self, file_path: Path, certificate_path: Path, password: str) -> Path:
        """Add PKCS#7 digital signature to PDF document."""
        output_path = self.generate_output_path(file_path, "_signed")
        
        try:
            # Note: This is a simplified implementation
            # Full PKCS#7 signature requires cryptographic libraries like cryptography or pyOpenSSL
            
            # For now, we'll add signature metadata and a signature field
            pdf_doc = fitz.open(str(file_path))
            
            # Add signature metadata
            metadata = pdf_doc.metadata
            metadata['author'] = 'Digital Signature System'
            metadata['subject'] = f'Digitally signed with certificate: {certificate_path.name}'
            metadata['creator'] = 'Watermark System'
            metadata['producer'] = 'PDF Processor v1.0'
            pdf_doc.set_metadata(metadata)
            
            # Add signature appearance on first page
            if len(pdf_doc) > 0:
                page = pdf_doc[0]
                self._add_signature_appearance(page, certificate_path)
            
            # Save signed PDF
            pdf_doc.save(str(output_path))
            pdf_doc.close()
            
            logger.info(f"Added digital signature to PDF: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to add digital signature to PDF: {e}")
            raise DigitalSignatureError(f"Failed to add digital signature: {e}")

    def extract_invisible_watermark(self, file_path: Path,output_path: Path, page=[1,1],mode = 'img',wm_shape=None) -> Optional[str]:
        assert wm_shape is not None, 'wm_shape needed'
        try:
            pdf_doc = fitz.open(str(file_path))


            # 从指定范围提取水印
            for i in range(page[0],page[1]):
                page = pdf_doc[i]

                # 将页面渲染为图像（300 DPI）
                mat = fitz.Matrix(300/72, 300/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # 转换为numpy数组
                img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                img_array = img_data.reshape(pix.height, pix.width, pix.n)

                #颜色空间转换 RGB -> BGR
                # if uint8_img.shape[2] == 3:  # 确保是彩色图片
                uint8_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                self.imageProcessor.extract(embed_img=uint8_img,wm_shape=wm_shape,out_wm_name=output_path/f'page{i}.png')


            pdf_doc.close()

        except Exception as e:
            logger.error(f"Failed to extract FFT invisible watermark from PDF: {e}")





    def crop_pdf(self,input_path: Path, pages_to_keep: list, output_name: str = "cropped_pymupdf.pdf"):
        """
        使用 PyMuPDF 提取 PDF 的特定页码。

        :param input_path: PDF 文件的 pathlib.Path 对象
        :param pages_to_keep: 要保留的页码列表（从 1 开始计数）
        :param output_name: 输出文件名称
        """
        try:
            # 打开文档
            doc = fitz.open(input_path)

            # 将人类阅读的页码 (1-based) 转换为索引 (0-based)
            # 并过滤掉超出范围的页码
            total_pages = len(doc)
            indices = [p - 1 for p in pages_to_keep if 1 <= p <= total_pages]

            if not indices:
                print("错误：没有有效的页码可供提取。")
                return None

            # 使用 select() 直接修改内存中的文档对象，只保留选定页面
            doc.select(indices)

            # 确定输出路径
            output_path = input_path.parent / output_name

            # 保存文档 (garbage=4 表示进行高压缩和清理)
            doc.save(output_path, garbage=4, deflate=True)
            doc.close()

            print(f"提取完成！文件已保存至: {output_path}")
            return output_path

        except Exception as e:
            print(f"PyMuPDF 处理出错: {e}")
            return None



    def _add_pdf_text_watermark(self, page: fitz.Page, layer_config: Dict[str, Any],
                                user_id: str, timestamp: datetime):
        """Add text watermark to PDF page (supports tiling)."""
        import math  # 确保导入 math

        # 1. 准备内容
        content = layer_config.get('content', '')
        content = content.replace('{user_id}', user_id)
        content = content.replace('{timestamp}', timestamp.strftime('%Y-%m-%d %H:%M:%S'))

        # 2. 获取页面尺寸
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        # 3. 获取字体配置
        font_config = layer_config.get('font', {})
        font_size = font_config.get('size', 12)
        font_color = self._parse_color(font_config.get('color', '#FF0000'))
        opacity = font_config.get('opacity', 0.5)

        # 4. 获取位置配置
        position_config = layer_config.get('position', {})
        rotation = position_config.get('rotation', 45)
        is_tiled = position_config.get('tiled', False)  # 默认为 False (单个)
        gap_x = position_config.get('gap_x', 50)  # 默认左右间隔 50
        gap_y = position_config.get('gap_y', 50)  # 默认上下间隔 50

        # 5. 计算文本宽度 (用于计算间隔步长)
        # 使用 PyMuPDF 内置的标准字体来估算长度，确保排版整齐
        temp_font = fitz.Font("helv")
        text_length = temp_font.text_length(content, fontsize=font_size)
        text_height = font_size * 1.2  # 估算行高

        # 创建 TextWriter
        text_writer = fitz.TextWriter(page_rect)

        if is_tiled:
            # --- 平铺模式逻辑 ---

            # 计算步长 = 文字本身长度 + 用户定义的间隔
            step_x = text_length + gap_x
            step_y = text_height + gap_y

            # 计算覆盖范围
            # 如果有旋转，我们需要在一个比页面大得多的区域绘制，
            # 否则旋转后角落会是空的。我们用页面对角线长度作为基准。
            diagonal = math.sqrt(page_width ** 2 + page_height ** 2)

            # 定义绘制的起始和结束坐标 (以中心为原点向外扩展)
            # 范围设为 -diagonal 到 +diagonal 确保旋转后能覆盖全图
            start_y = -diagonal
            end_y = diagonal
            start_x = -diagonal
            end_x = diagonal

            # 循环生成网格
            curr_y = start_y
            while curr_y < end_y:
                curr_x = start_x
                while curr_x < end_x:
                    point = fitz.Point(curr_x, curr_y)
                    text_writer.append(point, content, fontsize=font_size)
                    curr_x += step_x
                curr_y += step_y

            # 设置旋转中心为页面中心
            center = fitz.Point(page_width / 2, page_height / 2)
            matrix = fitz.Matrix(rotation)

            # 写入页面 (带旋转和颜色)
            text_writer.write_text(
                page,
                morph=(center, matrix),
                opacity=opacity,
                color=font_color
            )

        else:
            # --- 单个水印逻辑 (保持原有逻辑，已修复 color bug) ---

            # 计算单个位置
            x_pos, y_pos = self._calculate_position(
                position_config, page_width, page_height, content, font_size
            )

            point = fitz.Point(x_pos, y_pos)
            text_writer.append(point, content, fontsize=font_size)

            if rotation != 0:
                center = fitz.Point(page_width / 2, page_height / 2)
                matrix = fitz.Matrix(rotation)
                text_writer.write_text(page, morph=(center, matrix), opacity=opacity, color=font_color)
            else:
                text_writer.write_text(page, opacity=opacity, color=font_color)
    
    def _add_pdf_qrcode_watermark(self, page: fitz.Page, layer_config: Dict[str, Any],
                                user_id: str, timestamp: datetime):
        """Add QR code watermark to PDF page."""
        import qrcode
        from PIL import Image
        
        # Replace template variables in QR code content
        content = layer_config.get('content', '')
        content = content.replace('{user_id}', user_id)
        content = content.replace('{timestamp}', timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        content = content.replace('{file_id}', 'PDF_FILE')  # Could be passed as parameter
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(content)
        qr.make(fit=True)
        
        # Create QR code image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        qr_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Get position configuration
        position_config = layer_config.get('position', {})
        size = layer_config.get('size', 50)
        
        # Calculate position
        page_rect = page.rect
        if position_config.get('x') == 'bottom-right' and position_config.get('y') == 'bottom-right':
            x_pos = page_rect.width - size - 10
            y_pos = page_rect.height - size - 10
        else:
            x_pos = 10
            y_pos = 10
            
        # Insert QR code image
        img_rect = fitz.Rect(x_pos, y_pos, x_pos + size, y_pos + size)
        page.insert_image(img_rect, stream=img_buffer.getvalue())
    
    def _add_pdf_background_watermark(self, page: fitz.Page, watermark_data: str):
        """Add invisible background watermark using rasterization technique."""
        # Encode watermark data
        encoded_data = base64.b64encode(watermark_data.encode()).decode()
        
        # Create a very subtle pattern based on the watermark data
        # This is a simplified implementation - real steganography would be more sophisticated
        
        # Get page dimensions
        page_rect = page.rect
        
        # Create invisible pattern based on watermark data hash
        data_hash = hashlib.md5(encoded_data.encode()).hexdigest()
        
        # Add invisible elements based on hash
        for i, char in enumerate(data_hash[:8]):  # Use first 8 characters
            # Convert hex character to position
            hex_val = int(char, 16)
            x_offset = (hex_val % 4) * (page_rect.width / 4)
            y_offset = (hex_val // 4) * (page_rect.height / 4)
            
            # Add nearly invisible point
            point = fitz.Point(x_offset, y_offset)
            # Use very light color (almost white)
            color = (0.999, 0.999, 0.999)  # RGB values close to white
            
            # Draw tiny invisible rectangle
            rect = fitz.Rect(point.x, point.y, point.x + 0.1, point.y + 0.1)
            page.draw_rect(rect, color=color, fill=color, width=0.01)
        
        # Store encoded data in page metadata (if supported)
        try:
            # Add to page annotations as invisible annotation
            annot = page.add_text_annot(fitz.Point(0, 0), encoded_data)
            annot.set_info(content=encoded_data)
            annot.update()
        except Exception:
            pass  # Ignore if annotation fails
    
    def _extract_pdf_background_watermark(self, page: fitz.Page) -> Optional[str]:
        """Extract invisible watermark from PDF page."""
        try:
            # Try to extract from annotations first
            for annot in page.annots():
                if annot.type[1] == 'Text':  # Text annotation
                    content = annot.info.get('content', '')
                    if content:
                        try:
                            # Try to decode as base64
                            decoded_data = base64.b64decode(content).decode()
                            return decoded_data
                        except Exception:
                            continue
            
            # If no annotation found, try to analyze the invisible pattern
            # This would require more sophisticated image analysis
            # For now, return None if no annotation watermark found
            
        except Exception as e:
            logger.error(f"Failed to extract background watermark: {e}")
            
        return None
    
    def _add_signature_appearance(self, page: fitz.Page, certificate_path: Path):
        """Add visual signature appearance to PDF page."""
        # Get page dimensions
        page_rect = page.rect
        
        # Create signature box in bottom right corner
        sig_width = 150
        sig_height = 50
        sig_x = page_rect.width - sig_width - 20
        sig_y = page_rect.height - sig_height - 20
        
        sig_rect = fitz.Rect(sig_x, sig_y, sig_x + sig_width, sig_y + sig_height)
        
        # Draw signature box
        page.draw_rect(sig_rect, color=(0, 0, 0), width=1)
        
        # Add signature text
        sig_text = f"Digitally Signed\n{certificate_path.stem}\n{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        text_writer = fitz.TextWriter(page_rect)
        text_writer.append(
            fitz.Point(sig_x + 5, sig_y + 15),
            sig_text,
            fontsize=8,
            color=(0, 0, 0)
        )
        text_writer.write_text(page)
    
    def _string_to_bits(self, text: str) -> np.ndarray:
        """将字符串转换为比特序列"""
        # 将字符串编码为字节
        byte_data = text.encode('utf-8')
        
        # 转换为比特数组
        bits = []
        for byte in byte_data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        
        return np.array(bits, dtype=np.uint8)
    
    def _bits_to_string(self, bits: np.ndarray) -> str:
        """将比特序列转换回字符串"""
        try:
            # 确保比特数是8的倍数
            num_bits = len(bits)
            num_bytes = num_bits // 8
            
            if num_bytes == 0:
                return ""
            
            # 转换比特为字节
            byte_data = bytearray()
            for i in range(num_bytes):
                byte_val = 0
                for j in range(8):
                    bit_idx = i * 8 + j
                    if bit_idx < num_bits:
                        byte_val = (byte_val << 1) | int(bits[bit_idx])
                byte_data.append(byte_val)
            
            # 尝试解码为字符串
            # 首先尝试UTF-8
            try:
                text = byte_data.decode('utf-8', errors='strict')
                # 移除不可打印字符
                text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t ')
                return text
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试ASCII
                try:
                    text = byte_data.decode('ascii', errors='ignore')
                    text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t ')
                    return text
                except:
                    # 如果都失败，返回十六进制表示
                    return byte_data.hex()
            
        except Exception as e:
            logger.error(f"Failed to convert bits to string: {e}")
            # 返回比特的十六进制表示作为后备
            try:
                num_bytes = len(bits) // 8
                byte_data = bytearray()
                for i in range(num_bytes):
                    byte_val = 0
                    for j in range(8):
                        bit_idx = i * 8 + j
                        if bit_idx < len(bits):
                            byte_val = (byte_val << 1) | int(bits[bit_idx])
                    byte_data.append(byte_val)
                return byte_data.hex()
            except:
                return ""
    
    def _parse_color(self, color_str: str) -> Tuple[float, float, float]:
        """Parse color string to RGB tuple."""
        if color_str.startswith('#'):
            # Hex color
            hex_color = color_str[1:]
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                return (r, g, b)
        
        # Default to red if parsing fails
        return (1.0, 0.0, 0.0)
    
    def _calculate_position(self, position_config: Dict[str, Any], page_width: float, 
                          page_height: float, text: str, font_size: int) -> Tuple[float, float]:
        """Calculate text position based on configuration."""
        x_config = position_config.get('x', 'center')
        y_config = position_config.get('y', 'center')
        
        # Estimate text width (rough approximation)
        text_width = len(text) * font_size * 0.6
        text_height = font_size
        
        # Calculate X position
        if x_config == 'left':
            x_pos = 20
        elif x_config == 'right':
            x_pos = page_width - text_width - 20
        else:  # center
            x_pos = (page_width - text_width) / 2
            
        # Calculate Y position
        if y_config == 'top':
            y_pos = 20 + text_height
        elif y_config == 'bottom':
            y_pos = page_height - 20
        else:  # center
            y_pos = page_height / 2
            
        return (x_pos, y_pos)
    
    def verify_digital_signature(self, file_path: Path) -> Dict[str, Any]:
        """Verify digital signature in PDF document."""
        try:
            pdf_doc = fitz.open(str(file_path))
            
            # Check metadata for signature information
            metadata = pdf_doc.metadata
            
            # Look for signature-related metadata
            signature_info = {
                'is_signed': False,
                'signer': metadata.get('author', ''),
                'signature_date': metadata.get('creationDate', ''),
                'certificate_info': metadata.get('subject', ''),
                'valid': False  # Would require actual cryptographic verification
            }
            
            # Check if document appears to be signed
            if 'Digital Signature System' in signature_info['signer']:
                signature_info['is_signed'] = True
                
            # Check for signature annotations
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                for annot in page.annots():
                    if 'signature' in annot.info.get('content', '').lower():
                        signature_info['is_signed'] = True
                        break
                        
            pdf_doc.close()
            return signature_info
            
        except Exception as e:
            logger.error(f"Failed to verify PDF signature: {e}")
            return {'is_signed': False, 'error': str(e)}
    
    def get_pdf_info(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive PDF document information."""
        try:
            pdf_doc = fitz.open(str(file_path))
            
            info = {
                'page_count': len(pdf_doc),
                'metadata': pdf_doc.metadata,
                'is_encrypted': pdf_doc.needs_pass,
                'file_size': file_path.stat().st_size,
                'creation_date': pdf_doc.metadata.get('creationDate', ''),
                'modification_date': pdf_doc.metadata.get('modDate', ''),
                'author': pdf_doc.metadata.get('author', ''),
                'title': pdf_doc.metadata.get('title', ''),
                'subject': pdf_doc.metadata.get('subject', ''),
                'creator': pdf_doc.metadata.get('creator', ''),
                'producer': pdf_doc.metadata.get('producer', ''),
            }
            
            # Get page dimensions for first page
            if len(pdf_doc) > 0:
                first_page = pdf_doc[0]
                page_rect = first_page.rect
                info['page_dimensions'] = {
                    'width': page_rect.width,
                    'height': page_rect.height
                }
            
            pdf_doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            return {'error': str(e)}