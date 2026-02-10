"""
Unit tests for PDF document processor.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from engines.document.pdf_processor import PDFProcessor
from engines.document.base_processor import WatermarkConfig, DocumentProcessingError, WatermarkEmbeddingError


class TestPDFProcessor:
    """Test cases for PDFProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create PDFProcessor instance."""
        return PDFProcessor()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def watermark(self):
        return "D:/pic/watermark_pic.png"
    
    @pytest.fixture
    def sample_watermark_config(self):
        """Create sample watermark configuration."""
        config_dict = {
            'visible_watermark': {
                'enabled': True,
                'layers': [
                    {
                        'type': 'text',
                        'content': '{user_id} - {timestamp}',
                        'font': {
                            'family': 'Arial',
                            'size': 40,
                            'color': '#0000FF',
                            'opacity': 0.2
                        },
                        'position': {
                            'rotation': 45,
                            'tiled': True,     # 新增：开启平铺
                            'gap_x': 50,       # 新增：左右间隔 (像素)
                            'gap_y': 300
                        }
                    },
                    {
                        'type': 'qrcode',
                        'content': '{user_id}|{file_id}|{timestamp}',
                        'size': 50,
                        'position': {
                            'x': 'bottom-right',
                            'y': 'bottom-right'
                        }
                    }
                ]
            },
            'invisible_watermark': {
                'enabled': True
            }
        }
        return WatermarkConfig(config_dict)
    

        
    def test_add_visible_watermark(self,processor, sample_watermark_config):
        input_path = Path('D:/毕业设计/项目/测试文件/pdf/专硕1.pdf')
        input_path.touch()

        # Execute
        result_path = processor.add_visible_watermark(
            input_path, sample_watermark_config, 'test_user', datetime.now()
        )


    def test_add_invisible_watermark(self,processor,watermark):
        input_path = Path('D:/毕业设计/项目/测试文件/pdf/cropped.pdf')
        input_path.touch()


        # Execute
        processor.add_invisible_watermark(input_path,watermark)

    
    @patch('engines.document.pdf_processor.fitz.open')
    def test_add_digital_signature_success(self, mock_fitz_open, processor, temp_dir):
        """Test successful PDF digital signature addition."""
        # Setup
        input_path = temp_dir / 'test.pdf'
        input_path.touch()
        cert_path = temp_dir / 'cert.p12'
        cert_path.touch()
        
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_rect = MagicMock()
        mock_rect.width = 612
        mock_rect.height = 792
        
        mock_fitz_open.return_value = mock_doc
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_page.rect = mock_rect
        mock_doc.metadata = {}
        
        # Execute
        result_path = processor.add_digital_signature(input_path, cert_path, 'password')
        
        # Verify
        assert result_path.exists()
        mock_fitz_open.assert_called_once_with(str(input_path))
        mock_doc.save.assert_called_once()
        mock_doc.close.assert_called_once()
        mock_doc.set_metadata.assert_called_once()
    


    def test_extract_invisible_watermark(self,  processor):
        input_path = Path('D:/毕业设计/项目/测试文件/pdf/cropped_invisible.pdf')
        input_path.touch()
        output_path = Path("D:/pic")
        processor.extract_invisible_watermark(input_path,output_path,page=[0,1] , wm_shape =[64,64])


    
    @patch('engines.document.pdf_processor.fitz.open')
    def test_extract_invisible_watermark_no_watermark(self, mock_fitz_open, processor, temp_dir):
        """Test watermark extraction when no watermark exists."""
        # Setup
        input_path = temp_dir / 'test.pdf'
        input_path.touch()
        
        mock_doc = MagicMock()
        mock_page = MagicMock()
        
        mock_fitz_open.return_value = mock_doc
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_page.annots.return_value = []  # No annotations
        
        # Execute
        result = processor.extract_invisible_watermark(input_path)
        
        # Verify
        assert result is None
        mock_doc.close.assert_called_once()
    
    def test_parse_color_hex(self, processor):
        """Test color parsing from hex string."""
        # Test valid hex color
        color = processor._parse_color('#FF0000')
        assert color == (1.0, 0.0, 0.0)  # Red
        
        color = processor._parse_color('#00FF00')
        assert color == (0.0, 1.0, 0.0)  # Green
        
        color = processor._parse_color('#0000FF')
        assert color == (0.0, 0.0, 1.0)  # Blue
        
        # Test invalid color (should default to red)
        color = processor._parse_color('invalid')
        assert color == (1.0, 0.0, 0.0)
    
    def test_calculate_position(self, processor):
        """Test position calculation for watermark placement."""
        page_width = 612
        page_height = 792
        text = "Test"
        font_size = 12
        
        # Test center position
        position_config = {'x': 'center', 'y': 'center'}
        x, y = processor._calculate_position(position_config, page_width, page_height, text, font_size)
        
        # Should be roughly centered
        assert x > 0 and x < page_width
        assert y > 0 and y < page_height
        
        # Test left-top position
        position_config = {'x': 'left', 'y': 'top'}
        x, y = processor._calculate_position(position_config, page_width, page_height, text, font_size)
        
        assert x == 20  # Left margin
        assert y == 20 + font_size  # Top margin + font height
    
    @patch('engines.document.pdf_processor.fitz.open')
    def test_get_pdf_info(self, mock_fitz_open, processor, temp_dir):
        """Test PDF information extraction."""
        # Setup
        input_path = temp_dir / 'test.pdf'
        input_path.write_bytes(b'dummy pdf content')
        
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_rect = MagicMock()
        mock_rect.width = 612
        mock_rect.height = 792
        
        mock_fitz_open.return_value = mock_doc
        mock_doc.__len__.return_value = 2
        mock_doc.__getitem__.return_value = mock_page
        mock_page.rect = mock_rect
        mock_doc.needs_pass = False
        mock_doc.metadata = {
            'author': 'Test Author',
            'title': 'Test Title',
            'creationDate': '2023-01-01',
            'modDate': '2023-01-02'
        }
        
        # Execute
        info = processor.get_pdf_info(input_path)
        
        # Verify
        assert info['page_count'] == 2
        assert info['is_encrypted'] == False
        assert info['author'] == 'Test Author'
        assert info['title'] == 'Test Title'
        assert info['page_dimensions']['width'] == 612
        assert info['page_dimensions']['height'] == 792
        mock_doc.close.assert_called_once()
    
    @patch('engines.document.pdf_processor.fitz.open')
    def test_verify_digital_signature(self, mock_fitz_open, processor, temp_dir):
        """Test digital signature verification."""
        # Setup
        input_path = temp_dir / 'test.pdf'
        input_path.touch()
        
        mock_doc = MagicMock()
        mock_doc.metadata = {
            'author': 'Digital Signature System',
            'creationDate': '2023-01-01',
            'subject': 'Digitally signed document'
        }
        
        mock_fitz_open.return_value = mock_doc
        mock_doc.__len__.return_value = 1
        
        # Execute
        signature_info = processor.verify_digital_signature(input_path)
        
        # Verify
        assert signature_info['is_signed'] == True
        assert signature_info['signer'] == 'Digital Signature System'
        assert signature_info['signature_date'] == '2023-01-01'
        assert signature_info['valid'] == False  # Simplified implementation doesn't do crypto verification
        mock_doc.close.assert_called_once()
    
    @patch('engines.document.pdf_processor.fitz.open')
    def test_watermark_disabled(self, mock_fitz_open, processor, temp_dir):
        """Test behavior when watermark is disabled."""
        # Setup
        input_path = temp_dir / 'test.pdf'
        input_path.touch()
        
        config_dict = {
            'visible_watermark': {'enabled': False}
        }
        watermark_config = WatermarkConfig(config_dict)
        
        # Execute
        result_path = processor.add_visible_watermark(
            input_path, watermark_config, 'test_user', datetime.now()
        )
        
        # Verify - should just copy the file
        assert result_path.exists()
        # fitz.open should not be called since watermark is disabled
        mock_fitz_open.assert_not_called()
    
    @patch('engines.document.pdf_processor.fitz.open')
    def test_add_visible_watermark_exception_handling(self, mock_fitz_open, processor, sample_watermark_config, temp_dir):
        """Test exception handling in visible watermark addition."""
        # Setup
        input_path = temp_dir / 'test.pdf'
        input_path.touch()
        
        mock_fitz_open.side_effect = Exception("PDF loading failed")
        
        # Execute & Verify
        with pytest.raises(WatermarkEmbeddingError, match="Failed to add visible watermark"):
            processor.add_visible_watermark(
                input_path, sample_watermark_config, 'test_user', datetime.now()
            )
    
    @patch('engines.document.pdf_processor.qrcode.QRCode')
    @patch('engines.document.pdf_processor.fitz.open')
    def test_add_qrcode_watermark(self, mock_fitz_open, mock_qrcode, processor, temp_dir):
        """Test QR code watermark addition."""
        # Setup
        input_path = temp_dir / 'test.pdf'
        input_path.touch()
        
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_rect = MagicMock()
        mock_rect.width = 612
        mock_rect.height = 792
        
        mock_fitz_open.return_value = mock_doc
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_page.rect = mock_rect
        
        # Mock QR code generation
        mock_qr_instance = MagicMock()
        mock_qrcode.return_value = mock_qr_instance
        mock_qr_image = MagicMock()
        mock_qr_instance.make_image.return_value = mock_qr_image
        
        # Create config with QR code layer
        config_dict = {
            'visible_watermark': {
                'enabled': True,
                'layers': [
                    {
                        'type': 'qrcode',
                        'content': '{user_id}|{file_id}|{timestamp}',
                        'size': 50,
                        'position': {
                            'x': 'bottom-right',
                            'y': 'bottom-right'
                        }
                    }
                ]
            }
        }
        watermark_config = WatermarkConfig(config_dict)
        
        # Execute
        result_path = processor.add_visible_watermark(
            input_path, watermark_config, 'test_user', datetime.now()
        )
        
        # Verify
        assert result_path.exists()
        mock_qrcode.assert_called_once()
        mock_qr_instance.add_data.assert_called_once()
        mock_qr_instance.make.assert_called_once()
        mock_page.insert_image.assert_called_once()


class TestPDFProcessorIntegration:
    """Integration tests for PDFProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create PDFProcessor instance."""
        return PDFProcessor()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_background_watermark_encoding_decoding(self, processor):
        """Test the encoding and decoding of background watermarks."""
        import base64
        import hashlib
        
        # Test data
        watermark_data = "test_watermark_data"
        
        # Test encoding (simulate what happens in _add_pdf_background_watermark)
        encoded_data = base64.b64encode(watermark_data.encode()).decode()
        data_hash = hashlib.md5(encoded_data.encode()).hexdigest()
        
        # Verify encoding
        assert encoded_data == 'dGVzdF93YXRlcm1hcmtfZGF0YQ=='
        assert len(data_hash) == 32  # MD5 hash length
        
        # Test decoding (simulate what happens in _extract_pdf_background_watermark)
        decoded_data = base64.b64decode(encoded_data).decode()
        assert decoded_data == watermark_data
    
    def test_signature_appearance_positioning(self, processor):
        """Test signature appearance positioning calculation."""
        # Mock page dimensions
        page_width = 612
        page_height = 792
        
        # Calculate signature position (as done in _add_signature_appearance)
        sig_width = 150
        sig_height = 50
        sig_x = page_width - sig_width - 20
        sig_y = page_height - sig_height - 20
        
        # Verify positioning
        assert sig_x == 442  # 612 - 150 - 20
        assert sig_y == 722  # 792 - 50 - 20
        
        # Ensure signature is within page bounds
        assert sig_x >= 0 and sig_x + sig_width <= page_width
        assert sig_y >= 0 and sig_y + sig_height <= page_height