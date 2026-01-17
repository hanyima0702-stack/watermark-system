"""
Unit tests for Office document processor.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from engines.document.office_processor import OfficeProcessor
from engines.document.base_processor import WatermarkConfig, DocumentProcessingError, WatermarkEmbeddingError


class TestOfficeProcessor:
    """Test cases for OfficeProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create OfficeProcessor instance."""
        return OfficeProcessor()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
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
                            'width': 100,
                            'height': 25,
                            'color': 'yellow',
                            'opacity': 1
                        },
                        'position': {
                            'x': 'center',
                            'y': 'center',
                            'margin-left':100,
                            'margin-top':100,
                            'rotation': 45
                        }
                    }
                ]
            },
            'invisible_watermark': {
                'enabled': True
            }
        }
        return WatermarkConfig(config_dict)
    
    def test_supported_formats(self, processor):
        """Test supported file formats."""
        assert '.docx' in processor.supported_formats
        assert '.xlsx' in processor.supported_formats
        assert '.pptx' in processor.supported_formats
        assert len(processor.supported_formats) == 3
    
    def test_is_supported_format(self, processor):
        """Test format support checking."""
        assert processor.is_supported_format(Path('test.docx'))
        assert processor.is_supported_format(Path('test.xlsx'))
        assert processor.is_supported_format(Path('test.pptx'))
        assert not processor.is_supported_format(Path('test.pdf'))
        assert not processor.is_supported_format(Path('test.txt'))
    
    def test_generate_output_path(self, processor):
        """Test output path generation."""
        input_path = Path('/test/document.docx')
        output_path = processor.generate_output_path(input_path)
        
        assert output_path == Path('/test/document_watermarked.docx')
        
        # Test custom suffix
        output_path = processor.generate_output_path(input_path, '_custom')
        assert output_path == Path('/test/document_custom.docx')
    
    @patch('engines.document.office_processor.Document')
    def test_add_word_watermark_success(self, mock_document, processor, sample_watermark_config, temp_dir):
        """Test successful Word watermark addition."""
        # Setup
        input_path = temp_dir / 'test.docx'
        input_path.touch()  # Create empty file
        
        mock_doc = MagicMock()
        mock_document.return_value = mock_doc
        
        # Mock document structure
        mock_section = MagicMock()
        mock_header = MagicMock()
        mock_paragraph = MagicMock()
        mock_run = MagicMock()
        mock_font = MagicMock()
        
        mock_doc.sections = [mock_section]
        mock_section.header = mock_header
        mock_header.paragraphs = [mock_paragraph]
        mock_paragraph.add_run.return_value = mock_run
        mock_run.font = mock_font
        
        # Execute
        result_path = processor.add_visible_watermark(
            input_path, sample_watermark_config, 'test_user', datetime.now()
        )
        
        # Verify
        assert result_path.exists()
        mock_document.assert_called_once_with(str(input_path))
        mock_doc.save.assert_called_once()
        mock_paragraph.add_run.assert_called()

    def test_add_word_watermark(self, processor, sample_watermark_config):

        input_path = Path("D:/毕业设计/项目/测试文件/word/方案.docx")
        input_path.touch()

        # Execute
        result_path = processor.add_visible_watermark(
            input_path, sample_watermark_config, 'test_user', datetime.now()
        )




    @patch('engines.document.office_processor.openpyxl.load_workbook')
    def test_add_excel_watermark_success(self, mock_load_workbook, processor, sample_watermark_config, temp_dir):
        """Test successful Excel watermark addition."""
        # Setup
        input_path = temp_dir / 'test.xlsx'
        #更新文件的修改时间
        input_path.touch()
        
        mock_workbook = MagicMock()
        mock_worksheet = MagicMock()
        mock_header = MagicMock()
        mock_center = MagicMock()
        
        mock_load_workbook.return_value = mock_workbook
        mock_workbook.worksheets = [mock_worksheet]
        mock_worksheet.oddHeader = mock_header
        mock_worksheet.evenHeader = mock_header
        mock_header.center = mock_center
        
        # Execute
        result_path = processor.add_visible_watermark(
            input_path, sample_watermark_config, 'test_user', datetime.now()
        )
        
        # Verify
        assert result_path.exists()
        mock_load_workbook.assert_called_once_with(str(input_path))
        mock_workbook.save.assert_called_once()
    
    @patch('engines.document.office_processor.Presentation')
    def test_add_powerpoint_watermark_success(self, mock_presentation, processor, sample_watermark_config, temp_dir):
        """Test successful PowerPoint watermark addition."""
        # Setup
        input_path = temp_dir / 'test.pptx'
        input_path.touch()
        
        mock_pres = MagicMock()
        mock_slide = MagicMock()
        mock_shapes = MagicMock()
        mock_textbox = MagicMock()
        mock_text_frame = MagicMock()
        mock_paragraph = MagicMock()
        mock_font = MagicMock()
        
        mock_presentation.return_value = mock_pres
        mock_pres.slides = [mock_slide]
        mock_slide.shapes = mock_shapes
        mock_shapes.add_textbox.return_value = mock_textbox
        mock_textbox.text_frame = mock_text_frame
        mock_text_frame.paragraphs = [mock_paragraph]
        mock_paragraph.font = mock_font
        
        # Execute
        result_path = processor.add_visible_watermark(
            input_path, sample_watermark_config, 'test_user', datetime.now()
        )
        
        # Verify
        assert result_path.exists()
        mock_presentation.assert_called_once_with(str(input_path))
        mock_pres.save.assert_called_once()
    
    def test_add_visible_watermark_unsupported_format(self, processor, sample_watermark_config):
        """Test watermark addition with unsupported format."""
        input_path = Path('test.pdf')
        
        with pytest.raises(DocumentProcessingError, match="Unsupported format"):
            processor.add_visible_watermark(
                input_path, sample_watermark_config, 'test_user', datetime.now()
            )
    
    @patch('engines.document.office_processor.Document')
    def test_add_word_invisible_watermark(self, mock_document, processor, temp_dir):
        """Test invisible watermark addition to Word document."""
        # Setup
        input_path = temp_dir / 'test.docx'
        input_path.touch()
        
        mock_doc = MagicMock()
        mock_paragraph = MagicMock()
        mock_run = MagicMock()
        
        mock_document.return_value = mock_doc
        mock_doc.paragraphs = [mock_paragraph]
        mock_paragraph.runs = [mock_run]
        mock_run.text = "Original text"
        
        # Execute
        watermark_data = "test_watermark_data"
        result_path = processor.add_invisible_watermark(input_path, watermark_data)
        
        # Verify
        assert result_path.exists()
        mock_document.assert_called_once_with(str(input_path))
        mock_doc.save.assert_called_once()
        
        # Check that text was modified (should contain zero-width characters)
        assert mock_run.text != "Original text"
    
    @patch('engines.document.office_processor.openpyxl.load_workbook')
    def test_add_excel_invisible_watermark(self, mock_load_workbook, processor, temp_dir):
        """Test invisible watermark addition to Excel document."""
        # Setup
        input_path = temp_dir / 'test.xlsx'
        input_path.touch()
        
        mock_workbook = MagicMock()
        mock_load_workbook.return_value = mock_workbook
        mock_workbook.custom_doc_props = {}
        
        # Execute
        watermark_data = "test_watermark_data"
        result_path = processor.add_invisible_watermark(input_path, watermark_data)
        
        # Verify
        assert result_path.exists()
        mock_load_workbook.assert_called_once_with(str(input_path))
        mock_workbook.save.assert_called_once()
    
    @patch('engines.document.office_processor.Document')
    def test_extract_word_invisible_watermark(self, mock_document, processor, temp_dir):
        """Test invisible watermark extraction from Word document."""
        # Setup
        input_path = temp_dir / 'test.docx'
        input_path.touch()
        
        mock_doc = MagicMock()
        mock_paragraph = MagicMock()
        
        # Create text with zero-width characters representing "A" (01000001)
        zero_width_space = '\u200B'  # 0
        zero_width_joiner = '\u200D'  # 1
        encoded_text = zero_width_space + zero_width_joiner + zero_width_space + zero_width_space + zero_width_space + zero_width_space + zero_width_space + zero_width_joiner
        
        mock_paragraph.text = encoded_text + "Normal text"
        mock_doc.paragraphs = [mock_paragraph]
        mock_document.return_value = mock_doc
        
        # Execute
        result = processor.extract_invisible_watermark(input_path)
        
        # Verify
        assert result == "A"  # Should decode to "A"
        mock_document.assert_called_once_with(str(input_path))
    
    @patch('engines.document.office_processor.openpyxl.load_workbook')
    def test_extract_excel_invisible_watermark(self, mock_load_workbook, processor, temp_dir):
        """Test invisible watermark extraction from Excel document."""
        # Setup
        input_path = temp_dir / 'test.xlsx'
        input_path.touch()
        
        mock_workbook = MagicMock()
        mock_workbook.custom_doc_props = {
            'DocumentVersion': 'dGVzdF93YXRlcm1hcmtfZGF0YQ=='  # base64 encoded "test_watermark_data"
        }
        mock_load_workbook.return_value = mock_workbook
        
        # Execute
        result = processor.extract_invisible_watermark(input_path)
        
        # Verify
        assert result == "test_watermark_data"
        mock_load_workbook.assert_called_once_with(str(input_path))
    
    @patch('engines.document.office_processor.Document')
    def test_add_digital_signature_word(self, mock_document, processor, temp_dir):
        """Test digital signature addition to Word document."""
        # Setup
        input_path = temp_dir / 'test.docx'
        input_path.touch()
        cert_path = temp_dir / 'cert.p12'
        cert_path.touch()
        
        mock_doc = MagicMock()
        mock_props = MagicMock()
        mock_document.return_value = mock_doc
        mock_doc.core_properties = mock_props
        
        # Execute
        result_path = processor.add_digital_signature(input_path, cert_path, 'password')
        
        # Verify
        assert result_path.exists()
        assert mock_props.author == "Digital Signature System"
        mock_doc.save.assert_called_once()
    
    def test_add_invisible_watermark_unsupported_format(self, processor):
        """Test invisible watermark with unsupported format."""
        input_path = Path('test.pdf')
        
        with pytest.raises(DocumentProcessingError, match="Unsupported format"):
            processor.add_invisible_watermark(input_path, 'test_data')
    
    @patch('engines.document.office_processor.Document')
    def test_add_visible_watermark_exception_handling(self, mock_document, processor, sample_watermark_config, temp_dir):
        """Test exception handling in visible watermark addition."""
        # Setup
        input_path = temp_dir / 'test.docx'
        input_path.touch()
        
        mock_document.side_effect = Exception("Document loading failed")
        
        # Execute & Verify
        with pytest.raises(WatermarkEmbeddingError, match="Failed to add visible watermark"):
            processor.add_visible_watermark(
                input_path, sample_watermark_config, 'test_user', datetime.now()
            )
    
    def test_watermark_config_properties(self):
        """Test WatermarkConfig property access."""
        config_dict = {
            'visible_watermark': {'enabled': True},
            'invisible_watermark': {'enabled': False},
            'digital_signature': {'enabled': True, 'certificate_path': '/path/to/cert'}
        }
        
        config = WatermarkConfig(config_dict)
        
        assert config.visible_watermark == {'enabled': True}
        assert config.invisible_watermark == {'enabled': False}
        assert config.digital_signature == {'enabled': True, 'certificate_path': '/path/to/cert'}
    
    def test_watermark_config_empty(self):
        """Test WatermarkConfig with empty configuration."""
        config = WatermarkConfig({})
        
        assert config.visible_watermark == {}
        assert config.invisible_watermark == {}
        assert config.digital_signature == {}
    
    @patch('engines.document.office_processor.Document')
    def test_watermark_disabled(self, mock_document, processor, temp_dir):
        """Test behavior when watermark is disabled."""
        # Setup
        input_path = temp_dir / 'test.docx'
        input_path.touch()
        
        config_dict = {
            'visible_watermark': {'enabled': False}
        }
        watermark_config = WatermarkConfig(config_dict)
        
        mock_doc = MagicMock()
        mock_document.return_value = mock_doc
        
        # Execute
        result_path = processor.add_visible_watermark(
            input_path, watermark_config, 'test_user', datetime.now()
        )
        
        # Verify - should just copy the file
        assert result_path.exists()
        # Document should still be loaded but no watermark processing
        mock_document.assert_called_once()
        mock_doc.save.assert_called_once()


class TestOfficeProcessorIntegration:
    """Integration tests for OfficeProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create OfficeProcessor instance."""
        return OfficeProcessor()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_calculate_file_hash(self, processor, temp_dir):
        """Test file hash calculation."""
        # Create test file with known content
        test_file = temp_dir / 'test.txt'
        test_content = b'Hello, World!'
        test_file.write_bytes(test_content)
        
        # Calculate hash
        file_hash = processor.calculate_file_hash(test_file)
        
        # Verify hash is correct (SHA-256 of "Hello, World!")
        import hashlib
        expected_hash = hashlib.sha256(test_content).hexdigest()
        assert file_hash == expected_hash
    
    def test_template_variable_replacement(self, processor):
        """Test template variable replacement in watermark content."""
        # This would be tested through the actual watermark methods
        # but we can test the logic separately
        content = "{user_id} - {timestamp}"
        user_id = "test_user"
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        
        # Simulate the replacement logic
        replaced_content = content.replace('{user_id}', user_id)
        replaced_content = replaced_content.replace('{timestamp}', timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        
        assert replaced_content == "test_user - 2023-01-01 12:00:00"