"""
Unit tests for document conversion service.
"""
import pytest
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from engines.document.conversion_service import (
    DocumentConversionService, LibreOfficeConverter, ConversionFormat,
    ConversionJob, ConversionQualityChecker, DocumentProcessingError
)


class TestConversionFormat:
    """Test ConversionFormat enum."""
    
    def test_format_values(self):
        """Test format enum values."""
        assert ConversionFormat.PDF.value == "pdf"
        assert ConversionFormat.DOCX.value == "docx"
        assert ConversionFormat.XLSX.value == "xlsx"
        assert ConversionFormat.PPTX.value == "pptx"


class TestConversionJob:
    """Test ConversionJob dataclass."""
    
    def test_job_creation(self):
        """Test job creation with required fields."""
        input_path = Path('input.docx')
        output_path = Path('output.pdf')
        target_format = ConversionFormat.PDF
        options = {'quality': 'high'}
        
        job = ConversionJob(
            input_path=input_path,
            output_path=output_path,
            target_format=target_format,
            options=options
        )
        
        assert job.input_path == input_path
        assert job.output_path == output_path
        assert job.target_format == target_format
        assert job.options == options
        assert job.status == "pending"
        assert job.error_message is None
    
    def test_job_duration_calculation(self):
        """Test duration calculation."""
        job = ConversionJob(
            input_path=Path('input.docx'),
            output_path=Path('output.pdf'),
            target_format=ConversionFormat.PDF,
            options={}
        )
        
        # No duration when times not set
        assert job.duration is None
        
        # Set times
        job.start_time = 100.0
        job.end_time = 105.5
        
        assert job.duration == 5.5


class TestConversionQualityChecker:
    """Test ConversionQualityChecker."""
    
    @pytest.fixture
    def checker(self):
        """Create quality checker instance."""
        return ConversionQualityChecker()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_check_conversion_quality_success(self, checker, temp_dir):
        """Test successful quality check."""
        # Create test files
        input_file = temp_dir / 'input.docx'
        output_file = temp_dir / 'output.pdf'
        
        input_file.write_bytes(b'x' * 1000)  # 1KB input
        output_file.write_bytes(b'y' * 800)   # 0.8KB output
        
        # Check quality
        report = checker.check_conversion_quality(
            input_file, output_file, ConversionFormat.PDF
        )
        
        assert report['is_valid'] == True
        assert len(report['issues']) == 0
        assert report['metrics']['input_size'] == 1000
        assert report['metrics']['output_size'] == 800
        assert report['metrics']['size_ratio'] == 0.8
    
    def test_check_conversion_quality_file_too_small(self, checker, temp_dir):
        """Test quality check with output file too small."""
        # Create test files
        input_file = temp_dir / 'input.docx'
        output_file = temp_dir / 'output.pdf'
        
        input_file.write_bytes(b'x' * 1000)  # 1KB input
        output_file.write_bytes(b'y' * 50)    # 0.05KB output (5% ratio)
        
        # Check quality
        report = checker.check_conversion_quality(
            input_file, output_file, ConversionFormat.PDF
        )
        
        assert report['is_valid'] == False
        assert any('too small' in issue for issue in report['issues'])
        assert report['metrics']['size_ratio'] == 0.05
    
    def test_check_conversion_quality_file_too_large(self, checker, temp_dir):
        """Test quality check with output file too large."""
        # Create test files
        input_file = temp_dir / 'input.docx'
        output_file = temp_dir / 'output.pdf'
        
        input_file.write_bytes(b'x' * 100)     # 100B input
        output_file.write_bytes(b'y' * 1500)   # 1.5KB output (15x ratio)
        
        # Check quality
        report = checker.check_conversion_quality(
            input_file, output_file, ConversionFormat.PDF
        )
        
        assert report['is_valid'] == False
        assert any('too large' in issue for issue in report['issues'])
        assert report['metrics']['size_ratio'] == 15.0
    
    def test_check_conversion_quality_missing_output(self, checker, temp_dir):
        """Test quality check with missing output file."""
        # Create only input file
        input_file = temp_dir / 'input.docx'
        output_file = temp_dir / 'nonexistent.pdf'
        
        input_file.write_bytes(b'x' * 1000)
        
        # Check quality
        report = checker.check_conversion_quality(
            input_file, output_file, ConversionFormat.PDF
        )
        
        assert report['is_valid'] == False
        assert any('does not exist' in issue for issue in report['issues'])
    
    @patch('engines.document.conversion_service.fitz.open')
    def test_validate_pdf(self, mock_fitz_open, checker, temp_dir):
        """Test PDF validation."""
        # Setup
        output_file = temp_dir / 'output.pdf'
        output_file.write_bytes(b'dummy pdf')
        
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 5  # 5 pages
        mock_fitz_open.return_value = mock_doc
        
        quality_report = {'metrics': {}, 'issues': []}
        
        # Execute
        checker._validate_pdf(output_file, quality_report)
        
        # Verify
        assert quality_report['metrics']['page_count'] == 5
        assert len(quality_report['issues']) == 0
        mock_doc.close.assert_called_once()
    
    @patch('engines.document.conversion_service.fitz.open')
    def test_validate_pdf_no_pages(self, mock_fitz_open, checker, temp_dir):
        """Test PDF validation with no pages."""
        # Setup
        output_file = temp_dir / 'output.pdf'
        output_file.write_bytes(b'dummy pdf')
        
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 0  # No pages
        mock_fitz_open.return_value = mock_doc
        
        quality_report = {'metrics': {}, 'issues': []}
        
        # Execute
        checker._validate_pdf(output_file, quality_report)
        
        # Verify
        assert quality_report['metrics']['page_count'] == 0
        assert any('no pages' in issue for issue in quality_report['issues'])
    
    def test_validate_text_file(self, checker, temp_dir):
        """Test text file validation."""
        # Create test text file
        output_file = temp_dir / 'output.txt'
        content = "Hello, World!\nThis is a test file.\n"
        output_file.write_text(content, encoding='utf-8')
        
        quality_report = {'metrics': {}, 'issues': []}
        
        # Execute
        checker._validate_text_file(output_file, quality_report)
        
        # Verify
        assert quality_report['metrics']['character_count'] == len(content)
        assert quality_report['metrics']['line_count'] == 3  # Including empty line at end
        assert len(quality_report['issues']) == 0
    
    def test_validate_text_file_empty(self, checker, temp_dir):
        """Test text file validation with empty file."""
        # Create empty text file
        output_file = temp_dir / 'output.txt'
        output_file.write_text('', encoding='utf-8')
        
        quality_report = {'metrics': {}, 'issues': []}
        
        # Execute
        checker._validate_text_file(output_file, quality_report)
        
        # Verify
        assert quality_report['metrics']['character_count'] == 0
        assert any('empty' in issue for issue in quality_report['issues'])


class TestLibreOfficeConverter:
    """Test LibreOfficeConverter."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('engines.document.conversion_service.shutil.which')
    def test_find_libreoffice_success(self, mock_which):
        """Test successful LibreOffice detection."""
        mock_which.return_value = '/usr/bin/libreoffice'
        
        converter = LibreOfficeConverter()
        
        assert converter.libreoffice_path == '/usr/bin/libreoffice'
    
    @patch('engines.document.conversion_service.shutil.which')
    def test_find_libreoffice_not_found(self, mock_which):
        """Test LibreOffice not found."""
        mock_which.return_value = None
        
        with pytest.raises(DocumentProcessingError, match="LibreOffice not found"):
            LibreOfficeConverter()
    
    def test_get_conversion_filter_pdf(self):
        """Test conversion filter selection for PDF."""
        converter = LibreOfficeConverter('/usr/bin/libreoffice')
        
        # Test Word to PDF
        filter_name = converter._get_conversion_filter('docx', ConversionFormat.PDF)
        assert filter_name == 'writer_pdf_Export'
        
        # Test Excel to PDF
        filter_name = converter._get_conversion_filter('xlsx', ConversionFormat.PDF)
        assert filter_name == 'calc_pdf_Export'
        
        # Test PowerPoint to PDF
        filter_name = converter._get_conversion_filter('pptx', ConversionFormat.PDF)
        assert filter_name == 'impress_pdf_Export'
    
    def test_get_conversion_filter_unsupported(self):
        """Test conversion filter for unsupported combination."""
        converter = LibreOfficeConverter('/usr/bin/libreoffice')
        
        with pytest.raises(DocumentProcessingError, match="No conversion filter found"):
            converter._get_conversion_filter('xyz', ConversionFormat.PDF)
    
    def test_build_conversion_command(self, temp_dir):
        """Test LibreOffice command building."""
        converter = LibreOfficeConverter('/usr/bin/libreoffice')
        
        input_path = temp_dir / 'input.docx'
        output_path = temp_dir / 'output.pdf'
        filter_name = 'writer_pdf_Export'
        options = {'no_logo': True, 'invisible': True}
        
        cmd = converter._build_conversion_command(input_path, output_path, filter_name, options)
        
        expected_cmd = [
            '/usr/bin/libreoffice',
            '--headless',
            '--convert-to', 'pdf:writer_pdf_Export',
            '--outdir', str(output_path.parent),
            str(input_path),
            '--nologo',
            '--invisible'
        ]
        
        assert cmd == expected_cmd
    
    @patch('engines.document.conversion_service.subprocess.run')
    def test_convert_document_success(self, mock_subprocess, temp_dir):
        """Test successful document conversion."""
        # Setup
        converter = LibreOfficeConverter('/usr/bin/libreoffice')
        input_path = temp_dir / 'input.docx'
        output_path = temp_dir / 'output.pdf'
        
        input_path.write_bytes(b'dummy docx content')
        
        # Mock successful subprocess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ''
        mock_subprocess.return_value = mock_result
        
        # Create output file (simulate LibreOffice creating it)
        def create_output(*args, **kwargs):
            output_path.write_bytes(b'dummy pdf content')
            return mock_result
        
        mock_subprocess.side_effect = create_output
        
        # Execute
        job = converter.convert_document(input_path, output_path, ConversionFormat.PDF)
        
        # Verify
        assert job.status == "completed"
        assert job.error_message is None
        assert job.start_time is not None
        assert job.end_time is not None
        assert job.duration > 0
        mock_subprocess.assert_called_once()
    
    @patch('engines.document.conversion_service.subprocess.run')
    def test_convert_document_subprocess_failure(self, mock_subprocess, temp_dir):
        """Test document conversion with subprocess failure."""
        # Setup
        converter = LibreOfficeConverter('/usr/bin/libreoffice')
        input_path = temp_dir / 'input.docx'
        output_path = temp_dir / 'output.pdf'
        
        input_path.write_bytes(b'dummy docx content')
        
        # Mock failed subprocess
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'LibreOffice conversion error'
        mock_subprocess.return_value = mock_result
        
        # Execute
        job = converter.convert_document(input_path, output_path, ConversionFormat.PDF)
        
        # Verify
        assert job.status == "failed"
        assert "LibreOffice conversion failed" in job.error_message
        assert job.start_time is not None
        assert job.end_time is not None
    
    @patch('engines.document.conversion_service.subprocess.run')
    def test_convert_document_timeout(self, mock_subprocess, temp_dir):
        """Test document conversion timeout."""
        # Setup
        converter = LibreOfficeConverter('/usr/bin/libreoffice', timeout=1)
        input_path = temp_dir / 'input.docx'
        output_path = temp_dir / 'output.pdf'
        
        input_path.write_bytes(b'dummy docx content')
        
        # Mock timeout
        mock_subprocess.side_effect = subprocess.TimeoutExpired('cmd', 1)
        
        # Execute
        job = converter.convert_document(input_path, output_path, ConversionFormat.PDF)
        
        # Verify
        assert job.status == "failed"
        assert "timeout" in job.error_message
    
    def test_convert_document_missing_input(self, temp_dir):
        """Test conversion with missing input file."""
        converter = LibreOfficeConverter('/usr/bin/libreoffice')
        input_path = temp_dir / 'nonexistent.docx'
        output_path = temp_dir / 'output.pdf'
        
        # Execute
        job = converter.convert_document(input_path, output_path, ConversionFormat.PDF)
        
        # Verify
        assert job.status == "failed"
        assert "does not exist" in job.error_message


class TestDocumentConversionService:
    """Test DocumentConversionService."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('engines.document.conversion_service.LibreOfficeConverter')
    def test_convert_single_document(self, mock_converter_class, temp_dir):
        """Test single document conversion."""
        # Setup
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        
        service = DocumentConversionService()
        
        input_path = temp_dir / 'input.docx'
        output_path = temp_dir / 'output.pdf'
        options = {'quality': 'high'}
        
        expected_job = ConversionJob(
            input_path=input_path,
            output_path=output_path,
            target_format=ConversionFormat.PDF,
            options=options
        )
        expected_job.status = "completed"
        
        mock_converter.convert_document.return_value = expected_job
        
        # Execute
        result = service.convert_single_document(input_path, output_path, ConversionFormat.PDF, options)
        
        # Verify
        assert result.status == "completed"
        mock_converter.convert_document.assert_called_once_with(
            input_path, output_path, ConversionFormat.PDF, options
        )
    
    @patch('engines.document.conversion_service.LibreOfficeConverter')
    def test_convert_batch(self, mock_converter_class, temp_dir):
        """Test batch document conversion."""
        # Setup
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        
        service = DocumentConversionService(max_workers=2)
        
        # Create conversion jobs
        conversion_jobs = [
            (temp_dir / 'input1.docx', temp_dir / 'output1.pdf', ConversionFormat.PDF, {}),
            (temp_dir / 'input2.docx', temp_dir / 'output2.pdf', ConversionFormat.PDF, {}),
        ]
        
        # Mock converter to return completed jobs
        def mock_convert(*args):
            job = ConversionJob(
                input_path=args[0],
                output_path=args[1],
                target_format=args[2],
                options=args[3]
            )
            job.status = "completed"
            return job
        
        mock_converter.convert_document.side_effect = mock_convert
        
        # Execute
        results = service.convert_batch(conversion_jobs)
        
        # Verify
        assert len(results) == 2
        assert all(job.status == "completed" for job in results)
        assert mock_converter.convert_document.call_count == 2
    
    @patch('engines.document.conversion_service.LibreOfficeConverter')
    def test_convert_directory(self, mock_converter_class, temp_dir):
        """Test directory conversion."""
        # Setup
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        
        service = DocumentConversionService()
        
        # Create test files
        input_dir = temp_dir / 'input'
        output_dir = temp_dir / 'output'
        input_dir.mkdir()
        output_dir.mkdir()
        
        (input_dir / 'doc1.docx').touch()
        (input_dir / 'doc2.docx').touch()
        (input_dir / 'sheet1.xlsx').touch()
        
        # Mock converter
        def mock_convert(*args):
            job = ConversionJob(
                input_path=args[0],
                output_path=args[1],
                target_format=args[2],
                options=args[3] or {}
            )
            job.status = "completed"
            return job
        
        mock_converter.convert_document.side_effect = mock_convert
        
        # Execute
        results = service.convert_directory(
            input_dir, output_dir, ConversionFormat.PDF, ['*.docx', '*.xlsx']
        )
        
        # Verify
        assert len(results) == 3  # 2 docx + 1 xlsx
        assert all(job.status == "completed" for job in results)
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        service = DocumentConversionService()
        
        formats = service.get_supported_formats()
        
        assert 'input_formats' in formats
        assert 'output_formats' in formats
        assert 'docx' in formats['input_formats']
        assert 'pdf' in formats['output_formats']
    
    def test_estimate_conversion_time(self):
        """Test conversion time estimation."""
        service = DocumentConversionService()
        
        # Test PDF conversion
        time_estimate = service.estimate_conversion_time(10.0, ConversionFormat.PDF)
        assert time_estimate == 20.0  # 10MB * 2.0 seconds/MB
        
        # Test minimum time
        time_estimate = service.estimate_conversion_time(1.0, ConversionFormat.PDF)
        assert time_estimate == 5.0  # Minimum 5 seconds
        
        # Test text conversion (faster)
        time_estimate = service.estimate_conversion_time(10.0, ConversionFormat.TXT)
        assert time_estimate == 5.0  # 10MB * 0.5 seconds/MB
    
    def test_cleanup_temp_files(self, temp_dir):
        """Test temporary file cleanup."""
        service = DocumentConversionService()
        
        # Create temporary directory with files
        temp_subdir = temp_dir / 'temp_conversion'
        temp_subdir.mkdir()
        (temp_subdir / 'temp_file.txt').touch()
        
        assert temp_subdir.exists()
        
        # Cleanup
        service.cleanup_temp_files(temp_subdir)
        
        # Verify cleanup
        assert not temp_subdir.exists()


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def service(self):
        """Create PDFProcessor instance."""
        return DocumentConversionService()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)


    def test_convert_to_pdf(self, service, temp_dir):
        """Test convert_to_pdf convenience function."""
        from engines.document.conversion_service import convert_to_pdf

        
        input_path = Path('D:/毕业设计/项目/测试文件/conversion/方案.docx')
        expected_output = input_path / 'input.pdf'
        
        expected_job = ConversionJob(
            input_path=input_path,
            output_path=expected_output,
            target_format=ConversionFormat.PDF,
            options=None
        )

        
        # Execute
        result = convert_to_pdf(input_path)

    
    @patch('engines.document.conversion_service.DocumentConversionService')
    def test_batch_convert_to_pdf(self, mock_service_class, temp_dir):
        """Test batch_convert_to_pdf convenience function."""
        from engines.document.conversion_service import batch_convert_to_pdf
        
        # Setup
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        
        input_files = [temp_dir / 'input1.docx', temp_dir / 'input2.docx']
        output_dir = temp_dir / 'output'
        
        expected_jobs = [
            (input_files[0], output_dir / 'input1.pdf', ConversionFormat.PDF, None),
            (input_files[1], output_dir / 'input2.pdf', ConversionFormat.PDF, None),
        ]
        
        mock_service.convert_batch.return_value = []
        
        # Execute
        batch_convert_to_pdf(input_files, output_dir)
        
        # Verify
        mock_service.convert_batch.assert_called_once_with(expected_jobs)