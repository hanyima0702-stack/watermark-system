"""
Main document processing engine that coordinates all document processors.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .base_processor import DocumentProcessor, WatermarkConfig, DocumentProcessingError
from .office_processor import OfficeProcessor
from .pdf_processor import PDFProcessor
from .conversion_service import DocumentConversionService, ConversionFormat

logger = logging.getLogger(__name__)


class DocumentEngine:
    """Main document processing engine."""
    
    def __init__(self):
        self.processors: Dict[str, DocumentProcessor] = {}
        self.conversion_service = DocumentConversionService()
        
        # Register processors
        self._register_processors()
    
    def _register_processors(self):
        """Register all document processors."""
        # Office processor
        office_processor = OfficeProcessor()
        for format_ext in office_processor.supported_formats:
            self.processors[format_ext] = office_processor
        
        # PDF processor
        pdf_processor = PDFProcessor()
        for format_ext in pdf_processor.supported_formats:
            self.processors[format_ext] = pdf_processor
    
    def get_processor(self, file_path: Path) -> Optional[DocumentProcessor]:
        """Get appropriate processor for file format."""
        file_extension = file_path.suffix.lower()
        return self.processors.get(file_extension)
    
    def is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return self.get_processor(file_path) is not None
    
    def add_visible_watermark(self, file_path: Path, watermark_config: WatermarkConfig,
                            user_id: str, timestamp: datetime = None) -> Path:
        """Add visible watermark to document."""
        if timestamp is None:
            timestamp = datetime.now()
            
        processor = self.get_processor(file_path)
        if not processor:
            raise DocumentProcessingError(f"Unsupported file format: {file_path.suffix}")
        
        return processor.add_visible_watermark(file_path, watermark_config, user_id, timestamp)
    
    def add_invisible_watermark(self, file_path: Path, watermark_data: str) -> Path:
        """Add invisible watermark to document."""
        processor = self.get_processor(file_path)
        if not processor:
            raise DocumentProcessingError(f"Unsupported file format: {file_path.suffix}")
        
        return processor.add_invisible_watermark(file_path, watermark_data)
    
    def add_digital_signature(self, file_path: Path, certificate_path: Path, password: str) -> Path:
        """Add digital signature to document."""
        processor = self.get_processor(file_path)
        if not processor:
            raise DocumentProcessingError(f"Unsupported file format: {file_path.suffix}")
        
        return processor.add_digital_signature(file_path, certificate_path, password)
    
    def extract_invisible_watermark(self, file_path: Path) -> Optional[str]:
        """Extract invisible watermark from document."""
        processor = self.get_processor(file_path)
        if not processor:
            raise DocumentProcessingError(f"Unsupported file format: {file_path.suffix}")
        
        return processor.extract_invisible_watermark(file_path)
    
    def convert_document(self, input_path: Path, output_path: Path, 
                        target_format: ConversionFormat, options: Dict[str, Any] = None):
        """Convert document to different format."""
        return self.conversion_service.convert_single_document(
            input_path, output_path, target_format, options
        )
    
    def process_document_with_watermark(self, file_path: Path, watermark_config: WatermarkConfig,
                                      user_id: str, convert_to_pdf: bool = False) -> Dict[str, Any]:
        """Complete document processing workflow."""
        results = {
            'original_file': str(file_path),
            'processed_files': [],
            'errors': []
        }
        
        try:
            timestamp = datetime.now()
            
            # Add visible watermark if enabled
            if watermark_config.visible_watermark.get('enabled', False):
                try:
                    watermarked_file = self.add_visible_watermark(
                        file_path, watermark_config, user_id, timestamp
                    )
                    results['processed_files'].append({
                        'type': 'visible_watermark',
                        'path': str(watermarked_file)
                    })
                    file_path = watermarked_file  # Use watermarked file for next steps
                except Exception as e:
                    results['errors'].append(f"Visible watermark failed: {str(e)}")
            
            # Add invisible watermark if enabled
            if watermark_config.invisible_watermark.get('enabled', False):
                try:
                    watermark_data = f"{user_id}|{timestamp.isoformat()}"
                    invisible_watermarked_file = self.add_invisible_watermark(file_path, watermark_data)
                    results['processed_files'].append({
                        'type': 'invisible_watermark',
                        'path': str(invisible_watermarked_file)
                    })
                    file_path = invisible_watermarked_file  # Use for next steps
                except Exception as e:
                    results['errors'].append(f"Invisible watermark failed: {str(e)}")
            
            # Add digital signature if configured
            if watermark_config.digital_signature.get('enabled', False):
                try:
                    cert_path = Path(watermark_config.digital_signature.get('certificate_path', ''))
                    password = watermark_config.digital_signature.get('password', '')
                    
                    if cert_path.exists():
                        signed_file = self.add_digital_signature(file_path, cert_path, password)
                        results['processed_files'].append({
                            'type': 'digital_signature',
                            'path': str(signed_file)
                        })
                        file_path = signed_file
                except Exception as e:
                    results['errors'].append(f"Digital signature failed: {str(e)}")
            
            # Convert to PDF if requested
            if convert_to_pdf and file_path.suffix.lower() != '.pdf':
                try:
                    pdf_path = file_path.with_suffix('.pdf')
                    conversion_job = self.convert_document(
                        file_path, pdf_path, ConversionFormat.PDF
                    )
                    
                    if conversion_job.status == 'completed':
                        results['processed_files'].append({
                            'type': 'pdf_conversion',
                            'path': str(pdf_path)
                        })
                    else:
                        results['errors'].append(f"PDF conversion failed: {conversion_job.error_message}")
                        
                except Exception as e:
                    results['errors'].append(f"PDF conversion failed: {str(e)}")
            
            results['final_file'] = str(file_path)
            results['success'] = len(results['errors']) == 0
            
        except Exception as e:
            results['errors'].append(f"Processing failed: {str(e)}")
            results['success'] = False
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.processors.keys())
    
    def get_document_info(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive document information."""
        info = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'file_extension': file_path.suffix.lower(),
            'is_supported': self.is_supported_format(file_path),
            'processor_type': None
        }
        
        processor = self.get_processor(file_path)
        if processor:
            info['processor_type'] = processor.__class__.__name__
            
            # Get format-specific information
            if isinstance(processor, PDFProcessor):
                try:
                    pdf_info = processor.get_pdf_info(file_path)
                    info.update(pdf_info)
                except Exception as e:
                    info['error'] = str(e)
        
        return info