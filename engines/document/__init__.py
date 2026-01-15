"""
Document processing engine for watermark embedding and format conversion.
"""

from .document_engine import DocumentEngine
from .base_processor import DocumentProcessor, WatermarkConfig, DocumentProcessingError
from .office_processor import OfficeProcessor
from .pdf_processor import PDFProcessor
from .conversion_service import DocumentConversionService, ConversionFormat, ConversionJob

__all__ = [
    'DocumentEngine',
    'DocumentProcessor',
    'WatermarkConfig',
    'DocumentProcessingError',
    'OfficeProcessor',
    'PDFProcessor',
    'DocumentConversionService',
    'ConversionFormat',
    'ConversionJob'
]