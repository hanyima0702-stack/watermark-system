"""
Base document processor interface and common functionality.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class WatermarkConfig:
    """Configuration for watermark operations."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        
    @property
    def visible_watermark(self) -> Dict[str, Any]:
        return self.config.get('visible_watermark', {})
    
    @property
    def invisible_watermark(self) -> Dict[str, Any]:
        return self.config.get('invisible_watermark', {})
    
    @property
    def digital_signature(self) -> Dict[str, Any]:
        return self.config.get('digital_signature', {})


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    def __init__(self):
        self.supported_formats: List[str] = []
        
    @abstractmethod
    def add_visible_watermark(self, file_path: Path, watermark_config: WatermarkConfig, 
                            user_id: str, timestamp: datetime) -> Path:
        """Add visible watermark to document."""
        pass
    
    @abstractmethod
    def add_invisible_watermark(self, file_path: Path, watermark_data: str) -> Path:
        """Add invisible watermark to document."""
        pass
    
    @abstractmethod
    def add_digital_signature(self, file_path: Path, certificate_path: Path, 
                            password: str) -> Path:
        """Add digital signature to document."""
        pass
    
    @abstractmethod
    def extract_invisible_watermark(self, file_path: Path) -> Optional[str]:
        """Extract invisible watermark from document."""
        pass
    
    def is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in self.supported_formats
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def generate_output_path(self, input_path: Path, suffix: str = "_watermarked") -> Path:
        """Generate output file path."""
        return input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass


class UnsupportedFormatError(DocumentProcessingError):
    """Raised when document format is not supported."""
    pass


class WatermarkEmbeddingError(DocumentProcessingError):
    """Raised when watermark embedding fails."""
    pass


class DigitalSignatureError(DocumentProcessingError):
    """Raised when digital signature operation fails."""
    pass