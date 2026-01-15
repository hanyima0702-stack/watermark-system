"""
文件元数据模型
管理上传文件的元数据信息
"""

from sqlalchemy import Column, String, BigInteger, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from .base import Base
from datetime import datetime


class FileMetadata(Base):
    """文件元数据模型"""
    
    __tablename__ = 'file_metadata'
    
    file_id = Column(String(50), primary_key=True)
    original_name = Column(String(255), nullable=False)
    file_type = Column(String(100), nullable=False, index=True)
    file_hash = Column(String(64), nullable=False, index=True)
    file_size = Column(BigInteger, nullable=False)
    storage_path = Column(String(500), nullable=False)
    uploaded_by = Column(String(50), ForeignKey('users.user_id'), nullable=False, index=True)
    metadata = Column(JSON, default=dict)
    uploaded_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    
    # 关系定义
    uploader = relationship("User", back_populates="uploaded_files")
    watermark_tasks = relationship("WatermarkTask", back_populates="file", cascade="all, delete-orphan")
    extraction_results = relationship("ExtractionResult", back_populates="file", cascade="all, delete-orphan")
    
    # 复合索引
    __table_args__ = (
        Index('idx_file_metadata_composite', 'uploaded_by', 'file_type', 'uploaded_at'),
        Index('idx_file_hash_unique', 'file_hash', 'uploaded_by'),
    )
    
    def __repr__(self):
        return f"<FileMetadata(file_id='{self.file_id}', name='{self.original_name}')>"
    
    @property
    def file_extension(self) -> str:
        """获取文件扩展名"""
        return self.original_name.split('.')[-1].lower() if '.' in self.original_name else ''
    
    @property
    def size_mb(self) -> float:
        """获取文件大小（MB）"""
        return round(self.file_size / (1024 * 1024), 2)
    
    def is_document(self) -> bool:
        """判断是否为文档类型"""
        doc_extensions = {'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
        return self.file_extension in doc_extensions
    
    def is_image(self) -> bool:
        """判断是否为图像类型"""
        image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'webp'}
        return self.file_extension in image_extensions
    
    def is_video(self) -> bool:
        """判断是否为视频类型"""
        video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
        return self.file_extension in video_extensions
    
    def is_audio(self) -> bool:
        """判断是否为音频类型"""
        audio_extensions = {'mp3', 'wav', 'flac', 'aac', 'ogg', 'wma', 'm4a'}
        return self.file_extension in audio_extensions
    
    def get_media_type(self) -> str:
        """获取媒体类型"""
        if self.is_document():
            return 'document'
        elif self.is_image():
            return 'image'
        elif self.is_video():
            return 'video'
        elif self.is_audio():
            return 'audio'
        else:
            return 'unknown'
    
    def add_metadata(self, key: str, value):
        """添加元数据"""
        if not self.metadata:
            self.metadata = {}
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default=None):
        """获取元数据"""
        return (self.metadata or {}).get(key, default)