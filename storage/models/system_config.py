"""
系统配置模型
管理系统全局配置参数
"""

from sqlalchemy import Column, String, JSON, Text, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin
from typing import Any, Dict, Optional, Union
import json


class SystemConfig(Base, TimestampMixin):
    """系统配置模型"""
    
    __tablename__ = 'system_configs'
    
    config_key = Column(String(100), primary_key=True)
    config_value = Column(JSON, nullable=False)
    description = Column(Text)
    is_encrypted = Column(Boolean, default=False)
    updated_by = Column(String(50), ForeignKey('users.user_id'))
    
    # 关系定义
    updater = relationship("User", foreign_keys=[updated_by])
    
    def __repr__(self):
        return f"<SystemConfig(key='{self.config_key}', encrypted={self.is_encrypted})>"
    
    def get_value(self) -> Any:
        """获取配置值"""
        return self.config_value
    
    def set_value(self, value: Any):
        """设置配置值"""
        self.config_value = value
    
    def get_string_value(self, default: str = "") -> str:
        """获取字符串值"""
        if isinstance(self.config_value, str):
            return self.config_value
        return str(self.config_value) if self.config_value is not None else default
    
    def get_int_value(self, default: int = 0) -> int:
        """获取整数值"""
        try:
            if isinstance(self.config_value, int):
                return self.config_value
            return int(self.config_value)
        except (ValueError, TypeError):
            return default
    
    def get_float_value(self, default: float = 0.0) -> float:
        """获取浮点数值"""
        try:
            if isinstance(self.config_value, (int, float)):
                return float(self.config_value)
            return float(self.config_value)
        except (ValueError, TypeError):
            return default
    
    def get_bool_value(self, default: bool = False) -> bool:
        """获取布尔值"""
        if isinstance(self.config_value, bool):
            return self.config_value
        if isinstance(self.config_value, str):
            return self.config_value.lower() in ('true', '1', 'yes', 'on')
        return bool(self.config_value) if self.config_value is not None else default
    
    def get_list_value(self, default: list = None) -> list:
        """获取列表值"""
        if isinstance(self.config_value, list):
            return self.config_value
        if isinstance(self.config_value, str):
            try:
                return json.loads(self.config_value)
            except json.JSONDecodeError:
                return self.config_value.split(',') if self.config_value else (default or [])
        return default or []
    
    def get_dict_value(self, default: dict = None) -> dict:
        """获取字典值"""
        if isinstance(self.config_value, dict):
            return self.config_value
        if isinstance(self.config_value, str):
            try:
                return json.loads(self.config_value)
            except json.JSONDecodeError:
                return default or {}
        return default or {}
    
    def is_system_config(self) -> bool:
        """检查是否为系统级配置"""
        system_prefixes = ['system_', 'app_', 'db_', 'cache_', 'queue_']
        return any(self.config_key.startswith(prefix) for prefix in system_prefixes)
    
    def is_security_config(self) -> bool:
        """检查是否为安全配置"""
        security_keywords = ['password', 'secret', 'key', 'token', 'auth', 'ssl', 'tls']
        return any(keyword in self.config_key.lower() for keyword in security_keywords)
    
    def should_encrypt(self) -> bool:
        """检查是否应该加密"""
        return self.is_security_config() or self.is_encrypted
    
    @classmethod
    def create_config(cls, key: str, value: Any, description: str = "", 
                     is_encrypted: bool = False, updated_by: Optional[str] = None) -> 'SystemConfig':
        """创建系统配置"""
        return cls(
            config_key=key,
            config_value=value,
            description=description,
            is_encrypted=is_encrypted,
            updated_by=updated_by
        )
    
    @classmethod
    def get_default_configs(cls) -> Dict[str, Dict[str, Any]]:
        """获取默认配置"""
        return {
            'max_file_size_mb': {
                'value': 500,
                'description': '最大文件大小限制(MB)',
                'encrypted': False
            },
            'max_batch_files': {
                'value': 100,
                'description': '批量处理最大文件数',
                'encrypted': False
            },
            'task_timeout_seconds': {
                'value': 3600,
                'description': '任务超时时间(秒)',
                'encrypted': False
            },
            'cache_expire_seconds': {
                'value': 3600,
                'description': '缓存过期时间(秒)',
                'encrypted': False
            },
            'supported_document_formats': {
                'value': ['docx', 'xlsx', 'pptx', 'pdf'],
                'description': '支持的文档格式',
                'encrypted': False
            },
            'supported_image_formats': {
                'value': ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                'description': '支持的图像格式',
                'encrypted': False
            },
            'supported_video_formats': {
                'value': ['mp4', 'avi', 'mov', 'mkv', 'wmv'],
                'description': '支持的视频格式',
                'encrypted': False
            },
            'supported_audio_formats': {
                'value': ['mp3', 'wav', 'flac', 'aac', 'ogg'],
                'description': '支持的音频格式',
                'encrypted': False
            },
            'watermark_quality_threshold': {
                'value': 0.8,
                'description': '水印质量阈值',
                'encrypted': False
            },
            'extraction_confidence_threshold': {
                'value': 0.7,
                'description': '提取置信度阈值',
                'encrypted': False
            },
            'enable_gpu_acceleration': {
                'value': True,
                'description': '启用GPU加速',
                'encrypted': False
            },
            'max_concurrent_tasks': {
                'value': 10,
                'description': '最大并发任务数',
                'encrypted': False
            }
        }
    
    def to_dict(self):
        """转换为字典，排除敏感信息"""
        data = super().to_dict()
        
        # 如果是加密配置，不返回实际值
        if self.is_encrypted:
            data['config_value'] = '***encrypted***'
        
        # 添加关联对象信息
        if self.updater:
            data['updated_by_username'] = self.updater.username
        
        return data