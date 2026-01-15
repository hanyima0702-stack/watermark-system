"""
证据报告模型
管理水印溯源证据报告
"""

from sqlalchemy import Column, String, ForeignKey, DateTime, JSON, Index
from sqlalchemy.orm import relationship
from .base import Base
from datetime import datetime
from typing import Dict, Any, Optional


class EvidenceReport(Base):
    """证据报告模型"""
    
    __tablename__ = 'evidence_reports'
    
    report_id = Column(String(50), primary_key=True)
    result_id = Column(String(50), ForeignKey('extraction_results.result_id'), nullable=False, index=True)
    report_path = Column(String(500), nullable=False)
    report_metadata = Column(JSON, default=dict)
    generated_by = Column(String(50), ForeignKey('users.user_id'), nullable=False, index=True)
    generated_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    
    # 关系定义
    extraction_result = relationship("ExtractionResult", back_populates="evidence_reports")
    generator = relationship("User", back_populates="evidence_reports")
    
    # 索引
    __table_args__ = (
        Index('idx_evidence_reports_composite', 'result_id', 'generated_by', 'generated_at'),
    )
    
    def __repr__(self):
        return f"<EvidenceReport(report_id='{self.report_id}', path='{self.report_path}')>"
    
    def get_report_metadata(self, key: str, default=None):
        """获取报告元数据"""
        return (self.report_metadata or {}).get(key, default)
    
    def add_report_metadata(self, key: str, value):
        """添加报告元数据"""
        if not self.report_metadata:
            self.report_metadata = {}
        self.report_metadata[key] = value
    
    def set_report_metadata(self, metadata: Dict[str, Any]):
        """设置报告元数据"""
        self.report_metadata = metadata
    
    def get_report_type(self) -> str:
        """获取报告类型"""
        return self.get_report_metadata('report_type', 'standard')
    
    def get_report_format(self) -> str:
        """获取报告格式"""
        return self.get_report_metadata('format', 'pdf')
    
    def get_report_language(self) -> str:
        """获取报告语言"""
        return self.get_report_metadata('language', 'zh-CN')
    
    def get_report_template(self) -> str:
        """获取报告模板"""
        return self.get_report_metadata('template', 'default')
    
    def get_file_size(self) -> Optional[int]:
        """获取报告文件大小"""
        return self.get_report_metadata('file_size')
    
    def get_page_count(self) -> Optional[int]:
        """获取报告页数"""
        return self.get_report_metadata('page_count')
    
    def get_generation_time(self) -> Optional[float]:
        """获取生成耗时（秒）"""
        return self.get_report_metadata('generation_time')
    
    def is_signed(self) -> bool:
        """检查报告是否已签名"""
        return self.get_report_metadata('is_signed', False)
    
    def get_signature_info(self) -> Optional[Dict[str, Any]]:
        """获取签名信息"""
        return self.get_report_metadata('signature_info')
    
    def get_hash_value(self) -> Optional[str]:
        """获取报告文件哈希值"""
        return self.get_report_metadata('hash_value')
    
    def set_file_info(self, file_size: int, page_count: int, hash_value: str):
        """设置文件信息"""
        self.add_report_metadata('file_size', file_size)
        self.add_report_metadata('page_count', page_count)
        self.add_report_metadata('hash_value', hash_value)
    
    def set_generation_info(self, generation_time: float, template: str):
        """设置生成信息"""
        self.add_report_metadata('generation_time', generation_time)
        self.add_report_metadata('template', template)
    
    def set_signature_info(self, signature_info: Dict[str, Any]):
        """设置签名信息"""
        self.add_report_metadata('is_signed', True)
        self.add_report_metadata('signature_info', signature_info)
    
    def get_report_age_days(self) -> int:
        """获取报告生成天数"""
        return (datetime.utcnow() - self.generated_at).days
    
    def is_recent(self, days: int = 7) -> bool:
        """检查是否为最近生成的报告"""
        return self.get_report_age_days() <= days
    
    def to_dict(self):
        """转换为字典，包含关联对象信息"""
        data = super().to_dict()
        
        # 添加关联对象信息
        if self.extraction_result:
            data['confidence_score'] = float(self.extraction_result.confidence_score)
            data['extraction_method'] = self.extraction_result.extraction_method
            if self.extraction_result.file:
                data['source_file_name'] = self.extraction_result.file.original_name
        
        if self.generator:
            data['generator_username'] = self.generator.username
            data['generator_department'] = self.generator.department
        
        # 添加计算字段
        data['report_age_days'] = self.get_report_age_days()
        data['is_recent'] = self.is_recent()
        data['report_type'] = self.get_report_type()
        data['report_format'] = self.get_report_format()
        
        return data