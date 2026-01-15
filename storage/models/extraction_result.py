"""
水印提取结果模型
管理水印识别和提取的结果数据
"""

from sqlalchemy import Column, String, ForeignKey, DateTime, DECIMAL, JSON, Index
from sqlalchemy.orm import relationship
from .base import Base
from datetime import datetime
from typing import Optional, Dict, Any


class ExtractionResult(Base):
    """水印提取结果模型"""
    
    __tablename__ = 'extraction_results'
    
    result_id = Column(String(50), primary_key=True)
    file_id = Column(String(50), ForeignKey('file_metadata.file_id'), nullable=False, index=True)
    extracted_user_id = Column(String(50), ForeignKey('users.user_id'), index=True)
    extracted_timestamp = Column(DateTime(timezone=True))
    confidence_score = Column(DECIMAL(5, 4), default=0.0000, index=True)
    extraction_method = Column(String(50), nullable=False, index=True)
    extraction_details = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    
    # 关系定义
    file = relationship("FileMetadata", back_populates="extraction_results")
    extracted_user = relationship("User", foreign_keys=[extracted_user_id])
    evidence_reports = relationship("EvidenceReport", back_populates="extraction_result", cascade="all, delete-orphan")
    
    # 复合索引
    __table_args__ = (
        Index('idx_extraction_results_composite', 'file_id', 'confidence_score', 'created_at'),
        Index('idx_extraction_method_confidence', 'extraction_method', 'confidence_score'),
    )
    
    def __repr__(self):
        return f"<ExtractionResult(result_id='{self.result_id}', confidence={self.confidence_score})>"
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """检查是否为高置信度结果"""
        return float(self.confidence_score) >= threshold
    
    def is_medium_confidence(self, low_threshold: float = 0.5, high_threshold: float = 0.8) -> bool:
        """检查是否为中等置信度结果"""
        score = float(self.confidence_score)
        return low_threshold <= score < high_threshold
    
    def is_low_confidence(self, threshold: float = 0.5) -> bool:
        """检查是否为低置信度结果"""
        return float(self.confidence_score) < threshold
    
    def get_confidence_level(self) -> str:
        """获取置信度等级"""
        if self.is_high_confidence():
            return 'high'
        elif self.is_medium_confidence():
            return 'medium'
        else:
            return 'low'
    
    def get_extraction_detail(self, key: str, default=None):
        """获取提取详情"""
        return (self.extraction_details or {}).get(key, default)
    
    def add_extraction_detail(self, key: str, value):
        """添加提取详情"""
        if not self.extraction_details:
            self.extraction_details = {}
        self.extraction_details[key] = value
    
    def set_extraction_details(self, details: Dict[str, Any]):
        """设置提取详情"""
        self.extraction_details = details
    
    def get_watermark_data(self) -> Optional[str]:
        """获取提取的水印数据"""
        return self.get_extraction_detail('watermark_data')
    
    def get_extraction_coordinates(self) -> Optional[Dict[str, Any]]:
        """获取提取位置坐标"""
        return self.get_extraction_detail('coordinates')
    
    def get_algorithm_parameters(self) -> Optional[Dict[str, Any]]:
        """获取算法参数"""
        return self.get_extraction_detail('algorithm_parameters')
    
    def get_quality_assessment(self) -> Optional[Dict[str, float]]:
        """获取质量评估结果"""
        return self.get_extraction_detail('quality_assessment')
    
    def is_valid_extraction(self) -> bool:
        """检查是否为有效提取"""
        return (
            self.extracted_user_id is not None and
            self.confidence_score > 0 and
            self.get_watermark_data() is not None
        )
    
    def matches_user(self, user_id: str) -> bool:
        """检查是否匹配指定用户"""
        return self.extracted_user_id == user_id
    
    def get_extraction_age_days(self) -> int:
        """获取提取结果的天数"""
        if self.extracted_timestamp:
            return (datetime.utcnow() - self.extracted_timestamp).days
        return (datetime.utcnow() - self.created_at).days
    
    def to_dict(self):
        """转换为字典，包含关联对象信息"""
        data = super().to_dict()
        
        # 添加关联对象信息
        if self.file:
            data['file_name'] = self.file.original_name
            data['file_type'] = self.file.file_type
        
        if self.extracted_user:
            data['extracted_username'] = self.extracted_user.username
            data['extracted_department'] = self.extracted_user.department
        
        # 添加计算字段
        data['confidence_level'] = self.get_confidence_level()
        data['is_valid'] = self.is_valid_extraction()
        data['extraction_age_days'] = self.get_extraction_age_days()
        
        return data