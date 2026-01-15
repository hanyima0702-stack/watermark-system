"""
水印任务模型
管理水印处理任务的状态和进度
"""

from sqlalchemy import Column, String, ForeignKey, DateTime, Text, DECIMAL, JSON, Index, CheckConstraint
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin
from datetime import datetime
from typing import Optional, Dict, Any


class WatermarkTask(Base, TimestampMixin):
    """水印任务模型"""
    
    __tablename__ = 'watermark_tasks'
    
    task_id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey('users.user_id'), nullable=False, index=True)
    file_id = Column(String(50), ForeignKey('file_metadata.file_id'), nullable=False, index=True)
    config_id = Column(String(50), ForeignKey('watermark_configs.config_id'), nullable=False)
    task_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default='pending', index=True)
    progress = Column(DECIMAL(5, 2), default=0.00)
    error_message = Column(Text)
    processing_engine = Column(String(50))
    output_file_id = Column(String(50), ForeignKey('file_metadata.file_id'))
    processing_time = Column(DECIMAL(10, 3))
    quality_metrics = Column(JSON, default=dict)
    completed_at = Column(DateTime(timezone=True))
    
    # 关系定义
    user = relationship("User", back_populates="watermark_tasks")
    file = relationship("FileMetadata", back_populates="watermark_tasks", foreign_keys=[file_id])
    output_file = relationship("FileMetadata", foreign_keys=[output_file_id])
    config = relationship("WatermarkConfig", back_populates="watermark_tasks")
    
    # 约束条件
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')",
            name='check_task_status'
        ),
        CheckConstraint(
            "progress >= 0.00 AND progress <= 100.00",
            name='check_progress_range'
        ),
        Index('idx_watermark_tasks_composite', 'user_id', 'status', 'created_at'),
        Index('idx_task_type_status', 'task_type', 'status'),
    )
    
    def __repr__(self):
        return f"<WatermarkTask(task_id='{self.task_id}', status='{self.status}')>"
    
    def is_pending(self) -> bool:
        """检查任务是否待处理"""
        return self.status == 'pending'
    
    def is_processing(self) -> bool:
        """检查任务是否正在处理"""
        return self.status == 'processing'
    
    def is_completed(self) -> bool:
        """检查任务是否已完成"""
        return self.status == 'completed'
    
    def is_failed(self) -> bool:
        """检查任务是否失败"""
        return self.status == 'failed'
    
    def is_cancelled(self) -> bool:
        """检查任务是否已取消"""
        return self.status == 'cancelled'
    
    def is_finished(self) -> bool:
        """检查任务是否已结束（完成、失败或取消）"""
        return self.status in ('completed', 'failed', 'cancelled')
    
    def start_processing(self, engine: str):
        """开始处理任务"""
        self.status = 'processing'
        self.processing_engine = engine
        self.progress = 0.0
        self.error_message = None
    
    def update_progress(self, progress: float):
        """更新任务进度"""
        if 0 <= progress <= 100:
            self.progress = progress
    
    def complete_task(self, output_file_id: Optional[str] = None, 
                     processing_time: Optional[float] = None,
                     quality_metrics: Optional[Dict[str, Any]] = None):
        """完成任务"""
        self.status = 'completed'
        self.progress = 100.0
        self.completed_at = datetime.utcnow()
        if output_file_id:
            self.output_file_id = output_file_id
        if processing_time:
            self.processing_time = processing_time
        if quality_metrics:
            self.quality_metrics = quality_metrics
    
    def fail_task(self, error_message: str):
        """任务失败"""
        self.status = 'failed'
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
    
    def cancel_task(self, reason: str = "用户取消"):
        """取消任务"""
        if not self.is_finished():
            self.status = 'cancelled'
            self.error_message = reason
            self.completed_at = datetime.utcnow()
    
    def get_duration(self) -> Optional[float]:
        """获取任务持续时间（秒）"""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None
    
    def get_quality_metric(self, metric_name: str, default=None):
        """获取质量指标"""
        return (self.quality_metrics or {}).get(metric_name, default)
    
    def add_quality_metric(self, metric_name: str, value):
        """添加质量指标"""
        if not self.quality_metrics:
            self.quality_metrics = {}
        self.quality_metrics[metric_name] = value
    
    def to_dict(self):
        """转换为字典，包含关联对象信息"""
        data = super().to_dict()
        
        # 添加关联对象信息
        if self.file:
            data['file_name'] = self.file.original_name
            data['file_type'] = self.file.file_type
        
        if self.config:
            data['config_name'] = self.config.config_name
            data['watermark_type'] = self.config.watermark_type
        
        if self.user:
            data['username'] = self.user.username
            data['department'] = self.user.department
        
        # 计算持续时间
        data['duration'] = self.get_duration()
        
        return data