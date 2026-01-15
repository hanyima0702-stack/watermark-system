"""
审计日志模型
管理系统操作审计日志
"""

from sqlalchemy import Column, String, ForeignKey, DateTime, JSON, Text, Index
from sqlalchemy.dialects.postgresql import INET
from sqlalchemy.orm import relationship
from .base import Base
from datetime import datetime
from typing import Dict, Any, Optional


class AuditLog(Base):
    """审计日志模型"""
    
    __tablename__ = 'audit_logs'
    
    log_id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey('users.user_id'), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False, index=True)
    resource_id = Column(String(50), index=True)
    ip_address = Column(INET)
    user_agent = Column(Text)
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    
    # 关系定义
    user = relationship("User", back_populates="audit_logs")
    
    # 复合索引
    __table_args__ = (
        Index('idx_audit_logs_composite', 'user_id', 'action', 'timestamp'),
        Index('idx_audit_logs_resource', 'resource_type', 'resource_id', 'timestamp'),
        Index('idx_audit_logs_time_range', 'timestamp', 'action'),
    )
    
    def __repr__(self):
        return f"<AuditLog(log_id='{self.log_id}', action='{self.action}')>"
    
    def get_detail(self, key: str, default=None):
        """获取详情信息"""
        return (self.details or {}).get(key, default)
    
    def add_detail(self, key: str, value):
        """添加详情信息"""
        if not self.details:
            self.details = {}
        self.details[key] = value
    
    def set_details(self, details: Dict[str, Any]):
        """设置详情信息"""
        self.details = details
    
    def get_request_id(self) -> Optional[str]:
        """获取请求ID"""
        return self.get_detail('request_id')
    
    def get_session_id(self) -> Optional[str]:
        """获取会话ID"""
        return self.get_detail('session_id')
    
    def get_operation_result(self) -> Optional[str]:
        """获取操作结果"""
        return self.get_detail('result')
    
    def get_error_message(self) -> Optional[str]:
        """获取错误信息"""
        return self.get_detail('error_message')
    
    def get_processing_time(self) -> Optional[float]:
        """获取处理时间"""
        return self.get_detail('processing_time')
    
    def get_old_values(self) -> Optional[Dict[str, Any]]:
        """获取修改前的值"""
        return self.get_detail('old_values')
    
    def get_new_values(self) -> Optional[Dict[str, Any]]:
        """获取修改后的值"""
        return self.get_detail('new_values')
    
    def is_successful(self) -> bool:
        """检查操作是否成功"""
        result = self.get_operation_result()
        return result in ('success', 'completed', True) if result is not None else True
    
    def is_failed(self) -> bool:
        """检查操作是否失败"""
        return not self.is_successful()
    
    def is_security_event(self) -> bool:
        """检查是否为安全事件"""
        security_actions = {
            'login_failed', 'unauthorized_access', 'permission_denied',
            'password_changed', 'account_locked', 'suspicious_activity'
        }
        return self.action in security_actions
    
    def is_data_modification(self) -> bool:
        """检查是否为数据修改操作"""
        modification_actions = {'create', 'update', 'delete', 'modify'}
        return any(action in self.action.lower() for action in modification_actions)
    
    def get_log_age_days(self) -> int:
        """获取日志天数"""
        return (datetime.utcnow() - self.timestamp).days
    
    def is_recent(self, hours: int = 24) -> bool:
        """检查是否为最近的日志"""
        age_hours = (datetime.utcnow() - self.timestamp).total_seconds() / 3600
        return age_hours <= hours
    
    def set_request_info(self, request_id: str, session_id: str, ip_address: str, user_agent: str):
        """设置请求信息"""
        self.add_detail('request_id', request_id)
        self.add_detail('session_id', session_id)
        self.ip_address = ip_address
        self.user_agent = user_agent
    
    def set_operation_result(self, result: str, processing_time: Optional[float] = None, 
                           error_message: Optional[str] = None):
        """设置操作结果"""
        self.add_detail('result', result)
        if processing_time is not None:
            self.add_detail('processing_time', processing_time)
        if error_message:
            self.add_detail('error_message', error_message)
    
    def set_data_changes(self, old_values: Dict[str, Any], new_values: Dict[str, Any]):
        """设置数据变更信息"""
        self.add_detail('old_values', old_values)
        self.add_detail('new_values', new_values)
    
    @classmethod
    def create_log(cls, user_id: str, action: str, resource_type: str, 
                   resource_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> 'AuditLog':
        """创建审计日志"""
        return cls(
            log_id=cls.generate_id(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            timestamp=datetime.utcnow()
        )
    
    def to_dict(self):
        """转换为字典，包含关联对象信息"""
        data = super().to_dict()
        
        # 添加关联对象信息
        if self.user:
            data['username'] = self.user.username
            data['department'] = self.user.department
        
        # 添加计算字段
        data['log_age_days'] = self.get_log_age_days()
        data['is_recent'] = self.is_recent()
        data['is_successful'] = self.is_successful()
        data['is_security_event'] = self.is_security_event()
        data['is_data_modification'] = self.is_data_modification()
        
        return data