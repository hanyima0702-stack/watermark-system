"""
用户模型
管理系统用户信息和认证数据
"""

from sqlalchemy import Column, String, Boolean, ARRAY, Text
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class User(Base, TimestampMixin):
    """用户模型"""
    
    __tablename__ = 'users'
    
    user_id = Column(String(50), primary_key=True)
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    department = Column(String(100))
    roles = Column(ARRAY(Text), default=list)
    password_hash = Column(String(255))
    salt = Column(String(255))
    is_active = Column(Boolean, default=True, index=True)
    
    # 关系定义
    watermark_tasks = relationship("WatermarkTask", back_populates="user", cascade="all, delete-orphan")
    uploaded_files = relationship("FileMetadata", back_populates="uploader", cascade="all, delete-orphan")
    watermark_configs = relationship("WatermarkConfig", back_populates="creator", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    evidence_reports = relationship("EvidenceReport", back_populates="generator", cascade="all, delete-orphan")
    key_management = relationship("KeyManagement", back_populates="creator", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(user_id='{self.user_id}', username='{self.username}')>"
    
    def has_role(self, role: str) -> bool:
        """检查用户是否具有指定角色"""
        return role in (self.roles or [])
    
    def add_role(self, role: str):
        """添加角色"""
        if not self.roles:
            self.roles = []
        if role not in self.roles:
            self.roles.append(role)
    
    def remove_role(self, role: str):
        """移除角色"""
        if self.roles and role in self.roles:
            self.roles.remove(role)
    
    def is_admin(self) -> bool:
        """检查是否为管理员"""
        return self.has_role('admin')
    
    def is_operator(self) -> bool:
        """检查是否为操作员"""
        return self.has_role('operator') or self.is_admin()
    
    def to_dict(self):
        """转换为字典，排除敏感信息"""
        data = super().to_dict()
        # 移除敏感字段
        data.pop('password_hash', None)
        data.pop('salt', None)
        return data