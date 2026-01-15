"""
密钥管理模型
管理系统加密密钥和数字证书
"""

from sqlalchemy import Column, String, Text, Integer, Boolean, ForeignKey, DateTime, CheckConstraint, Index
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin
from datetime import datetime, timedelta
from typing import Optional


class KeyManagement(Base, TimestampMixin):
    """密钥管理模型"""
    
    __tablename__ = 'key_management'
    
    key_id = Column(String(50), primary_key=True)
    key_type = Column(String(50), nullable=False, index=True)
    key_data = Column(Text, nullable=False)  # 加密存储
    key_version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True, index=True)
    expires_at = Column(DateTime(timezone=True), index=True)
    created_by = Column(String(50), ForeignKey('users.user_id'), nullable=False)
    rotated_at = Column(DateTime(timezone=True))
    
    # 关系定义
    creator = relationship("User", back_populates="key_management")
    
    # 约束条件
    __table_args__ = (
        CheckConstraint(
            "key_type IN ('encryption', 'signing', 'watermark')",
            name='check_key_type'
        ),
        Index('idx_key_management_composite', 'key_type', 'is_active', 'expires_at'),
    )
    
    def __repr__(self):
        return f"<KeyManagement(key_id='{self.key_id}', type='{self.key_type}')>"
    
    def is_encryption_key(self) -> bool:
        """检查是否为加密密钥"""
        return self.key_type == 'encryption'
    
    def is_signing_key(self) -> bool:
        """检查是否为签名密钥"""
        return self.key_type == 'signing'
    
    def is_watermark_key(self) -> bool:
        """检查是否为水印密钥"""
        return self.key_type == 'watermark'
    
    def is_expired(self) -> bool:
        """检查密钥是否已过期"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_expiring_soon(self, days: int = 30) -> bool:
        """检查密钥是否即将过期"""
        if not self.expires_at:
            return False
        warning_date = datetime.utcnow() + timedelta(days=days)
        return self.expires_at <= warning_date
    
    def get_remaining_days(self) -> Optional[int]:
        """获取剩余有效天数"""
        if not self.expires_at:
            return None
        remaining = self.expires_at - datetime.utcnow()
        return max(0, remaining.days)
    
    def get_age_days(self) -> int:
        """获取密钥年龄（天数）"""
        return (datetime.utcnow() - self.created_at).days
    
    def is_current_version(self) -> bool:
        """检查是否为当前版本"""
        return self.is_active and not self.is_expired()
    
    def needs_rotation(self, max_age_days: int = 365) -> bool:
        """检查是否需要轮换"""
        return (
            self.get_age_days() >= max_age_days or
            self.is_expired() or
            self.is_expiring_soon()
        )
    
    def deactivate(self):
        """停用密钥"""
        self.is_active = False
        self.rotated_at = datetime.utcnow()
    
    def rotate(self, new_key_data: str, new_expires_at: Optional[datetime] = None) -> 'KeyManagement':
        """轮换密钥，返回新密钥对象"""
        # 停用当前密钥
        self.deactivate()
        
        # 创建新密钥
        new_key = KeyManagement(
            key_id=self.generate_id(),
            key_type=self.key_type,
            key_data=new_key_data,
            key_version=self.key_version + 1,
            is_active=True,
            expires_at=new_expires_at or (datetime.utcnow() + timedelta(days=365)),
            created_by=self.created_by
        )
        
        return new_key
    
    def set_expiration(self, expires_at: datetime):
        """设置过期时间"""
        self.expires_at = expires_at
    
    def extend_expiration(self, days: int):
        """延长过期时间"""
        if self.expires_at:
            self.expires_at += timedelta(days=days)
        else:
            self.expires_at = datetime.utcnow() + timedelta(days=days)
    
    @classmethod
    def create_key(cls, key_type: str, key_data: str, created_by: str,
                   expires_in_days: int = 365) -> 'KeyManagement':
        """创建新密钥"""
        return cls(
            key_id=cls.generate_id(),
            key_type=key_type,
            key_data=key_data,
            key_version=1,
            is_active=True,
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
            created_by=created_by
        )
    
    def get_status(self) -> str:
        """获取密钥状态"""
        if not self.is_active:
            return 'inactive'
        elif self.is_expired():
            return 'expired'
        elif self.is_expiring_soon():
            return 'expiring'
        else:
            return 'active'
    
    def to_dict(self):
        """转换为字典，排除敏感信息"""
        data = super().to_dict()
        
        # 移除敏感的密钥数据
        data.pop('key_data', None)
        
        # 添加关联对象信息
        if self.creator:
            data['created_by_username'] = self.creator.username
        
        # 添加计算字段
        data['status'] = self.get_status()
        data['age_days'] = self.get_age_days()
        data['remaining_days'] = self.get_remaining_days()
        data['needs_rotation'] = self.needs_rotation()
        
        return data