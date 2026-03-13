"""
用户会话模型
管理用户登录会话信息
"""

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class UserSession(Base, TimestampMixin):
    """用户会话模型"""

    __tablename__ = 'user_sessions'

    id = Column(String(36), primary_key=True)
    user_id = Column(String(50), ForeignKey('users.user_id'), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    last_activity_at = Column(DateTime(timezone=True), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True, index=True)

    # Relationship
    user = relationship("User", backref="sessions")

    def __repr__(self):
        return f"<UserSession(id='{self.id}', user_id='{self.user_id}', is_active={self.is_active})>"
