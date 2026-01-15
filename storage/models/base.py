"""
SQLAlchemy基础模型
定义所有模型的基类和通用字段
"""

from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declared_attr
from datetime import datetime
import uuid


class BaseModel:
    """基础模型类，包含通用字段和方法"""
    
    @declared_attr
    def __tablename__(cls):
        # 自动生成表名（类名转下划线格式）
        import re
        return re.sub('([A-Z])', r'_\1', cls.__name__).lower().lstrip('_')
    
    def to_dict(self):
        """转换为字典"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update_from_dict(self, data: dict):
        """从字典更新属性"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def generate_id(cls) -> str:
        """生成唯一ID"""
        return str(uuid.uuid4())


class TimestampMixin:
    """时间戳混入类"""
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)


# 创建基础模型类
Base = declarative_base(cls=BaseModel)