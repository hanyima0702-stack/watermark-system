"""
水印配置模型
管理水印配置模板和参数
"""

from sqlalchemy import Column, String, Boolean, JSON, ForeignKey, DateTime, Index, CheckConstraint
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin
from typing import Dict, Any, Optional


class WatermarkConfig(Base, TimestampMixin):
    """水印配置模型"""
    
    __tablename__ = 'watermark_configs'
    
    config_id = Column(String(50), primary_key=True)
    config_name = Column(String(100), nullable=False, index=True)
    watermark_type = Column(String(20), nullable=False, index=True)
    visible_config = Column(JSON, default=dict)
    invisible_config = Column(JSON, default=dict)
    template_variables = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, index=True)
    created_by = Column(String(50), ForeignKey('users.user_id'), nullable=False, index=True)
    
    # 关系定义
    creator = relationship("User", back_populates="watermark_configs")
    watermark_tasks = relationship("WatermarkTask", back_populates="config")
    
    # 约束条件
    __table_args__ = (
        CheckConstraint(
            "watermark_type IN ('visible', 'invisible', 'both')",
            name='check_watermark_type'
        ),
        Index('idx_config_name_user', 'config_name', 'created_by'),
    )
    
    def __repr__(self):
        return f"<WatermarkConfig(config_id='{self.config_id}', name='{self.config_name}')>"
    
    def is_visible_enabled(self) -> bool:
        """检查是否启用明水印"""
        return self.watermark_type in ('visible', 'both')
    
    def is_invisible_enabled(self) -> bool:
        """检查是否启用暗水印"""
        return self.watermark_type in ('invisible', 'both')
    
    def get_visible_config(self) -> Dict[str, Any]:
        """获取明水印配置"""
        return self.visible_config or {}
    
    def get_invisible_config(self) -> Dict[str, Any]:
        """获取暗水印配置"""
        return self.invisible_config or {}
    
    def get_template_variables(self) -> Dict[str, str]:
        """获取模板变量"""
        return self.template_variables or {}
    
    def set_visible_config(self, config: Dict[str, Any]):
        """设置明水印配置"""
        self.visible_config = config
        if self.watermark_type == 'invisible':
            self.watermark_type = 'both'
        elif self.watermark_type != 'both':
            self.watermark_type = 'visible'
    
    def set_invisible_config(self, config: Dict[str, Any]):
        """设置暗水印配置"""
        self.invisible_config = config
        if self.watermark_type == 'visible':
            self.watermark_type = 'both'
        elif self.watermark_type != 'both':
            self.watermark_type = 'invisible'
    
    def add_template_variable(self, key: str, value: str):
        """添加模板变量"""
        if not self.template_variables:
            self.template_variables = {}
        self.template_variables[key] = value
    
    def remove_template_variable(self, key: str):
        """移除模板变量"""
        if self.template_variables and key in self.template_variables:
            del self.template_variables[key]
    
    def validate_config(self) -> tuple[bool, Optional[str]]:
        """验证配置有效性"""
        if self.is_visible_enabled():
            visible_config = self.get_visible_config()
            if not visible_config.get('enabled', False):
                return False, "明水印已启用但配置无效"
        
        if self.is_invisible_enabled():
            invisible_config = self.get_invisible_config()
            if not invisible_config.get('enabled', False):
                return False, "暗水印已启用但配置无效"
        
        return True, None
    
    def clone(self, new_name: str, created_by: str) -> 'WatermarkConfig':
        """克隆配置"""
        new_config = WatermarkConfig(
            config_id=self.generate_id(),
            config_name=new_name,
            watermark_type=self.watermark_type,
            visible_config=self.visible_config.copy() if self.visible_config else {},
            invisible_config=self.invisible_config.copy() if self.invisible_config else {},
            template_variables=self.template_variables.copy() if self.template_variables else {},
            is_active=True,
            created_by=created_by
        )
        return new_config