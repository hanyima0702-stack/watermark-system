"""
系统配置数据访问对象
"""

from typing import List, Optional, Any
from sqlalchemy.orm import selectinload

from .base_dao import BaseDAO
from ..models.system_config import SystemConfig


class SystemConfigDAO(BaseDAO[SystemConfig]):
    """系统配置数据访问对象"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager, SystemConfig)
    
    def get_model_class(self):
        return SystemConfig
    
    def _add_relationship_loading(self, query):
        """加载配置关联对象"""
        return query.options(
            selectinload(SystemConfig.updater)
        )
    
    async def get_config_value(self, config_key: str, default: Any = None) -> Any:
        """获取配置值"""
        config = await self.get_by_id(config_key)
        return config.get_value() if config else default
    
    async def set_config_value(self, config_key: str, value: Any, 
                             description: str = "", updated_by: Optional[str] = None) -> bool:
        """设置配置值"""
        config = await self.get_by_id(config_key)
        
        if config:
            # 更新现有配置
            result = await self.update(config_key, {
                'config_value': value,
                'updated_by': updated_by
            })
            return result is not None
        else:
            # 创建新配置
            new_config = SystemConfig.create_config(
                key=config_key,
                value=value,
                description=description,
                updated_by=updated_by
            )
            result = await self.create(new_config)
            return result is not None
    
    async def get_string_config(self, config_key: str, default: str = "") -> str:
        """获取字符串配置"""
        config = await self.get_by_id(config_key)
        return config.get_string_value(default) if config else default
    
    async def get_int_config(self, config_key: str, default: int = 0) -> int:
        """获取整数配置"""
        config = await self.get_by_id(config_key)
        return config.get_int_value(default) if config else default
    
    async def get_float_config(self, config_key: str, default: float = 0.0) -> float:
        """获取浮点数配置"""
        config = await self.get_by_id(config_key)
        return config.get_float_value(default) if config else default
    
    async def get_bool_config(self, config_key: str, default: bool = False) -> bool:
        """获取布尔配置"""
        config = await self.get_by_id(config_key)
        return config.get_bool_value(default) if config else default
    
    async def get_list_config(self, config_key: str, default: list = None) -> list:
        """获取列表配置"""
        config = await self.get_by_id(config_key)
        return config.get_list_value(default or []) if config else (default or [])
    
    async def get_dict_config(self, config_key: str, default: dict = None) -> dict:
        """获取字典配置"""
        config = await self.get_by_id(config_key)
        return config.get_dict_value(default or {}) if config else (default or {})
    
    async def get_system_configs(self) -> List[SystemConfig]:
        """获取系统级配置"""
        async with self.get_session() as session:
            query = select(SystemConfig).where(
                or_(
                    SystemConfig.config_key.startswith('system_'),
                    SystemConfig.config_key.startswith('app_'),
                    SystemConfig.config_key.startswith('db_'),
                    SystemConfig.config_key.startswith('cache_'),
                    SystemConfig.config_key.startswith('queue_')
                )
            ).order_by(SystemConfig.config_key)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_security_configs(self) -> List[SystemConfig]:
        """获取安全相关配置"""
        async with self.get_session() as session:
            query = select(SystemConfig).where(
                or_(
                    SystemConfig.config_key.ilike('%password%'),
                    SystemConfig.config_key.ilike('%secret%'),
                    SystemConfig.config_key.ilike('%key%'),
                    SystemConfig.config_key.ilike('%token%'),
                    SystemConfig.config_key.ilike('%auth%'),
                    SystemConfig.is_encrypted == True
                )
            ).order_by(SystemConfig.config_key)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def search_configs(self, keyword: str, limit: int = 100, offset: int = 0) -> List[SystemConfig]:
        """搜索配置"""
        async with self.get_session() as session:
            query = select(SystemConfig).where(
                or_(
                    SystemConfig.config_key.ilike(f"%{keyword}%"),
                    SystemConfig.description.ilike(f"%{keyword}%")
                )
            ).order_by(SystemConfig.config_key).limit(limit).offset(offset)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def backup_configs(self) -> dict:
        """备份所有配置"""
        configs = await self.get_all()
        return {
            config.config_key: {
                'value': config.config_value,
                'description': config.description,
                'is_encrypted': config.is_encrypted
            }
            for config in configs
        }
    
    async def restore_configs(self, config_data: dict, updated_by: Optional[str] = None) -> int:
        """恢复配置"""
        restored_count = 0
        
        for key, data in config_data.items():
            success = await self.set_config_value(
                config_key=key,
                value=data['value'],
                description=data.get('description', ''),
                updated_by=updated_by
            )
            if success:
                restored_count += 1
        
        return restored_count