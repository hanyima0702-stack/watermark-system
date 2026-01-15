"""
水印配置数据访问对象
"""

from typing import List, Optional
from sqlalchemy.orm import selectinload

from .base_dao import BaseDAO
from ..models.watermark_config import WatermarkConfig


class WatermarkConfigDAO(BaseDAO[WatermarkConfig]):
    """水印配置数据访问对象"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager, WatermarkConfig)
    
    def get_model_class(self):
        return WatermarkConfig
    
    def _add_relationship_loading(self, query):
        """加载配置关联对象"""
        return query.options(
            selectinload(WatermarkConfig.creator),
            selectinload(WatermarkConfig.watermark_tasks)
        )
    
    async def get_user_configs(self, user_id: str, is_active: Optional[bool] = None,
                             limit: int = 100, offset: int = 0) -> List[WatermarkConfig]:
        """获取用户配置列表"""
        filters = {'created_by': user_id}
        if is_active is not None:
            filters['is_active'] = is_active
        
        return await self.find_by_filters(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_active_configs(self, user_id: Optional[str] = None,
                               limit: int = 100, offset: int = 0) -> List[WatermarkConfig]:
        """获取活跃配置"""
        filters = {'is_active': True}
        if user_id:
            filters['created_by'] = user_id
        
        return await self.find_by_filters(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by='-created_at'
        )
    
    async def get_configs_by_type(self, watermark_type: str, user_id: Optional[str] = None,
                                limit: int = 100, offset: int = 0) -> List[WatermarkConfig]:
        """根据水印类型获取配置"""
        filters = {'watermark_type': watermark_type, 'is_active': True}
        if user_id:
            filters['created_by'] = user_id
        
        return await self.find_by_filters(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by='-created_at'
        )
    
    async def get_by_name(self, config_name: str, user_id: str) -> Optional[WatermarkConfig]:
        """根据配置名称获取配置"""
        results = await self.find_by_filters(
            filters={'config_name': config_name, 'created_by': user_id},
            limit=1
        )
        return results[0] if results else None
    
    async def search_configs(self, keyword: str, user_id: Optional[str] = None,
                           limit: int = 100, offset: int = 0) -> List[WatermarkConfig]:
        """搜索配置"""
        async with self.get_session() as session:
            query = select(WatermarkConfig).where(
                WatermarkConfig.config_name.ilike(f"%{keyword}%")
            )
            
            if user_id:
                query = query.where(WatermarkConfig.created_by == user_id)
            
            query = query.order_by(WatermarkConfig.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def activate_config(self, config_id: str) -> bool:
        """激活配置"""
        result = await self.update(config_id, {'is_active': True})
        return result is not None
    
    async def deactivate_config(self, config_id: str) -> bool:
        """停用配置"""
        result = await self.update(config_id, {'is_active': False})
        return result is not None
    
    async def clone_config(self, config_id: str, new_name: str, user_id: str) -> Optional[WatermarkConfig]:
        """克隆配置"""
        original_config = await self.get_by_id(config_id)
        if not original_config:
            return None
        
        cloned_config = original_config.clone(new_name, user_id)
        return await self.create(cloned_config)