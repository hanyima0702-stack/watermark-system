"""
密钥管理数据访问对象
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_, func, desc
from sqlalchemy.orm import selectinload

from .base_dao import BaseDAO
from ..models.key_management import KeyManagement


class KeyManagementDAO(BaseDAO[KeyManagement]):
    """密钥管理数据访问对象"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager, KeyManagement)
    
    def get_model_class(self):
        return KeyManagement
    
    def _add_relationship_loading(self, query):
        """加载密钥关联对象"""
        return query.options(
            selectinload(KeyManagement.creator)
        )
    
    async def get_active_keys(self, key_type: Optional[str] = None,
                            limit: int = 100, offset: int = 0) -> List[KeyManagement]:
        """获取活跃密钥"""
        filters = {'is_active': True}
        if key_type:
            filters['key_type'] = key_type
        
        return await self.find_by_filters(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_keys_by_type(self, key_type: str, include_inactive: bool = False,
                             limit: int = 100, offset: int = 0) -> List[KeyManagement]:
        """根据密钥类型获取密钥"""
        filters = {'key_type': key_type}
        if not include_inactive:
            filters['is_active'] = True
        
        return await self.find_by_filters(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_current_key(self, key_type: str) -> Optional[KeyManagement]:
        """获取指定类型的当前密钥"""
        async with self.get_session() as session:
            query = select(KeyManagement).where(
                and_(
                    KeyManagement.key_type == key_type,
                    KeyManagement.is_active == True,
                    or_(
                        KeyManagement.expires_at.is_(None),
                        KeyManagement.expires_at > datetime.utcnow()
                    )
                )
            ).order_by(desc(KeyManagement.key_version)).limit(1)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def get_expired_keys(self, limit: int = 100, offset: int = 0) -> List[KeyManagement]:
        """获取已过期的密钥"""
        async with self.get_session() as session:
            query = select(KeyManagement).where(
                and_(
                    KeyManagement.expires_at.isnot(None),
                    KeyManagement.expires_at <= datetime.utcnow()
                )
            ).order_by(KeyManagement.expires_at).limit(limit).offset(offset)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_expiring_keys(self, days: int = 30, limit: int = 100) -> List[KeyManagement]:
        """获取即将过期的密钥"""
        warning_date = datetime.utcnow() + timedelta(days=days)
        async with self.get_session() as session:
            query = select(KeyManagement).where(
                and_(
                    KeyManagement.is_active == True,
                    KeyManagement.expires_at.isnot(None),
                    KeyManagement.expires_at <= warning_date,
                    KeyManagement.expires_at > datetime.utcnow()
                )
            ).order_by(KeyManagement.expires_at).limit(limit)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_keys_needing_rotation(self, max_age_days: int = 365) -> List[KeyManagement]:
        """获取需要轮换的密钥"""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        async with self.get_session() as session:
            query = select(KeyManagement).where(
                and_(
                    KeyManagement.is_active == True,
                    or_(
                        KeyManagement.created_at <= cutoff_date,
                        and_(
                            KeyManagement.expires_at.isnot(None),
                            KeyManagement.expires_at <= datetime.utcnow() + timedelta(days=30)
                        )
                    )
                )
            ).order_by(KeyManagement.created_at)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_key_versions(self, key_type: str, limit: int = 10) -> List[KeyManagement]:
        """获取密钥的所有版本"""
        return await self.find_by_filters(
            filters={'key_type': key_type},
            limit=limit,
            order_by='-key_version',
            load_relationships=True
        )
    
    async def deactivate_key(self, key_id: str) -> bool:
        """停用密钥"""
        result = await self.update(key_id, {
            'is_active': False,
            'rotated_at': datetime.utcnow()
        })
        return result is not None
    
    async def rotate_key(self, key_id: str, new_key_data: str,
                       new_expires_at: Optional[datetime] = None) -> Optional[KeyManagement]:
        """轮换密钥"""
        old_key = await self.get_by_id(key_id)
        if not old_key:
            return None
        
        # 停用旧密钥
        await self.deactivate_key(key_id)
        
        # 创建新密钥
        new_key = KeyManagement(
            key_id=KeyManagement.generate_id(),
            key_type=old_key.key_type,
            key_data=new_key_data,
            key_version=old_key.key_version + 1,
            is_active=True,
            expires_at=new_expires_at or (datetime.utcnow() + timedelta(days=365)),
            created_by=old_key.created_by
        )
        
        return await self.create(new_key)
    
    async def extend_key_expiration(self, key_id: str, days: int) -> bool:
        """延长密钥过期时间"""
        key = await self.get_by_id(key_id)
        if not key:
            return False
        
        if key.expires_at:
            new_expires_at = key.expires_at + timedelta(days=days)
        else:
            new_expires_at = datetime.utcnow() + timedelta(days=days)
        
        result = await self.update(key_id, {'expires_at': new_expires_at})
        return result is not None
    
    async def get_key_statistics(self) -> dict:
        """获取密钥统计信息"""
        async with self.get_session() as session:
            # 总密钥数
            total_keys = await self.count()
            
            # 活跃密钥数
            active_keys = await self.count({'is_active': True})
            
            # 按类型统计
            type_query = select(
                KeyManagement.key_type,
                func.count(KeyManagement.key_id).label('total'),
                func.sum(func.cast(KeyManagement.is_active, Integer)).label('active')
            ).group_by(KeyManagement.key_type)
            type_result = await session.execute(type_query)
            type_stats = {
                row.key_type: {'total': row.total, 'active': row.active}
                for row in type_result
            }
            
            # 即将过期的密钥数
            expiring_count = len(await self.get_expiring_keys())
            
            # 已过期的密钥数
            expired_count = len(await self.get_expired_keys())
            
            return {
                'total_keys': total_keys,
                'active_keys': active_keys,
                'inactive_keys': total_keys - active_keys,
                'expiring_keys': expiring_count,
                'expired_keys': expired_count,
                'type_distribution': type_stats
            }
    
    async def cleanup_old_keys(self, days: int = 1095) -> int:  # 3年
        """清理旧密钥"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        async with self.get_session() as session:
            query = delete(KeyManagement).where(
                and_(
                    KeyManagement.is_active == False,
                    KeyManagement.rotated_at < cutoff_time
                )
            )
            result = await session.execute(query)
            return result.rowcount