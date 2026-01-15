"""
审计日志数据访问对象
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_, func, desc
from sqlalchemy.orm import selectinload

from .base_dao import BaseDAO
from ..models.audit_log import AuditLog


class AuditLogDAO(BaseDAO[AuditLog]):
    """审计日志数据访问对象"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager, AuditLog)
    
    def get_model_class(self):
        return AuditLog
    
    def _add_relationship_loading(self, query):
        """加载日志关联对象"""
        return query.options(
            selectinload(AuditLog.user)
        )
    
    async def get_user_logs(self, user_id: str, action: Optional[str] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """获取用户日志"""
        filters = {'user_id': user_id}
        if action:
            filters['action'] = action
        
        # 时间范围过滤需要特殊处理
        if start_time or end_time:
            time_filters = {}
            if start_time:
                time_filters['gte'] = start_time
            if end_time:
                time_filters['lte'] = end_time
            filters['timestamp'] = time_filters
        
        return await self.find_by_filters(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by='-timestamp',
            load_relationships=True
        )
    
    async def get_logs_by_action(self, action: str, limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """根据操作类型获取日志"""
        return await self.find_by_filters(
            filters={'action': action},
            limit=limit,
            offset=offset,
            order_by='-timestamp',
            load_relationships=True
        )
    
    async def get_logs_by_resource(self, resource_type: str, resource_id: Optional[str] = None,
                                 limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """根据资源类型获取日志"""
        filters = {'resource_type': resource_type}
        if resource_id:
            filters['resource_id'] = resource_id
        
        return await self.find_by_filters(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by='-timestamp',
            load_relationships=True
        )
    
    async def get_security_events(self, limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """获取安全事件日志"""
        security_actions = [
            'login_failed', 'unauthorized_access', 'permission_denied',
            'password_changed', 'account_locked', 'suspicious_activity'
        ]
        
        return await self.find_by_filters(
            filters={'action': security_actions},
            limit=limit,
            offset=offset,
            order_by='-timestamp',
            load_relationships=True
        )
    
    async def get_failed_operations(self, limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """获取失败的操作日志"""
        async with self.get_session() as session:
            query = select(AuditLog).where(
                AuditLog.details['result'].astext.in_(['failed', 'error'])
            ).order_by(desc(AuditLog.timestamp)).limit(limit).offset(offset)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_recent_logs(self, hours: int = 24, limit: int = 100) -> List[AuditLog]:
        """获取最近的日志"""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        async with self.get_session() as session:
            query = select(AuditLog).where(
                AuditLog.timestamp >= since_time
            ).order_by(desc(AuditLog.timestamp)).limit(limit)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def search_logs(self, keyword: str, user_id: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """搜索日志"""
        async with self.get_session() as session:
            query = select(AuditLog).where(
                or_(
                    AuditLog.action.ilike(f"%{keyword}%"),
                    AuditLog.resource_type.ilike(f"%{keyword}%"),
                    AuditLog.resource_id.ilike(f"%{keyword}%")
                )
            )
            
            if user_id:
                query = query.where(AuditLog.user_id == user_id)
            
            if start_time:
                query = query.where(AuditLog.timestamp >= start_time)
            
            if end_time:
                query = query.where(AuditLog.timestamp <= end_time)
            
            query = query.order_by(desc(AuditLog.timestamp)).limit(limit).offset(offset)
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_audit_statistics(self, days: int = 30) -> dict:
        """获取审计统计信息"""
        async with self.get_session() as session:
            since_time = datetime.utcnow() - timedelta(days=days)
            base_query = select(AuditLog).where(AuditLog.timestamp >= since_time)
            
            # 总日志数
            total_count = await self.count({'timestamp': {'gte': since_time}})
            
            # 按操作统计
            action_query = select(
                AuditLog.action,
                func.count(AuditLog.log_id).label('count')
            ).select_from(base_query.subquery()).group_by(AuditLog.action)
            action_result = await session.execute(action_query)
            action_stats = {row.action: row.count for row in action_result}
            
            # 按用户统计
            user_query = select(
                AuditLog.user_id,
                func.count(AuditLog.log_id).label('count')
            ).select_from(base_query.subquery()).group_by(AuditLog.user_id).order_by(desc('count')).limit(10)
            user_result = await session.execute(user_query)
            user_stats = {row.user_id: row.count for row in user_result}
            
            # 安全事件统计
            security_actions = [
                'login_failed', 'unauthorized_access', 'permission_denied',
                'password_changed', 'account_locked', 'suspicious_activity'
            ]
            security_query = select(func.count()).select_from(
                base_query.where(AuditLog.action.in_(security_actions)).subquery()
            )
            security_result = await session.execute(security_query)
            security_count = security_result.scalar()
            
            return {
                'total_logs': total_count,
                'security_events': security_count,
                'action_distribution': action_stats,
                'top_users': user_stats
            }
    
    async def cleanup_old_logs(self, days: int = 365) -> int:
        """清理旧日志"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        async with self.get_session() as session:
            query = delete(AuditLog).where(AuditLog.timestamp < cutoff_time)
            result = await session.execute(query)
            return result.rowcount