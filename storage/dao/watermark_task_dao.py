"""
水印任务数据访问对象
提供水印任务相关的数据库操作
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import select, and_, func, desc
from sqlalchemy.orm import selectinload

from .base_dao import BaseDAO
from ..models.watermark_task import WatermarkTask


class WatermarkTaskDAO(BaseDAO[WatermarkTask]):
    """水印任务数据访问对象"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager, WatermarkTask)
    
    def get_model_class(self):
        return WatermarkTask
    
    def _add_relationship_loading(self, query):
        """加载任务关联对象"""
        return query.options(
            selectinload(WatermarkTask.user),
            selectinload(WatermarkTask.file),
            selectinload(WatermarkTask.config),
            selectinload(WatermarkTask.output_file)
        )
    
    async def get_user_tasks(self, user_id: str, status: Optional[str] = None,
                           limit: int = 100, offset: int = 0) -> List[WatermarkTask]:
        """获取用户任务列表"""
        filters = {'user_id': user_id}
        if status:
            filters['status'] = status
        
        return await self.find_by_filters(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_pending_tasks(self, limit: int = 100) -> List[WatermarkTask]:
        """获取待处理任务"""
        return await self.find_by_filters(
            filters={'status': 'pending'},
            limit=limit,
            order_by='created_at',
            load_relationships=True
        )
    
    async def get_processing_tasks(self, engine: Optional[str] = None) -> List[WatermarkTask]:
        """获取正在处理的任务"""
        filters = {'status': 'processing'}
        if engine:
            filters['processing_engine'] = engine
        
        return await self.find_by_filters(
            filters=filters,
            order_by='created_at',
            load_relationships=True
        )
    
    async def get_tasks_by_status(self, status: str, limit: int = 100, offset: int = 0) -> List[WatermarkTask]:
        """根据状态获取任务"""
        return await self.find_by_filters(
            filters={'status': status},
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_tasks_by_type(self, task_type: str, limit: int = 100, offset: int = 0) -> List[WatermarkTask]:
        """根据任务类型获取任务"""
        return await self.find_by_filters(
            filters={'task_type': task_type},
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_recent_tasks(self, hours: int = 24, limit: int = 100) -> List[WatermarkTask]:
        """获取最近的任务"""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        async with self.get_session() as session:
            query = select(WatermarkTask).where(
                WatermarkTask.created_at >= since_time
            ).order_by(desc(WatermarkTask.created_at)).limit(limit)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_failed_tasks(self, limit: int = 100, offset: int = 0) -> List[WatermarkTask]:
        """获取失败的任务"""
        return await self.find_by_filters(
            filters={'status': 'failed'},
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_timeout_tasks(self, timeout_hours: int = 24) -> List[WatermarkTask]:
        """获取超时的任务"""
        timeout_time = datetime.utcnow() - timedelta(hours=timeout_hours)
        async with self.get_session() as session:
            query = select(WatermarkTask).where(
                and_(
                    WatermarkTask.status == 'processing',
                    WatermarkTask.created_at < timeout_time
                )
            )
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def update_task_progress(self, task_id: str, progress: float) -> bool:
        """更新任务进度"""
        result = await self.update(task_id, {'progress': progress})
        return result is not None
    
    async def start_task_processing(self, task_id: str, engine: str) -> bool:
        """开始处理任务"""
        result = await self.update(task_id, {
            'status': 'processing',
            'processing_engine': engine,
            'progress': 0.0
        })
        return result is not None
    
    async def complete_task(self, task_id: str, output_file_id: Optional[str] = None,
                          processing_time: Optional[float] = None,
                          quality_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """完成任务"""
        update_data = {
            'status': 'completed',
            'progress': 100.0,
            'completed_at': datetime.utcnow()
        }
        
        if output_file_id:
            update_data['output_file_id'] = output_file_id
        if processing_time:
            update_data['processing_time'] = processing_time
        if quality_metrics:
            update_data['quality_metrics'] = quality_metrics
        
        result = await self.update(task_id, update_data)
        return result is not None
    
    async def fail_task(self, task_id: str, error_message: str) -> bool:
        """任务失败"""
        result = await self.update(task_id, {
            'status': 'failed',
            'error_message': error_message,
            'completed_at': datetime.utcnow()
        })
        return result is not None
    
    async def cancel_task(self, task_id: str, reason: str = "用户取消") -> bool:
        """取消任务"""
        result = await self.update(task_id, {
            'status': 'cancelled',
            'error_message': reason,
            'completed_at': datetime.utcnow()
        })
        return result is not None
    
    async def get_task_statistics(self, user_id: Optional[str] = None, 
                                days: int = 30) -> Dict[str, Any]:
        """获取任务统计信息"""
        async with self.get_session() as session:
            # 基础查询条件
            base_query = select(WatermarkTask)
            if user_id:
                base_query = base_query.where(WatermarkTask.user_id == user_id)
            
            # 时间范围
            since_time = datetime.utcnow() - timedelta(days=days)
            base_query = base_query.where(WatermarkTask.created_at >= since_time)
            
            # 总任务数
            total_query = select(func.count()).select_from(base_query.subquery())
            total_result = await session.execute(total_query)
            total_tasks = total_result.scalar()
            
            # 按状态统计
            status_query = select(
                WatermarkTask.status,
                func.count(WatermarkTask.task_id).label('count')
            ).select_from(base_query.subquery()).group_by(WatermarkTask.status)
            status_result = await session.execute(status_query)
            status_stats = {row.status: row.count for row in status_result}
            
            # 按任务类型统计
            type_query = select(
                WatermarkTask.task_type,
                func.count(WatermarkTask.task_id).label('count')
            ).select_from(base_query.subquery()).group_by(WatermarkTask.task_type)
            type_result = await session.execute(type_query)
            type_stats = {row.task_type: row.count for row in type_result}
            
            # 平均处理时间
            avg_time_query = select(
                func.avg(WatermarkTask.processing_time)
            ).select_from(
                base_query.where(WatermarkTask.processing_time.isnot(None)).subquery()
            )
            avg_time_result = await session.execute(avg_time_query)
            avg_processing_time = avg_time_result.scalar() or 0
            
            return {
                'total_tasks': total_tasks,
                'status_distribution': status_stats,
                'type_distribution': type_stats,
                'avg_processing_time': float(avg_processing_time),
                'success_rate': (status_stats.get('completed', 0) / total_tasks * 100) if total_tasks > 0 else 0
            }
    
    async def cleanup_old_tasks(self, days: int = 90) -> int:
        """清理旧任务记录"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        async with self.get_session() as session:
            query = delete(WatermarkTask).where(
                and_(
                    WatermarkTask.created_at < cutoff_time,
                    WatermarkTask.status.in_(['completed', 'failed', 'cancelled'])
                )
            )
            result = await session.execute(query)
            return result.rowcount