"""
任务管理器
提供异步任务调度和管理功能
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict

from celery import Celery
from celery.result import AsyncResult
from celery.exceptions import Retry, WorkerLostError

from .celery_app import get_celery_app
from ..cache.cache_manager import CacheManager, CacheNamespace

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    task_name: str
    queue: str
    priority: TaskPriority
    args: List[Any]
    kwargs: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskInfo':
        """从字典创建"""
        data['priority'] = TaskPriority(data['priority'])
        data['status'] = TaskStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


class TaskManager:
    """任务管理器"""
    
    def __init__(self, cache_manager: CacheManager, config: Dict[str, Any]):
        self.cache_manager = cache_manager
        self.config = config
        self.celery_app = get_celery_app()
        
        # 任务配置
        self.default_queue = config.get('default_queue', 'default')
        self.task_timeout = config.get('task_timeout', 3600)  # 1小时
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 60)  # 60秒
        
        # 队列映射
        self.queue_mapping = {
            'document': 'document_queue',
            'image': 'image_queue',
            'media': 'media_queue',
            'extraction': 'extraction_queue',
            'report': 'report_queue',
            'priority': 'priority_queue'
        }
    
    async def submit_task(self, task_name: str, args: List[Any] = None, 
                         kwargs: Dict[str, Any] = None, queue: str = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         countdown: int = 0, eta: datetime = None,
                         expires: datetime = None) -> str:
        """提交任务"""
        try:
            args = args or []
            kwargs = kwargs or {}
            
            # 确定队列
            if queue is None:
                queue = self.default_queue
            elif queue in self.queue_mapping:
                queue = self.queue_mapping[queue]
            
            # 设置任务选项
            task_options = {
                'queue': queue,
                'priority': priority.value,
                'countdown': countdown,
                'eta': eta,
                'expires': expires,
                'retry': True,
                'retry_policy': {
                    'max_retries': self.max_retries,
                    'interval_start': self.retry_delay,
                    'interval_step': 0.2,
                    'interval_max': 200,
                }
            }
            
            # 提交任务
            result = self.celery_app.send_task(
                task_name,
                args=args,
                kwargs=kwargs,
                **task_options
            )
            
            # 创建任务信息
            task_info = TaskInfo(
                task_id=result.id,
                task_name=task_name,
                queue=queue,
                priority=priority,
                args=args,
                kwargs=kwargs,
                status=TaskStatus.PENDING,
                created_at=datetime.utcnow(),
                max_retries=self.max_retries
            )
            
            # 缓存任务信息
            await self.cache_manager.set(
                CacheNamespace.TASK,
                f"task_info:{result.id}",
                task_info.to_dict(),
                expire=self.task_timeout * 2
            )
            
            logger.info(f"任务提交成功: {task_name} -> {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"提交任务失败: {task_name}, {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        try:
            # 从Celery获取状态
            result = AsyncResult(task_id, app=self.celery_app)
            celery_status = result.status
            
            # 转换状态
            if celery_status in TaskStatus.__members__:
                return TaskStatus(celery_status)
            else:
                return TaskStatus.PENDING
                
        except Exception as e:
            logger.error(f"获取任务状态失败: {task_id}, {e}")
            return None
    
    async def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        try:
            # 从缓存获取任务信息
            task_dict = await self.cache_manager.get(
                CacheNamespace.TASK,
                f"task_info:{task_id}"
            )
            
            if task_dict:
                task_info = TaskInfo.from_dict(task_dict)
                
                # 更新状态
                current_status = await self.get_task_status(task_id)
                if current_status:
                    task_info.status = current_status
                
                return task_info
            
            # 如果缓存中没有，尝试从Celery获取基本信息
            result = AsyncResult(task_id, app=self.celery_app)
            if result.status:
                return TaskInfo(
                    task_id=task_id,
                    task_name="unknown",
                    queue="unknown",
                    priority=TaskPriority.NORMAL,
                    args=[],
                    kwargs={},
                    status=TaskStatus(result.status),
                    created_at=datetime.utcnow(),
                    result=result.result if result.successful() else None,
                    error=str(result.result) if result.failed() else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取任务信息失败: {task_id}, {e}")
            return None
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """获取任务结果"""
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            # 异步等待结果
            if timeout:
                return result.get(timeout=timeout)
            else:
                return result.result if result.ready() else None
                
        except Exception as e:
            logger.error(f"获取任务结果失败: {task_id}, {e}")
            return None
    
    async def cancel_task(self, task_id: str, terminate: bool = False) -> bool:
        """取消任务"""
        try:
            # 撤销任务
            self.celery_app.control.revoke(task_id, terminate=terminate)
            
            # 更新任务信息
            task_info = await self.get_task_info(task_id)
            if task_info:
                task_info.status = TaskStatus.REVOKED
                task_info.completed_at = datetime.utcnow()
                
                await self.cache_manager.set(
                    CacheNamespace.TASK,
                    f"task_info:{task_id}",
                    task_info.to_dict()
                )
            
            logger.info(f"任务取消成功: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消任务失败: {task_id}, {e}")
            return False
    
    async def retry_task(self, task_id: str, countdown: int = None) -> bool:
        """重试任务"""
        try:
            task_info = await self.get_task_info(task_id)
            if not task_info:
                return False
            
            # 检查重试次数
            if task_info.retry_count >= task_info.max_retries:
                logger.warning(f"任务重试次数已达上限: {task_id}")
                return False
            
            # 重新提交任务
            new_task_id = await self.submit_task(
                task_name=task_info.task_name,
                args=task_info.args,
                kwargs=task_info.kwargs,
                queue=task_info.queue,
                priority=task_info.priority,
                countdown=countdown or self.retry_delay
            )
            
            # 更新重试信息
            new_task_info = await self.get_task_info(new_task_id)
            if new_task_info:
                new_task_info.retry_count = task_info.retry_count + 1
                await self.cache_manager.set(
                    CacheNamespace.TASK,
                    f"task_info:{new_task_id}",
                    new_task_info.to_dict()
                )
            
            logger.info(f"任务重试成功: {task_id} -> {new_task_id}")
            return True
            
        except Exception as e:
            logger.error(f"重试任务失败: {task_id}, {e}")
            return False
    
    async def get_queue_stats(self, queue_name: str = None) -> Dict[str, Any]:
        """获取队列统计信息"""
        try:
            # 获取队列信息
            inspect = self.celery_app.control.inspect()
            
            # 活跃任务
            active_tasks = inspect.active()
            
            # 预定任务
            scheduled_tasks = inspect.scheduled()
            
            # 保留任务
            reserved_tasks = inspect.reserved()
            
            # 统计信息
            stats = {
                'active_tasks': 0,
                'scheduled_tasks': 0,
                'reserved_tasks': 0,
                'workers': []
            }
            
            if active_tasks:
                for worker, tasks in active_tasks.items():
                    worker_active = len([t for t in tasks if not queue_name or t.get('delivery_info', {}).get('routing_key') == queue_name])
                    stats['active_tasks'] += worker_active
                    stats['workers'].append({
                        'worker': worker,
                        'active_tasks': worker_active
                    })
            
            if scheduled_tasks:
                for worker, tasks in scheduled_tasks.items():
                    worker_scheduled = len([t for t in tasks if not queue_name or t.get('delivery_info', {}).get('routing_key') == queue_name])
                    stats['scheduled_tasks'] += worker_scheduled
            
            if reserved_tasks:
                for worker, tasks in reserved_tasks.items():
                    worker_reserved = len([t for t in tasks if not queue_name or t.get('delivery_info', {}).get('routing_key') == queue_name])
                    stats['reserved_tasks'] += worker_reserved
            
            return stats
            
        except Exception as e:
            logger.error(f"获取队列统计失败: {e}")
            return {}
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """获取Worker统计信息"""
        try:
            inspect = self.celery_app.control.inspect()
            
            # Worker状态
            stats = inspect.stats()
            
            # 注册的任务
            registered = inspect.registered()
            
            # 活跃队列
            active_queues = inspect.active_queues()
            
            return {
                'workers': list(stats.keys()) if stats else [],
                'worker_count': len(stats) if stats else 0,
                'registered_tasks': registered,
                'active_queues': active_queues,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"获取Worker统计失败: {e}")
            return {}
    
    async def purge_queue(self, queue_name: str) -> int:
        """清空队列"""
        try:
            # 清空指定队列
            result = self.celery_app.control.purge()
            
            logger.info(f"队列清空成功: {queue_name}")
            return result
            
        except Exception as e:
            logger.error(f"清空队列失败: {queue_name}, {e}")
            return 0
    
    async def get_task_history(self, limit: int = 100, 
                             status: TaskStatus = None) -> List[TaskInfo]:
        """获取任务历史"""
        try:
            # 从缓存获取任务信息
            task_keys = await self.cache_manager.keys(CacheNamespace.TASK, "task_info:*")
            
            tasks = []
            for task_key in task_keys[:limit]:
                task_dict = await self.cache_manager.get(CacheNamespace.TASK, task_key)
                if task_dict:
                    task_info = TaskInfo.from_dict(task_dict)
                    
                    # 过滤状态
                    if status is None or task_info.status == status:
                        tasks.append(task_info)
            
            # 按创建时间排序
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            
            return tasks
            
        except Exception as e:
            logger.error(f"获取任务历史失败: {e}")
            return []
    
    async def cleanup_completed_tasks(self, days: int = 7) -> int:
        """清理已完成的任务"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            task_keys = await self.cache_manager.keys(CacheNamespace.TASK, "task_info:*")
            cleaned_count = 0
            
            for task_key in task_keys:
                task_dict = await self.cache_manager.get(CacheNamespace.TASK, task_key)
                if task_dict:
                    task_info = TaskInfo.from_dict(task_dict)
                    
                    # 检查是否为已完成的旧任务
                    if (task_info.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.REVOKED] and
                        task_info.completed_at and task_info.completed_at < cutoff_time):
                        
                        await self.cache_manager.delete(CacheNamespace.TASK, task_key)
                        cleaned_count += 1
            
            logger.info(f"清理已完成任务: {cleaned_count} 个")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理已完成任务失败: {e}")
            return 0