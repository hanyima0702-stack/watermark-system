"""
队列监控器
提供队列监控和管理工具
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from celery import Celery
from .celery_app import get_celery_app
from ..cache.cache_manager import CacheManager, CacheNamespace

logger = logging.getLogger(__name__)


@dataclass
class QueueMetrics:
    """队列指标"""
    queue_name: str
    active_tasks: int
    scheduled_tasks: int
    reserved_tasks: int
    failed_tasks: int
    processed_tasks: int
    avg_processing_time: float
    last_updated: datetime


@dataclass
class WorkerMetrics:
    """Worker指标"""
    worker_name: str
    status: str
    active_tasks: int
    processed_tasks: int
    failed_tasks: int
    load_average: List[float]
    memory_usage: Dict[str, Any]
    last_heartbeat: datetime


class QueueMonitor:
    """队列监控器"""
    
    def __init__(self, cache_manager: CacheManager, config: Dict[str, Any]):
        self.cache_manager = cache_manager
        self.config = config
        self.celery_app = get_celery_app()
        
        # 监控配置
        self.monitor_interval = config.get('monitor_interval', 30)  # 30秒
        self.metrics_retention = config.get('metrics_retention', 24 * 3600)  # 24小时
        self.alert_thresholds = config.get('alert_thresholds', {
            'queue_length': 100,
            'failed_rate': 0.1,
            'processing_time': 300
        })
        
        # 监控任务
        self._monitor_task = None
        self._running = False
    
    async def start_monitoring(self):
        """开始监控"""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("队列监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("队列监控已停止")
    
    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                # 收集队列指标
                await self._collect_queue_metrics()
                
                # 收集Worker指标
                await self._collect_worker_metrics()
                
                # 检查告警条件
                await self._check_alerts()
                
                # 等待下次监控
                await asyncio.sleep(self.monitor_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    async def _collect_queue_metrics(self):
        """收集队列指标"""
        try:
            inspect = self.celery_app.control.inspect()
            
            # 获取队列信息
            active_tasks = inspect.active() or {}
            scheduled_tasks = inspect.scheduled() or {}
            reserved_tasks = inspect.reserved() or {}
            
            # 队列列表
            queues = ['default', 'document_queue', 'image_queue', 'media_queue', 
                     'extraction_queue', 'report_queue', 'priority_queue']
            
            for queue_name in queues:
                # 统计各队列的任务数
                active_count = sum(
                    len([t for t in tasks if t.get('delivery_info', {}).get('routing_key') == queue_name])
                    for tasks in active_tasks.values()
                )
                
                scheduled_count = sum(
                    len([t for t in tasks if t.get('delivery_info', {}).get('routing_key') == queue_name])
                    for tasks in scheduled_tasks.values()
                )
                
                reserved_count = sum(
                    len([t for t in tasks if t.get('delivery_info', {}).get('routing_key') == queue_name])
                    for tasks in reserved_tasks.values()
                )
                
                # 从缓存获取历史数据
                failed_tasks = await self._get_queue_failed_count(queue_name)
                processed_tasks = await self._get_queue_processed_count(queue_name)
                avg_processing_time = await self._get_queue_avg_time(queue_name)
                
                # 创建队列指标
                metrics = QueueMetrics(
                    queue_name=queue_name,
                    active_tasks=active_count,
                    scheduled_tasks=scheduled_count,
                    reserved_tasks=reserved_count,
                    failed_tasks=failed_tasks,
                    processed_tasks=processed_tasks,
                    avg_processing_time=avg_processing_time,
                    last_updated=datetime.utcnow()
                )
                
                # 保存指标
                await self._save_queue_metrics(metrics)
            
        except Exception as e:
            logger.error(f"收集队列指标失败: {e}")
    
    async def _collect_worker_metrics(self):
        """收集Worker指标"""
        try:
            inspect = self.celery_app.control.inspect()
            
            # 获取Worker统计信息
            stats = inspect.stats() or {}
            active_tasks = inspect.active() or {}
            
            for worker_name, worker_stats in stats.items():
                # 获取Worker活跃任务数
                worker_active = len(active_tasks.get(worker_name, []))
                
                # 从统计信息获取其他指标
                pool_stats = worker_stats.get('pool', {})
                rusage = worker_stats.get('rusage', {})
                
                # 创建Worker指标
                metrics = WorkerMetrics(
                    worker_name=worker_name,
                    status='online',
                    active_tasks=worker_active,
                    processed_tasks=worker_stats.get('total', {}).get('tasks.completed', 0),
                    failed_tasks=worker_stats.get('total', {}).get('tasks.failed', 0),
                    load_average=pool_stats.get('processes', []),
                    memory_usage={
                        'rss': rusage.get('maxrss', 0),
                        'vms': rusage.get('vms', 0)
                    },
                    last_heartbeat=datetime.utcnow()
                )
                
                # 保存指标
                await self._save_worker_metrics(metrics)
            
        except Exception as e:
            logger.error(f"收集Worker指标失败: {e}")
    
    async def _save_queue_metrics(self, metrics: QueueMetrics):
        """保存队列指标"""
        try:
            # 保存当前指标
            await self.cache_manager.set(
                CacheNamespace.STATS,
                f"queue_metrics:{metrics.queue_name}",
                {
                    'queue_name': metrics.queue_name,
                    'active_tasks': metrics.active_tasks,
                    'scheduled_tasks': metrics.scheduled_tasks,
                    'reserved_tasks': metrics.reserved_tasks,
                    'failed_tasks': metrics.failed_tasks,
                    'processed_tasks': metrics.processed_tasks,
                    'avg_processing_time': metrics.avg_processing_time,
                    'last_updated': metrics.last_updated.isoformat()
                },
                expire=self.metrics_retention
            )
            
            # 保存历史指标
            timestamp = int(metrics.last_updated.timestamp())
            await self.cache_manager.set(
                CacheNamespace.STATS,
                f"queue_history:{metrics.queue_name}:{timestamp}",
                {
                    'active_tasks': metrics.active_tasks,
                    'scheduled_tasks': metrics.scheduled_tasks,
                    'reserved_tasks': metrics.reserved_tasks,
                    'timestamp': timestamp
                },
                expire=self.metrics_retention
            )
            
        except Exception as e:
            logger.error(f"保存队列指标失败: {e}")
    
    async def _save_worker_metrics(self, metrics: WorkerMetrics):
        """保存Worker指标"""
        try:
            await self.cache_manager.set(
                CacheNamespace.STATS,
                f"worker_metrics:{metrics.worker_name}",
                {
                    'worker_name': metrics.worker_name,
                    'status': metrics.status,
                    'active_tasks': metrics.active_tasks,
                    'processed_tasks': metrics.processed_tasks,
                    'failed_tasks': metrics.failed_tasks,
                    'load_average': metrics.load_average,
                    'memory_usage': metrics.memory_usage,
                    'last_heartbeat': metrics.last_heartbeat.isoformat()
                },
                expire=self.metrics_retention
            )
            
        except Exception as e:
            logger.error(f"保存Worker指标失败: {e}")
    
    async def _get_queue_failed_count(self, queue_name: str) -> int:
        """获取队列失败任务数"""
        try:
            # 从缓存获取失败计数
            count = await self.cache_manager.get(
                CacheNamespace.STATS,
                f"queue_failed:{queue_name}"
            )
            return count or 0
        except:
            return 0
    
    async def _get_queue_processed_count(self, queue_name: str) -> int:
        """获取队列处理任务数"""
        try:
            count = await self.cache_manager.get(
                CacheNamespace.STATS,
                f"queue_processed:{queue_name}"
            )
            return count or 0
        except:
            return 0
    
    async def _get_queue_avg_time(self, queue_name: str) -> float:
        """获取队列平均处理时间"""
        try:
            avg_time = await self.cache_manager.get(
                CacheNamespace.STATS,
                f"queue_avg_time:{queue_name}"
            )
            return avg_time or 0.0
        except:
            return 0.0
    
    async def _check_alerts(self):
        """检查告警条件"""
        try:
            # 检查队列长度告警
            await self._check_queue_length_alerts()
            
            # 检查失败率告警
            await self._check_failure_rate_alerts()
            
            # 检查处理时间告警
            await self._check_processing_time_alerts()
            
        except Exception as e:
            logger.error(f"检查告警失败: {e}")
    
    async def _check_queue_length_alerts(self):
        """检查队列长度告警"""
        threshold = self.alert_thresholds.get('queue_length', 100)
        
        queue_keys = await self.cache_manager.keys(CacheNamespace.STATS, "queue_metrics:*")
        
        for queue_key in queue_keys:
            metrics = await self.cache_manager.get(CacheNamespace.STATS, queue_key)
            if metrics:
                total_tasks = (metrics.get('active_tasks', 0) + 
                             metrics.get('scheduled_tasks', 0) + 
                             metrics.get('reserved_tasks', 0))
                
                if total_tasks > threshold:
                    await self._send_alert(
                        'queue_length',
                        f"队列 {metrics['queue_name']} 任务数量过多: {total_tasks}",
                        {'queue_name': metrics['queue_name'], 'task_count': total_tasks}
                    )
    
    async def _check_failure_rate_alerts(self):
        """检查失败率告警"""
        threshold = self.alert_thresholds.get('failed_rate', 0.1)
        
        queue_keys = await self.cache_manager.keys(CacheNamespace.STATS, "queue_metrics:*")
        
        for queue_key in queue_keys:
            metrics = await self.cache_manager.get(CacheNamespace.STATS, queue_key)
            if metrics:
                failed = metrics.get('failed_tasks', 0)
                processed = metrics.get('processed_tasks', 0)
                
                if processed > 0:
                    failure_rate = failed / processed
                    if failure_rate > threshold:
                        await self._send_alert(
                            'failure_rate',
                            f"队列 {metrics['queue_name']} 失败率过高: {failure_rate:.2%}",
                            {'queue_name': metrics['queue_name'], 'failure_rate': failure_rate}
                        )
    
    async def _check_processing_time_alerts(self):
        """检查处理时间告警"""
        threshold = self.alert_thresholds.get('processing_time', 300)
        
        queue_keys = await self.cache_manager.keys(CacheNamespace.STATS, "queue_metrics:*")
        
        for queue_key in queue_keys:
            metrics = await self.cache_manager.get(CacheNamespace.STATS, queue_key)
            if metrics:
                avg_time = metrics.get('avg_processing_time', 0)
                
                if avg_time > threshold:
                    await self._send_alert(
                        'processing_time',
                        f"队列 {metrics['queue_name']} 平均处理时间过长: {avg_time:.1f}秒",
                        {'queue_name': metrics['queue_name'], 'avg_time': avg_time}
                    )
    
    async def _send_alert(self, alert_type: str, message: str, data: Dict[str, Any]):
        """发送告警"""
        try:
            alert_data = {
                'type': alert_type,
                'message': message,
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'severity': 'warning'
            }
            
            # 保存告警记录
            await self.cache_manager.set(
                CacheNamespace.STATS,
                f"alert:{alert_type}:{int(datetime.utcnow().timestamp())}",
                alert_data,
                expire=7 * 24 * 3600  # 7天
            )
            
            logger.warning(f"队列告警: {message}")
            
        except Exception as e:
            logger.error(f"发送告警失败: {e}")
    
    async def get_queue_metrics(self, queue_name: str = None) -> Dict[str, Any]:
        """获取队列指标"""
        try:
            if queue_name:
                # 获取特定队列指标
                metrics = await self.cache_manager.get(
                    CacheNamespace.STATS,
                    f"queue_metrics:{queue_name}"
                )
                return metrics or {}
            else:
                # 获取所有队列指标
                queue_keys = await self.cache_manager.keys(CacheNamespace.STATS, "queue_metrics:*")
                all_metrics = {}
                
                for queue_key in queue_keys:
                    metrics = await self.cache_manager.get(CacheNamespace.STATS, queue_key)
                    if metrics:
                        queue_name = metrics['queue_name']
                        all_metrics[queue_name] = metrics
                
                return all_metrics
                
        except Exception as e:
            logger.error(f"获取队列指标失败: {e}")
            return {}
    
    async def get_worker_metrics(self) -> Dict[str, Any]:
        """获取Worker指标"""
        try:
            worker_keys = await self.cache_manager.keys(CacheNamespace.STATS, "worker_metrics:*")
            all_metrics = {}
            
            for worker_key in worker_keys:
                metrics = await self.cache_manager.get(CacheNamespace.STATS, worker_key)
                if metrics:
                    worker_name = metrics['worker_name']
                    all_metrics[worker_name] = metrics
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"获取Worker指标失败: {e}")
            return {}
    
    async def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取告警记录"""
        try:
            alert_keys = await self.cache_manager.keys(CacheNamespace.STATS, "alert:*")
            alerts = []
            
            # 按时间戳排序
            alert_keys.sort(reverse=True)
            
            for alert_key in alert_keys[:limit]:
                alert_data = await self.cache_manager.get(CacheNamespace.STATS, alert_key)
                if alert_data:
                    alerts.append(alert_data)
            
            return alerts
            
        except Exception as e:
            logger.error(f"获取告警记录失败: {e}")
            return []