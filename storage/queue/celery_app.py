"""
Celery应用配置
配置和创建Celery应用实例
"""

import os
import logging
from typing import Dict, Any, Optional
from celery import Celery
from celery.signals import worker_ready, worker_shutdown
from kombu import Queue, Exchange

logger = logging.getLogger(__name__)


def create_celery_app(config: Dict[str, Any]) -> Celery:
    """创建Celery应用"""
    
    # 基础配置
    broker_url = config.get('broker_url', 'redis://localhost:6379/1')
    result_backend = config.get('result_backend', 'redis://localhost:6379/2')
    
    # 创建Celery应用
    app = Celery('watermark_system')
    
    # 配置Celery
    app.conf.update(
        # 消息代理配置
        broker_url=broker_url,
        result_backend=result_backend,
        
        # 任务序列化
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        
        # 任务路由
        task_routes={
            'watermark.document.*': {'queue': 'document_queue'},
            'watermark.image.*': {'queue': 'image_queue'},
            'watermark.media.*': {'queue': 'media_queue'},
            'watermark.extraction.*': {'queue': 'extraction_queue'},
            'watermark.report.*': {'queue': 'report_queue'},
        },
        
        # 队列配置
        task_default_queue='default',
        task_queues=(
            Queue('default', Exchange('default'), routing_key='default'),
            Queue('document_queue', Exchange('document'), routing_key='document'),
            Queue('image_queue', Exchange('image'), routing_key='image'),
            Queue('media_queue', Exchange('media'), routing_key='media'),
            Queue('extraction_queue', Exchange('extraction'), routing_key='extraction'),
            Queue('report_queue', Exchange('report'), routing_key='report'),
            Queue('priority_queue', Exchange('priority'), routing_key='priority'),
        ),
        
        # 任务执行配置
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_reject_on_worker_lost=True,
        
        # 任务结果配置
        result_expires=3600,  # 1小时
        result_persistent=True,
        
        # 任务重试配置
        task_default_retry_delay=60,  # 60秒
        task_max_retries=3,
        
        # 任务时间限制
        task_soft_time_limit=1800,  # 30分钟软限制
        task_time_limit=3600,       # 1小时硬限制
        
        # Worker配置
        worker_max_tasks_per_child=1000,
        worker_disable_rate_limits=False,
        
        # 监控配置
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # 安全配置
        worker_hijack_root_logger=False,
        worker_log_color=False,
        
        # 自定义配置
        **config.get('celery_settings', {})
    )
    
    # 自动发现任务
    app.autodiscover_tasks([
        'business.task_scheduler.tasks',
        'engines.document.tasks',
        'engines.image.tasks', 
        'engines.media.tasks',
        'business.extraction.tasks',
        'business.report.tasks'
    ])
    
    # 注册信号处理器
    @worker_ready.connect
    def worker_ready_handler(sender=None, **kwargs):
        logger.info(f"Celery worker ready: {sender}")
    
    @worker_shutdown.connect
    def worker_shutdown_handler(sender=None, **kwargs):
        logger.info(f"Celery worker shutdown: {sender}")
    
    return app


class CeleryConfig:
    """Celery配置类"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'broker_url': os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1'),
            'result_backend': os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2'),
            'celery_settings': {
                # 可以在这里添加更多自定义设置
            }
        }
    
    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        """获取开发环境配置"""
        config = CeleryConfig.get_default_config()
        config['celery_settings'].update({
            'task_always_eager': False,  # 开发时可以设为True进行同步测试
            'task_eager_propagates': True,
            'worker_log_level': 'DEBUG',
        })
        return config
    
    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """获取生产环境配置"""
        config = CeleryConfig.get_default_config()
        config['celery_settings'].update({
            'worker_log_level': 'INFO',
            'worker_max_memory_per_child': 200000,  # 200MB
            'worker_pool_restarts': True,
        })
        return config
    
    @staticmethod
    def get_test_config() -> Dict[str, Any]:
        """获取测试环境配置"""
        config = CeleryConfig.get_default_config()
        config['celery_settings'].update({
            'task_always_eager': True,
            'task_eager_propagates': True,
            'broker_url': 'memory://',
            'result_backend': 'cache+memory://',
        })
        return config


# 全局Celery应用实例
celery_app: Optional[Celery] = None


def get_celery_app() -> Celery:
    """获取Celery应用实例"""
    global celery_app
    
    if celery_app is None:
        # 根据环境变量选择配置
        env = os.getenv('ENVIRONMENT', 'development')
        
        if env == 'production':
            config = CeleryConfig.get_production_config()
        elif env == 'test':
            config = CeleryConfig.get_test_config()
        else:
            config = CeleryConfig.get_development_config()
        
        celery_app = create_celery_app(config)
    
    return celery_app


def init_celery_app(config: Dict[str, Any]) -> Celery:
    """初始化Celery应用"""
    global celery_app
    celery_app = create_celery_app(config)
    return celery_app