# Task Scheduler Service — RabbitMQ 异步任务调度
#
# 核心模块:
#   task_protocol   - 标准化消息协议 (无外部依赖)
#   rabbitmq_client - RabbitMQ 连接管理 (依赖 aio_pika)
#   task_producer   - 任务生产者 (依赖 rabbitmq_client)
#   task_consumer   - Worker 消费者 (依赖所有引擎)
#
# task_protocol 可以独立导入，其余模块按需 lazy import

from .task_protocol import (
    WatermarkTaskMessage,
    FileCategory,
    WatermarkAction,
    TaskPriority,
    should_use_async,
    QUEUE_ROUTING,
    ASYNC_THRESHOLD_BYTES,
)


def get_task_producer():
    """延迟导入，避免在 aio_pika 未安装时报错"""
    from .task_producer import get_task_producer as _get
    return _get()


def init_task_producer(rabbitmq_url: str):
    from .task_producer import init_task_producer as _init
    return _init(rabbitmq_url)


def shutdown_task_producer():
    from .task_producer import shutdown_task_producer as _shutdown
    return _shutdown()


__all__ = [
    "WatermarkTaskMessage",
    "FileCategory",
    "WatermarkAction",
    "TaskPriority",
    "should_use_async",
    "QUEUE_ROUTING",
    "ASYNC_THRESHOLD_BYTES",
    "get_task_producer",
    "init_task_producer",
    "shutdown_task_producer",
]
