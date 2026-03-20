"""
任务生产者
API Gateway 层调用，将水印处理任务封装为标准协议消息并投递到 RabbitMQ
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .rabbitmq_client import RabbitMQClient
from .task_protocol import WatermarkTaskMessage, TaskPriority

logger = logging.getLogger(__name__)


class TaskProducer:
    """
    任务生产者

    职责：
    1. 将 API 请求参数封装为 WatermarkTaskMessage
    2. 投递到 RabbitMQ 对应的队列
    3. 在数据库中创建 pending 状态的任务记录
    """

    def __init__(self, rabbitmq_client: RabbitMQClient):
        self._mq = rabbitmq_client

    async def submit_task(self, msg: WatermarkTaskMessage) -> str:
        """
        提交水印处理任务到消息队列

        Args:
            msg: 标准化任务消息

        Returns:
            task_id
        """
        routing_key = msg.routing_key

        await self._mq.publish(
            routing_key=routing_key,
            body=msg.to_json(),
            priority=msg.priority,
            correlation_id=msg.correlation_id,
            headers={
                "task_id": msg.task_id,
                "file_category": msg.file_category,
                "action": msg.action,
                "retry_count": msg.retry_count,
            },
        )

        logger.info(
            f"任务已投递: task_id={msg.task_id}, "
            f"queue={routing_key}, file={msg.original_filename}"
        )
        return msg.task_id


# ============= 全局单例 =============

_producer: Optional[TaskProducer] = None
_mq_client: Optional[RabbitMQClient] = None


async def init_task_producer(rabbitmq_url: str) -> TaskProducer:
    """初始化全局任务生产者（在 API Gateway 启动时调用）"""
    global _producer, _mq_client

    _mq_client = RabbitMQClient(url=rabbitmq_url)
    await _mq_client.connect()
    _producer = TaskProducer(_mq_client)
    logger.info("TaskProducer 初始化完成")
    return _producer


async def shutdown_task_producer():
    """关闭全局任务生产者"""
    global _mq_client
    if _mq_client:
        await _mq_client.close()


def get_task_producer() -> Optional[TaskProducer]:
    """获取全局任务生产者实例"""
    return _producer
