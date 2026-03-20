"""
RabbitMQ 连接管理
提供异步连接池、Exchange/Queue 声明、消息发布与消费的底层封装
"""

import asyncio
import logging
from typing import Callable, Optional, Dict, Any

import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractRobustConnection, AbstractRobustChannel

from .task_protocol import (
    EXCHANGE_NAME,
    DEAD_LETTER_EXCHANGE,
    QUEUE_ROUTING,
    FileCategory,
)

logger = logging.getLogger(__name__)

# 所有队列定义
ALL_QUEUES: Dict[str, Dict[str, Any]] = {
    "watermark.image": {"durable": True, "prefetch": 4},
    "watermark.document": {"durable": True, "prefetch": 2},
    "watermark.video": {"durable": True, "prefetch": 1},
    "watermark.audio": {"durable": True, "prefetch": 2},
    "watermark.default": {"durable": True, "prefetch": 2},
}


class RabbitMQClient:
    """
    RabbitMQ 异步客户端

    负责：
    - 连接管理与自动重连
    - Exchange / Queue 拓扑声明
    - 消息发布（带优先级和持久化）
    - 消费者注册（竞争消费者模式）
    """

    def __init__(self, url: str = "amqp://guest:guest@localhost:5672/"):
        self._url = url
        self._connection: Optional[AbstractRobustConnection] = None
        self._channel: Optional[AbstractRobustChannel] = None
        self._exchange: Optional[aio_pika.Exchange] = None
        self._dlx_exchange: Optional[aio_pika.Exchange] = None

    async def connect(self):
        """建立连接并声明拓扑"""
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()

        # 声明主 Exchange (topic 类型，支持灵活路由)
        self._exchange = await self._channel.declare_exchange(
            EXCHANGE_NAME,
            ExchangeType.TOPIC,
            durable=True,
        )

        # 声明死信 Exchange
        self._dlx_exchange = await self._channel.declare_exchange(
            DEAD_LETTER_EXCHANGE,
            ExchangeType.TOPIC,
            durable=True,
        )

        # 声明所有队列并绑定
        for queue_name, opts in ALL_QUEUES.items():
            # 死信队列
            dlq_name = f"{queue_name}.dlq"
            dlq = await self._channel.declare_queue(dlq_name, durable=True)
            await dlq.bind(self._dlx_exchange, routing_key=queue_name)

            # 主队列（带死信路由和优先级支持）
            queue = await self._channel.declare_queue(
                queue_name,
                durable=opts["durable"],
                arguments={
                    "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE,
                    "x-dead-letter-routing-key": queue_name,
                    "x-max-priority": 10,
                    "x-message-ttl": 3600000,  # 1h TTL
                },
            )
            await queue.bind(self._exchange, routing_key=queue_name)

        logger.info("RabbitMQ 拓扑声明完成")

    async def publish(
        self,
        routing_key: str,
        body: str,
        priority: int = 5,
        correlation_id: str = "",
        headers: Optional[Dict[str, Any]] = None,
    ):
        """
        发布消息到指定路由

        Args:
            routing_key: 路由键 (如 watermark.video)
            body: JSON 消息体
            priority: 优先级 0-10
            correlation_id: 关联 ID，用于追踪
            headers: 自定义消息头
        """
        if not self._exchange:
            raise RuntimeError("RabbitMQ 未连接，请先调用 connect()")

        message = Message(
            body=body.encode("utf-8"),
            delivery_mode=DeliveryMode.PERSISTENT,
            priority=min(priority, 10),
            correlation_id=correlation_id,
            content_type="application/json",
            headers=headers or {},
        )
        await self._exchange.publish(message, routing_key=routing_key)
        logger.info(f"消息已发布: routing_key={routing_key}, correlation_id={correlation_id}")

    async def consume(
        self,
        queue_name: str,
        callback: Callable,
        prefetch_count: int = None,
    ):
        """
        注册消费者（竞争消费者模式）

        多个 Worker 实例监听同一队列时，RabbitMQ 会自动做负载均衡分发。

        Args:
            queue_name: 队列名称
            callback: 消息处理回调 async def handler(message: aio_pika.IncomingMessage)
            prefetch_count: 预取数量，控制并发
        """
        if not self._channel:
            raise RuntimeError("RabbitMQ 未连接，请先调用 connect()")

        prefetch = prefetch_count or ALL_QUEUES.get(queue_name, {}).get("prefetch", 2)
        await self._channel.set_qos(prefetch_count=prefetch)

        queue = await self._channel.declare_queue(
            queue_name,
            durable=True,
            passive=True,  # 不重新创建，只获取引用
        )
        await queue.consume(callback)
        logger.info(f"消费者已注册: queue={queue_name}, prefetch={prefetch}")

    async def close(self):
        """关闭连接"""
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            logger.info("RabbitMQ 连接已关闭")
