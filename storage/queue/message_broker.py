"""
消息代理
提供发布/订阅消息传递功能
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime
from enum import Enum

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型"""
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    FILE_UPLOADED = "file_uploaded"
    FILE_PROCESSED = "file_processed"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    SYSTEM_ALERT = "system_alert"


class MessageBroker:
    """消息代理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6379)
        self.db = config.get('db', 3)
        self.password = config.get('password')
        
        self._client: Optional[Redis] = None
        self._pubsub = None
        self._subscribers: Dict[str, List[Callable]] = {}
        self._running = False
        self._listen_task = None
    
    async def initialize(self):
        """初始化消息代理"""
        try:
            # 创建Redis客户端
            self._client = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            
            # 创建发布订阅对象
            self._pubsub = self._client.pubsub()
            
            # 测试连接
            await self._client.ping()
            
            logger.info(f"消息代理初始化成功，连接到: {self.host}:{self.port}/{self.db}")
            
        except Exception as e:
            logger.error(f"消息代理初始化失败: {e}")
            raise    
  
  async def publish(self, topic: str, message: Dict[str, Any], 
                     message_type: MessageType = None) -> bool:
        """发布消息"""
        try:
            # 构建消息
            msg_data = {
                'type': message_type.value if message_type else 'custom',
                'timestamp': datetime.utcnow().isoformat(),
                'data': message
            }
            
            # 发布消息
            result = await self._client.publish(topic, json.dumps(msg_data))
            
            logger.debug(f"消息发布成功: {topic}, 订阅者数量: {result}")
            return result > 0
            
        except Exception as e:
            logger.error(f"发布消息失败: {topic}, {e}")
            return False
    
    async def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None]):
        """订阅主题"""
        try:
            # 添加回调函数
            if topic not in self._subscribers:
                self._subscribers[topic] = []
                # 订阅主题
                await self._pubsub.subscribe(topic)
            
            self._subscribers[topic].append(callback)
            
            # 启动监听任务
            if not self._running:
                await self._start_listening()
            
            logger.info(f"订阅主题成功: {topic}")
            
        except Exception as e:
            logger.error(f"订阅主题失败: {topic}, {e}")
            raise
    
    async def unsubscribe(self, topic: str, callback: Callable = None):
        """取消订阅"""
        try:
            if topic not in self._subscribers:
                return
            
            if callback:
                # 移除特定回调
                if callback in self._subscribers[topic]:
                    self._subscribers[topic].remove(callback)
            else:
                # 移除所有回调
                self._subscribers[topic].clear()
            
            # 如果没有回调了，取消订阅
            if not self._subscribers[topic]:
                await self._pubsub.unsubscribe(topic)
                del self._subscribers[topic]
            
            logger.info(f"取消订阅成功: {topic}")
            
        except Exception as e:
            logger.error(f"取消订阅失败: {topic}, {e}")
    
    async def _start_listening(self):
        """开始监听消息"""
        if self._running:
            return
        
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_messages())
        logger.info("消息监听已启动")
    
    async def _listen_messages(self):
        """监听消息循环"""
        try:
            async for message in self._pubsub.listen():
                if message['type'] == 'message':
                    await self._handle_message(message)
        except asyncio.CancelledError:
            logger.info("消息监听已停止")
        except Exception as e:
            logger.error(f"消息监听错误: {e}")
    
    async def _handle_message(self, message):
        """处理接收到的消息"""
        try:
            topic = message['channel']
            data = json.loads(message['data'])
            
            # 调用订阅者回调
            if topic in self._subscribers:
                for callback in self._subscribers[topic]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    except Exception as e:
                        logger.error(f"回调函数执行失败: {topic}, {e}")
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
    
    async def close(self):
        """关闭消息代理"""
        self._running = False
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub:
            await self._pubsub.close()
        
        if self._client:
            await self._client.close()
        
        logger.info("消息代理已关闭")