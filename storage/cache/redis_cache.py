"""
Redis缓存服务实现
提供基于Redis的缓存功能
"""

import asyncio
import json
import logging
import pickle
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis缓存服务"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6379)
        self.db = config.get('db', 0)
        self.password = config.get('password')
        self.max_connections = config.get('max_connections', 20)
        self.socket_timeout = config.get('socket_timeout', 5)
        self.socket_connect_timeout = config.get('socket_connect_timeout', 5)
        self.retry_on_timeout = config.get('retry_on_timeout', True)
        self.health_check_interval = config.get('health_check_interval', 30)
        
        self._client: Optional[Redis] = None
        self._connection_pool = None
        
    async def initialize(self):
        """初始化Redis连接"""
        try:
            # 创建连接池
            self._connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
                health_check_interval=self.health_check_interval
            )
            
            # 创建Redis客户端
            self._client = Redis(connection_pool=self._connection_pool)
            
            # 测试连接
            await self._test_connection()
            
            logger.info(f"Redis缓存初始化成功，连接到: {self.host}:{self.port}/{self.db}")
            
        except Exception as e:
            logger.error(f"Redis缓存初始化失败: {e}")
            raise
    
    async def _test_connection(self):
        """测试Redis连接"""
        try:
            await self._client.ping()
        except Exception as e:
            raise ConnectionError(f"无法连接到Redis服务器: {e}")
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None, 
                 serialize: bool = True) -> bool:
        """设置缓存值"""
        try:
            if serialize:
                if isinstance(value, (dict, list, tuple)):
                    # JSON序列化
                    serialized_value = json.dumps(value, ensure_ascii=False)
                else:
                    # Pickle序列化
                    serialized_value = pickle.dumps(value)
            else:
                serialized_value = value
            
            result = await self._client.set(key, serialized_value, ex=expire)
            
            if result:
                logger.debug(f"缓存设置成功: {key}")
            
            return result
            
        except RedisError as e:
            logger.error(f"设置缓存失败: {key}, {e}")
            return False
    
    async def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """获取缓存值"""
        try:
            value = await self._client.get(key)
            
            if value is None:
                return None
            
            if deserialize:
                try:
                    # 尝试JSON反序列化
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    try:
                        # 尝试Pickle反序列化
                        return pickle.loads(value)
                    except (pickle.PickleError, TypeError):
                        # 返回原始字符串
                        return value.decode('utf-8') if isinstance(value, bytes) else value
            else:
                return value.decode('utf-8') if isinstance(value, bytes) else value
                
        except RedisError as e:
            logger.error(f"获取缓存失败: {key}, {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            result = await self._client.delete(key)
            
            if result:
                logger.debug(f"缓存删除成功: {key}")
            
            return result > 0
            
        except RedisError as e:
            logger.error(f"删除缓存失败: {key}, {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            result = await self._client.exists(key)
            return result > 0
        except RedisError as e:
            logger.error(f"检查缓存存在性失败: {key}, {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置缓存过期时间"""
        try:
            result = await self._client.expire(key, seconds)
            return result
        except RedisError as e:
            logger.error(f"设置缓存过期时间失败: {key}, {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """获取缓存剩余生存时间"""
        try:
            return await self._client.ttl(key)
        except RedisError as e:
            logger.error(f"获取缓存TTL失败: {key}, {e}")
            return -1
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的所有键"""
        try:
            keys = await self._client.keys(pattern)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
        except RedisError as e:
            logger.error(f"获取缓存键失败: {pattern}, {e}")
            return []
    
    async def mset(self, mapping: Dict[str, Any], serialize: bool = True) -> bool:
        """批量设置缓存"""
        try:
            if serialize:
                serialized_mapping = {}
                for key, value in mapping.items():
                    if isinstance(value, (dict, list, tuple)):
                        serialized_mapping[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        serialized_mapping[key] = pickle.dumps(value)
            else:
                serialized_mapping = mapping
            
            result = await self._client.mset(serialized_mapping)
            
            if result:
                logger.debug(f"批量缓存设置成功: {len(mapping)} 个键")
            
            return result
            
        except RedisError as e:
            logger.error(f"批量设置缓存失败: {e}")
            return False
    
    async def mget(self, keys: List[str], deserialize: bool = True) -> List[Optional[Any]]:
        """批量获取缓存"""
        try:
            values = await self._client.mget(keys)
            
            if not deserialize:
                return [value.decode('utf-8') if isinstance(value, bytes) and value else value 
                       for value in values]
            
            results = []
            for value in values:
                if value is None:
                    results.append(None)
                    continue
                
                try:
                    # 尝试JSON反序列化
                    results.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    try:
                        # 尝试Pickle反序列化
                        results.append(pickle.loads(value))
                    except (pickle.PickleError, TypeError):
                        # 返回原始字符串
                        results.append(value.decode('utf-8') if isinstance(value, bytes) else value)
            
            return results
            
        except RedisError as e:
            logger.error(f"批量获取缓存失败: {e}")
            return [None] * len(keys)
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """递增计数器"""
        try:
            return await self._client.incrby(key, amount)
        except RedisError as e:
            logger.error(f"递增计数器失败: {key}, {e}")
            return 0
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """递减计数器"""
        try:
            return await self._client.decrby(key, amount)
        except RedisError as e:
            logger.error(f"递减计数器失败: {key}, {e}")
            return 0
    
    async def hset(self, name: str, mapping: Dict[str, Any], serialize: bool = True) -> int:
        """设置哈希表字段"""
        try:
            if serialize:
                serialized_mapping = {}
                for key, value in mapping.items():
                    if isinstance(value, (dict, list, tuple)):
                        serialized_mapping[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        serialized_mapping[key] = str(value)
            else:
                serialized_mapping = {k: str(v) for k, v in mapping.items()}
            
            return await self._client.hset(name, mapping=serialized_mapping)
            
        except RedisError as e:
            logger.error(f"设置哈希表失败: {name}, {e}")
            return 0
    
    async def hget(self, name: str, key: str, deserialize: bool = True) -> Optional[Any]:
        """获取哈希表字段值"""
        try:
            value = await self._client.hget(name, key)
            
            if value is None:
                return None
            
            if deserialize:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value.decode('utf-8') if isinstance(value, bytes) else value
            else:
                return value.decode('utf-8') if isinstance(value, bytes) else value
                
        except RedisError as e:
            logger.error(f"获取哈希表字段失败: {name}.{key}, {e}")
            return None
    
    async def hgetall(self, name: str, deserialize: bool = True) -> Dict[str, Any]:
        """获取哈希表所有字段"""
        try:
            data = await self._client.hgetall(name)
            
            if not data:
                return {}
            
            result = {}
            for key, value in data.items():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                
                if deserialize:
                    try:
                        result[key_str] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key_str] = value.decode('utf-8') if isinstance(value, bytes) else value
                else:
                    result[key_str] = value.decode('utf-8') if isinstance(value, bytes) else value
            
            return result
            
        except RedisError as e:
            logger.error(f"获取哈希表所有字段失败: {name}, {e}")
            return {}
    
    async def hdel(self, name: str, *keys: str) -> int:
        """删除哈希表字段"""
        try:
            return await self._client.hdel(name, *keys)
        except RedisError as e:
            logger.error(f"删除哈希表字段失败: {name}, {e}")
            return 0
    
    async def sadd(self, name: str, *values: Any) -> int:
        """添加集合成员"""
        try:
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list, tuple)):
                    serialized_values.append(json.dumps(value, ensure_ascii=False))
                else:
                    serialized_values.append(str(value))
            
            return await self._client.sadd(name, *serialized_values)
            
        except RedisError as e:
            logger.error(f"添加集合成员失败: {name}, {e}")
            return 0
    
    async def smembers(self, name: str, deserialize: bool = True) -> set:
        """获取集合所有成员"""
        try:
            members = await self._client.smembers(name)
            
            if not deserialize:
                return {member.decode('utf-8') if isinstance(member, bytes) else member 
                       for member in members}
            
            result = set()
            for member in members:
                try:
                    result.add(json.loads(member))
                except (json.JSONDecodeError, TypeError):
                    result.add(member.decode('utf-8') if isinstance(member, bytes) else member)
            
            return result
            
        except RedisError as e:
            logger.error(f"获取集合成员失败: {name}, {e}")
            return set()
    
    async def srem(self, name: str, *values: Any) -> int:
        """删除集合成员"""
        try:
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list, tuple)):
                    serialized_values.append(json.dumps(value, ensure_ascii=False))
                else:
                    serialized_values.append(str(value))
            
            return await self._client.srem(name, *serialized_values)
            
        except RedisError as e:
            logger.error(f"删除集合成员失败: {name}, {e}")
            return 0
    
    async def lpush(self, name: str, *values: Any) -> int:
        """从左侧推入列表"""
        try:
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list, tuple)):
                    serialized_values.append(json.dumps(value, ensure_ascii=False))
                else:
                    serialized_values.append(pickle.dumps(value))
            
            return await self._client.lpush(name, *serialized_values)
            
        except RedisError as e:
            logger.error(f"推入列表失败: {name}, {e}")
            return 0
    
    async def rpop(self, name: str, deserialize: bool = True) -> Optional[Any]:
        """从右侧弹出列表元素"""
        try:
            value = await self._client.rpop(name)
            
            if value is None:
                return None
            
            if deserialize:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    try:
                        return pickle.loads(value)
                    except (pickle.PickleError, TypeError):
                        return value.decode('utf-8') if isinstance(value, bytes) else value
            else:
                return value.decode('utf-8') if isinstance(value, bytes) else value
                
        except RedisError as e:
            logger.error(f"弹出列表元素失败: {name}, {e}")
            return None
    
    async def llen(self, name: str) -> int:
        """获取列表长度"""
        try:
            return await self._client.llen(name)
        except RedisError as e:
            logger.error(f"获取列表长度失败: {name}, {e}")
            return 0
    
    async def flushdb(self) -> bool:
        """清空当前数据库"""
        try:
            result = await self._client.flushdb()
            logger.info("Redis数据库已清空")
            return result
        except RedisError as e:
            logger.error(f"清空数据库失败: {e}")
            return False
    
    async def info(self) -> Dict[str, Any]:
        """获取Redis服务器信息"""
        try:
            info = await self._client.info()
            return info
        except RedisError as e:
            logger.error(f"获取Redis信息失败: {e}")
            return {}
    
    async def ping(self) -> bool:
        """测试连接"""
        try:
            result = await self._client.ping()
            return result
        except RedisError as e:
            logger.error(f"Redis连接测试失败: {e}")
            return False
    
    async def close(self):
        """关闭Redis连接"""
        if self._client:
            await self._client.close()
            self._client = None
        
        if self._connection_pool:
            await self._connection_pool.disconnect()
            self._connection_pool = None
        
        logger.info("Redis连接已关闭")