"""
缓存管理器
提供统一的缓存接口和管理功能
"""

import asyncio
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from enum import Enum

from .redis_cache import RedisCache

logger = logging.getLogger(__name__)


class CacheNamespace(Enum):
    """缓存命名空间"""
    USER = "user"
    FILE = "file"
    CONFIG = "config"
    TASK = "task"
    SESSION = "session"
    TEMP = "temp"
    STATS = "stats"


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_config: Dict[str, Any]):
        self.cache_config = cache_config
        self.default_expire = cache_config.get('default_expire', 3600)  # 1小时
        self.key_prefix = cache_config.get('key_prefix', 'watermark')
        
        # 不同命名空间的默认过期时间
        self.namespace_expires = {
            CacheNamespace.USER: cache_config.get('user_expire', 1800),      # 30分钟
            CacheNamespace.FILE: cache_config.get('file_expire', 3600),      # 1小时
            CacheNamespace.CONFIG: cache_config.get('config_expire', 7200),  # 2小时
            CacheNamespace.TASK: cache_config.get('task_expire', 1800),      # 30分钟
            CacheNamespace.SESSION: cache_config.get('session_expire', 3600), # 1小时
            CacheNamespace.TEMP: cache_config.get('temp_expire', 300),       # 5分钟
            CacheNamespace.STATS: cache_config.get('stats_expire', 600),     # 10分钟
        }
        
        self._redis_cache = None
    
    async def initialize(self):
        """初始化缓存管理器"""
        try:
            # 初始化Redis缓存
            redis_config = self.cache_config.get('redis', {})
            self._redis_cache = RedisCache(redis_config)
            await self._redis_cache.initialize()
            
            logger.info("缓存管理器初始化成功")
            
        except Exception as e:
            logger.error(f"缓存管理器初始化失败: {e}")
            raise
    
    def _build_key(self, namespace: Union[CacheNamespace, str], key: str) -> str:
        """构建缓存键"""
        if isinstance(namespace, CacheNamespace):
            namespace = namespace.value
        
        return f"{self.key_prefix}:{namespace}:{key}"
    
    def _get_expire_time(self, namespace: Union[CacheNamespace, str], 
                        expire: Optional[int] = None) -> int:
        """获取过期时间"""
        if expire is not None:
            return expire
        
        if isinstance(namespace, str):
            namespace = CacheNamespace(namespace)
        
        return self.namespace_expires.get(namespace, self.default_expire)
    
    async def set(self, namespace: Union[CacheNamespace, str], key: str, 
                 value: Any, expire: Optional[int] = None) -> bool:
        """设置缓存"""
        try:
            cache_key = self._build_key(namespace, key)
            expire_time = self._get_expire_time(namespace, expire)
            
            result = await self._redis_cache.set(cache_key, value, expire_time)
            
            if result:
                logger.debug(f"缓存设置成功: {namespace}.{key}")
            
            return result
            
        except Exception as e:
            logger.error(f"设置缓存失败: {namespace}.{key}, {e}")
            return False
    
    async def get(self, namespace: Union[CacheNamespace, str], key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            cache_key = self._build_key(namespace, key)
            return await self._redis_cache.get(cache_key)
            
        except Exception as e:
            logger.error(f"获取缓存失败: {namespace}.{key}, {e}")
            return None
    
    async def delete(self, namespace: Union[CacheNamespace, str], key: str) -> bool:
        """删除缓存"""
        try:
            cache_key = self._build_key(namespace, key)
            result = await self._redis_cache.delete(cache_key)
            
            if result:
                logger.debug(f"缓存删除成功: {namespace}.{key}")
            
            return result
            
        except Exception as e:
            logger.error(f"删除缓存失败: {namespace}.{key}, {e}")
            return False
    
    async def exists(self, namespace: Union[CacheNamespace, str], key: str) -> bool:
        """检查缓存是否存在"""
        try:
            cache_key = self._build_key(namespace, key)
            return await self._redis_cache.exists(cache_key)
        except Exception as e:
            logger.error(f"检查缓存存在性失败: {namespace}.{key}, {e}")
            return False
    
    async def expire(self, namespace: Union[CacheNamespace, str], key: str, 
                    seconds: int) -> bool:
        """设置缓存过期时间"""
        try:
            cache_key = self._build_key(namespace, key)
            return await self._redis_cache.expire(cache_key, seconds)
        except Exception as e:
            logger.error(f"设置缓存过期时间失败: {namespace}.{key}, {e}")
            return False
    
    async def ttl(self, namespace: Union[CacheNamespace, str], key: str) -> int:
        """获取缓存剩余生存时间"""
        try:
            cache_key = self._build_key(namespace, key)
            return await self._redis_cache.ttl(cache_key)
        except Exception as e:
            logger.error(f"获取缓存TTL失败: {namespace}.{key}, {e}")
            return -1
    
    async def keys(self, namespace: Union[CacheNamespace, str], 
                  pattern: str = "*") -> List[str]:
        """获取命名空间下匹配模式的所有键"""
        try:
            cache_pattern = self._build_key(namespace, pattern)
            keys = await self._redis_cache.keys(cache_pattern)
            
            # 移除前缀，只返回实际的键名
            prefix = self._build_key(namespace, "")
            return [key.replace(prefix, "") for key in keys]
            
        except Exception as e:
            logger.error(f"获取缓存键失败: {namespace}.{pattern}, {e}")
            return []
    
    async def clear_namespace(self, namespace: Union[CacheNamespace, str]) -> int:
        """清空命名空间下的所有缓存"""
        try:
            keys = await self.keys(namespace)
            if not keys:
                return 0
            
            deleted_count = 0
            for key in keys:
                if await self.delete(namespace, key):
                    deleted_count += 1
            
            logger.info(f"清空命名空间缓存: {namespace}, 删除 {deleted_count} 个键")
            return deleted_count
            
        except Exception as e:
            logger.error(f"清空命名空间缓存失败: {namespace}, {e}")
            return 0
    
    # 用户相关缓存方法
    async def set_user_cache(self, user_id: str, data: Dict[str, Any], 
                           expire: Optional[int] = None) -> bool:
        """设置用户缓存"""
        return await self.set(CacheNamespace.USER, user_id, data, expire)
    
    async def get_user_cache(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户缓存"""
        return await self.get(CacheNamespace.USER, user_id)
    
    async def delete_user_cache(self, user_id: str) -> bool:
        """删除用户缓存"""
        return await self.delete(CacheNamespace.USER, user_id)
    
    # 文件相关缓存方法
    async def set_file_cache(self, file_id: str, metadata: Dict[str, Any], 
                           expire: Optional[int] = None) -> bool:
        """设置文件缓存"""
        return await self.set(CacheNamespace.FILE, file_id, metadata, expire)
    
    async def get_file_cache(self, file_id: str) -> Optional[Dict[str, Any]]:
        """获取文件缓存"""
        return await self.get(CacheNamespace.FILE, file_id)
    
    async def delete_file_cache(self, file_id: str) -> bool:
        """删除文件缓存"""
        return await self.delete(CacheNamespace.FILE, file_id)
    
    # 配置相关缓存方法
    async def set_config_cache(self, config_key: str, config_value: Any, 
                             expire: Optional[int] = None) -> bool:
        """设置配置缓存"""
        return await self.set(CacheNamespace.CONFIG, config_key, config_value, expire)
    
    async def get_config_cache(self, config_key: str) -> Optional[Any]:
        """获取配置缓存"""
        return await self.get(CacheNamespace.CONFIG, config_key)
    
    async def delete_config_cache(self, config_key: str) -> bool:
        """删除配置缓存"""
        return await self.delete(CacheNamespace.CONFIG, config_key)
    
    # 任务相关缓存方法
    async def set_task_cache(self, task_id: str, task_data: Dict[str, Any], 
                           expire: Optional[int] = None) -> bool:
        """设置任务缓存"""
        return await self.set(CacheNamespace.TASK, task_id, task_data, expire)
    
    async def get_task_cache(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务缓存"""
        return await self.get(CacheNamespace.TASK, task_id)
    
    async def delete_task_cache(self, task_id: str) -> bool:
        """删除任务缓存"""
        return await self.delete(CacheNamespace.TASK, task_id)
    
    # 统计相关缓存方法
    async def set_stats_cache(self, stats_key: str, stats_data: Dict[str, Any], 
                            expire: Optional[int] = None) -> bool:
        """设置统计缓存"""
        return await self.set(CacheNamespace.STATS, stats_key, stats_data, expire)
    
    async def get_stats_cache(self, stats_key: str) -> Optional[Dict[str, Any]]:
        """获取统计缓存"""
        return await self.get(CacheNamespace.STATS, stats_key)
    
    # 计数器方法
    async def incr_counter(self, namespace: Union[CacheNamespace, str], 
                          key: str, amount: int = 1) -> int:
        """递增计数器"""
        try:
            cache_key = self._build_key(namespace, key)
            return await self._redis_cache.incr(cache_key, amount)
        except Exception as e:
            logger.error(f"递增计数器失败: {namespace}.{key}, {e}")
            return 0
    
    async def decr_counter(self, namespace: Union[CacheNamespace, str], 
                          key: str, amount: int = 1) -> int:
        """递减计数器"""
        try:
            cache_key = self._build_key(namespace, key)
            return await self._redis_cache.decr(cache_key, amount)
        except Exception as e:
            logger.error(f"递减计数器失败: {namespace}.{key}, {e}")
            return 0
    
    # 哈希表方法
    async def hset(self, namespace: Union[CacheNamespace, str], name: str, 
                  mapping: Dict[str, Any]) -> int:
        """设置哈希表字段"""
        try:
            cache_key = self._build_key(namespace, name)
            return await self._redis_cache.hset(cache_key, mapping)
        except Exception as e:
            logger.error(f"设置哈希表失败: {namespace}.{name}, {e}")
            return 0
    
    async def hget(self, namespace: Union[CacheNamespace, str], name: str, 
                  key: str) -> Optional[Any]:
        """获取哈希表字段值"""
        try:
            cache_key = self._build_key(namespace, name)
            return await self._redis_cache.hget(cache_key, key)
        except Exception as e:
            logger.error(f"获取哈希表字段失败: {namespace}.{name}.{key}, {e}")
            return None
    
    async def hgetall(self, namespace: Union[CacheNamespace, str], 
                     name: str) -> Dict[str, Any]:
        """获取哈希表所有字段"""
        try:
            cache_key = self._build_key(namespace, name)
            return await self._redis_cache.hgetall(cache_key)
        except Exception as e:
            logger.error(f"获取哈希表所有字段失败: {namespace}.{name}, {e}")
            return {}
    
    async def hdel(self, namespace: Union[CacheNamespace, str], name: str, 
                  *keys: str) -> int:
        """删除哈希表字段"""
        try:
            cache_key = self._build_key(namespace, name)
            return await self._redis_cache.hdel(cache_key, *keys)
        except Exception as e:
            logger.error(f"删除哈希表字段失败: {namespace}.{name}, {e}")
            return 0
    
    # 临时数据方法
    async def set_temp_data(self, key: str, data: Any, expire: int = 300) -> bool:
        """设置临时数据（默认5分钟过期）"""
        return await self.set(CacheNamespace.TEMP, key, data, expire)
    
    async def get_temp_data(self, key: str) -> Optional[Any]:
        """获取临时数据"""
        return await self.get(CacheNamespace.TEMP, key)
    
    async def delete_temp_data(self, key: str) -> bool:
        """删除临时数据"""
        return await self.delete(CacheNamespace.TEMP, key)
    
    # 缓存统计方法
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            redis_info = await self._redis_cache.info()
            
            # 统计各命名空间的键数量
            namespace_stats = {}
            for namespace in CacheNamespace:
                keys = await self.keys(namespace)
                namespace_stats[namespace.value] = len(keys)
            
            return {
                'redis_info': {
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'used_memory': redis_info.get('used_memory', 0),
                    'used_memory_human': redis_info.get('used_memory_human', '0B'),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0)
                },
                'namespace_stats': namespace_stats,
                'total_keys': sum(namespace_stats.values())
            }
            
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """缓存健康检查"""
        try:
            return await self._redis_cache.ping()
        except Exception as e:
            logger.error(f"缓存健康检查失败: {e}")
            return False
    
    async def close(self):
        """关闭缓存连接"""
        if self._redis_cache:
            await self._redis_cache.close()
            self._redis_cache = None
        
        logger.info("缓存管理器已关闭")