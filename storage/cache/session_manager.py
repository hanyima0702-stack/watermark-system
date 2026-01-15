"""
会话管理器
提供用户会话管理和认证状态维护功能
"""

import asyncio
import logging
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .cache_manager import CacheManager, CacheNamespace

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """会话数据"""
    session_id: str
    user_id: str
    username: str
    roles: List[str]
    department: str
    created_at: datetime
    last_accessed: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """从字典创建"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class SessionManager:
    """会话管理器"""
    
    def __init__(self, cache_manager: CacheManager, config: Dict[str, Any]):
        self.cache_manager = cache_manager
        self.config = config
        
        # 会话配置
        self.session_expire = config.get('session_expire', 3600)  # 1小时
        self.max_sessions_per_user = config.get('max_sessions_per_user', 5)
        self.session_cleanup_interval = config.get('session_cleanup_interval', 300)  # 5分钟
        self.remember_me_expire = config.get('remember_me_expire', 7 * 24 * 3600)  # 7天
        
        # 会话键前缀
        self.session_prefix = "session"
        self.user_sessions_prefix = "user_sessions"
        
        # 清理任务
        self._cleanup_task = None
    
    async def initialize(self):
        """初始化会话管理器"""
        try:
            # 启动会话清理任务
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
            
            logger.info("会话管理器初始化成功")
            
        except Exception as e:
            logger.error(f"会话管理器初始化失败: {e}")
            raise
    
    def generate_session_id(self) -> str:
        """生成会话ID"""
        return secrets.token_urlsafe(32)
    
    async def create_session(self, user_id: str, username: str, roles: List[str],
                           department: str, ip_address: str, user_agent: str,
                           remember_me: bool = False) -> SessionData:
        """创建会话"""
        try:
            # 生成会话ID
            session_id = self.generate_session_id()
            
            # 创建会话数据
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
                username=username,
                roles=roles,
                department=department,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent,
                is_active=True,
                metadata={}
            )
            
            # 设置过期时间
            expire_time = self.remember_me_expire if remember_me else self.session_expire
            
            # 保存会话数据
            await self.cache_manager.set(
                CacheNamespace.SESSION,
                f"{self.session_prefix}:{session_id}",
                session_data.to_dict(),
                expire_time
            )
            
            # 更新用户会话列表
            await self._add_user_session(user_id, session_id)
            
            # 检查并清理用户的旧会话
            await self._cleanup_user_sessions(user_id)
            
            logger.info(f"会话创建成功: {session_id} for user {user_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """获取会话"""
        try:
            session_key = f"{self.session_prefix}:{session_id}"
            session_dict = await self.cache_manager.get(CacheNamespace.SESSION, session_key)
            
            if not session_dict:
                return None
            
            session_data = SessionData.from_dict(session_dict)
            
            # 检查会话是否活跃
            if not session_data.is_active:
                return None
            
            # 更新最后访问时间
            session_data.last_accessed = datetime.utcnow()
            await self.cache_manager.set(
                CacheNamespace.SESSION,
                session_key,
                session_data.to_dict()
            )
            
            return session_data
            
        except Exception as e:
            logger.error(f"获取会话失败: {session_id}, {e}")
            return None
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """更新会话数据"""
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return False
            
            # 更新字段
            for key, value in updates.items():
                if hasattr(session_data, key):
                    setattr(session_data, key, value)
            
            # 更新元数据
            if 'metadata' in updates:
                if session_data.metadata is None:
                    session_data.metadata = {}
                session_data.metadata.update(updates['metadata'])
            
            # 保存更新
            session_key = f"{self.session_prefix}:{session_id}"
            await self.cache_manager.set(
                CacheNamespace.SESSION,
                session_key,
                session_data.to_dict()
            )
            
            logger.debug(f"会话更新成功: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新会话失败: {session_id}, {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        try:
            # 获取会话数据
            session_data = await self.get_session(session_id)
            if session_data:
                # 从用户会话列表中移除
                await self._remove_user_session(session_data.user_id, session_id)
            
            # 删除会话数据
            session_key = f"{self.session_prefix}:{session_id}"
            result = await self.cache_manager.delete(CacheNamespace.SESSION, session_key)
            
            if result:
                logger.info(f"会话删除成功: {session_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"删除会话失败: {session_id}, {e}")
            return False
    
    async def invalidate_session(self, session_id: str) -> bool:
        """使会话失效（标记为非活跃）"""
        try:
            return await self.update_session(session_id, {'is_active': False})
        except Exception as e:
            logger.error(f"使会话失效失败: {session_id}, {e}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """获取用户的所有会话"""
        try:
            # 获取用户会话ID列表
            user_sessions_key = f"{self.user_sessions_prefix}:{user_id}"
            session_ids = await self.cache_manager.get(CacheNamespace.SESSION, user_sessions_key)
            
            if not session_ids:
                return []
            
            # 获取会话数据
            sessions = []
            for session_id in session_ids:
                session_data = await self.get_session(session_id)
                if session_data:
                    sessions.append(session_data)
            
            return sessions
            
        except Exception as e:
            logger.error(f"获取用户会话失败: {user_id}, {e}")
            return []
    
    async def delete_user_sessions(self, user_id: str, exclude_session_id: Optional[str] = None) -> int:
        """删除用户的所有会话"""
        try:
            sessions = await self.get_user_sessions(user_id)
            deleted_count = 0
            
            for session in sessions:
                if exclude_session_id and session.session_id == exclude_session_id:
                    continue
                
                if await self.delete_session(session.session_id):
                    deleted_count += 1
            
            logger.info(f"删除用户会话: {user_id}, 删除 {deleted_count} 个会话")
            return deleted_count
            
        except Exception as e:
            logger.error(f"删除用户会话失败: {user_id}, {e}")
            return 0
    
    async def validate_session(self, session_id: str, ip_address: str = None,
                             user_agent: str = None) -> Optional[SessionData]:
        """验证会话"""
        try:
            session_data = await self.get_session(session_id)
            
            if not session_data:
                return None
            
            # 检查IP地址（可选）
            if ip_address and self.config.get('check_ip', False):
                if session_data.ip_address != ip_address:
                    logger.warning(f"会话IP地址不匹配: {session_id}")
                    await self.invalidate_session(session_id)
                    return None
            
            # 检查User-Agent（可选）
            if user_agent and self.config.get('check_user_agent', False):
                if session_data.user_agent != user_agent:
                    logger.warning(f"会话User-Agent不匹配: {session_id}")
                    await self.invalidate_session(session_id)
                    return None
            
            return session_data
            
        except Exception as e:
            logger.error(f"验证会话失败: {session_id}, {e}")
            return None
    
    async def refresh_session(self, session_id: str) -> bool:
        """刷新会话（延长过期时间）"""
        try:
            session_key = f"{self.session_prefix}:{session_id}"
            
            # 检查会话是否存在
            if not await self.cache_manager.exists(CacheNamespace.SESSION, session_key):
                return False
            
            # 延长过期时间
            await self.cache_manager.expire(CacheNamespace.SESSION, session_key, self.session_expire)
            
            # 更新最后访问时间
            await self.update_session(session_id, {'last_accessed': datetime.utcnow()})
            
            logger.debug(f"会话刷新成功: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"刷新会话失败: {session_id}, {e}")
            return False
    
    async def _add_user_session(self, user_id: str, session_id: str):
        """添加用户会话"""
        try:
            user_sessions_key = f"{self.user_sessions_prefix}:{user_id}"
            session_ids = await self.cache_manager.get(CacheNamespace.SESSION, user_sessions_key) or []
            
            if session_id not in session_ids:
                session_ids.append(session_id)
                await self.cache_manager.set(
                    CacheNamespace.SESSION,
                    user_sessions_key,
                    session_ids,
                    self.session_expire * 2  # 用户会话列表的过期时间更长
                )
            
        except Exception as e:
            logger.error(f"添加用户会话失败: {user_id}, {e}")
    
    async def _remove_user_session(self, user_id: str, session_id: str):
        """移除用户会话"""
        try:
            user_sessions_key = f"{self.user_sessions_prefix}:{user_id}"
            session_ids = await self.cache_manager.get(CacheNamespace.SESSION, user_sessions_key) or []
            
            if session_id in session_ids:
                session_ids.remove(session_id)
                if session_ids:
                    await self.cache_manager.set(
                        CacheNamespace.SESSION,
                        user_sessions_key,
                        session_ids
                    )
                else:
                    await self.cache_manager.delete(CacheNamespace.SESSION, user_sessions_key)
            
        except Exception as e:
            logger.error(f"移除用户会话失败: {user_id}, {e}")
    
    async def _cleanup_user_sessions(self, user_id: str):
        """清理用户的旧会话"""
        try:
            sessions = await self.get_user_sessions(user_id)
            
            if len(sessions) <= self.max_sessions_per_user:
                return
            
            # 按最后访问时间排序，删除最旧的会话
            sessions.sort(key=lambda s: s.last_accessed)
            sessions_to_delete = sessions[:-self.max_sessions_per_user]
            
            for session in sessions_to_delete:
                await self.delete_session(session.session_id)
            
            logger.info(f"清理用户旧会话: {user_id}, 删除 {len(sessions_to_delete)} 个会话")
            
        except Exception as e:
            logger.error(f"清理用户会话失败: {user_id}, {e}")
    
    async def _cleanup_expired_sessions(self):
        """定期清理过期会话"""
        while True:
            try:
                await asyncio.sleep(self.session_cleanup_interval)
                
                # 获取所有会话键
                session_keys = await self.cache_manager.keys(CacheNamespace.SESSION, f"{self.session_prefix}:*")
                
                cleaned_count = 0
                for session_key in session_keys:
                    # 检查TTL
                    ttl = await self.cache_manager.ttl(CacheNamespace.SESSION, session_key)
                    if ttl <= 0:  # 已过期或不存在
                        await self.cache_manager.delete(CacheNamespace.SESSION, session_key)
                        cleaned_count += 1
                
                if cleaned_count > 0:
                    logger.info(f"清理过期会话: {cleaned_count} 个")
                
            except asyncio.CancelledError:
                logger.info("会话清理任务已取消")
                break
            except Exception as e:
                logger.error(f"清理过期会话失败: {e}")
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        try:
            # 获取所有会话键
            session_keys = await self.cache_manager.keys(CacheNamespace.SESSION, f"{self.session_prefix}:*")
            total_sessions = len(session_keys)
            
            # 统计活跃会话
            active_sessions = 0
            user_sessions = {}
            
            for session_key in session_keys:
                session_dict = await self.cache_manager.get(CacheNamespace.SESSION, session_key)
                if session_dict and session_dict.get('is_active', True):
                    active_sessions += 1
                    user_id = session_dict.get('user_id')
                    if user_id:
                        user_sessions[user_id] = user_sessions.get(user_id, 0) + 1
            
            return {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'inactive_sessions': total_sessions - active_sessions,
                'unique_users': len(user_sessions),
                'avg_sessions_per_user': sum(user_sessions.values()) / len(user_sessions) if user_sessions else 0
            }
            
        except Exception as e:
            logger.error(f"获取会话统计失败: {e}")
            return {}
    
    async def close(self):
        """关闭会话管理器"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        logger.info("会话管理器已关闭")