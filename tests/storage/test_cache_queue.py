"""
缓存和消息队列服务单元测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from storage.cache import CacheManager, SessionManager, CacheNamespace
from storage.cache.session_manager import SessionData
from storage.queue import TaskManager, MessageBroker, QueueMonitor
from storage.queue.task_manager import TaskInfo, TaskPriority, TaskStatus


class TestCacheManager:
    """缓存管理器测试"""
    
    @pytest.fixture
    def cache_config(self):
        return {
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'default_expire': 3600,
            'key_prefix': 'test'
        }
    
    @pytest.fixture
    def cache_manager(self, cache_config):
        return CacheManager(cache_config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, cache_manager):
        """测试缓存管理器初始化"""
        with patch('storage.cache.redis_cache.RedisCache.initialize', new_callable=AsyncMock):
            await cache_manager.initialize()
            assert cache_manager._redis_cache is not None
    
    @pytest.mark.asyncio
    async def test_set_get(self, cache_manager):
        """测试设置和获取缓存"""
        # 模拟Redis缓存
        mock_redis = Mock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value={'test': 'data'})
        cache_manager._redis_cache = mock_redis
        
        # 测试设置缓存
        result = await cache_manager.set(CacheNamespace.USER, 'test_key', {'test': 'data'})
        assert result is True
        
        # 测试获取缓存
        data = await cache_manager.get(CacheNamespace.USER, 'test_key')
        assert data == {'test': 'data'}
    
    @pytest.mark.asyncio
    async def test_delete(self, cache_manager):
        """测试删除缓存"""
        mock_redis = Mock()
        mock_redis.delete = AsyncMock(return_value=True)
        cache_manager._redis_cache = mock_redis
        
        result = await cache_manager.delete(CacheNamespace.USER, 'test_key')
        assert result is True
    
    @pytest.mark.asyncio
    async def test_user_cache_methods(self, cache_manager):
        """测试用户缓存方法"""
        mock_redis = Mock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value={'user_id': 'test_user'})
        mock_redis.delete = AsyncMock(return_value=True)
        cache_manager._redis_cache = mock_redis
        
        # 测试设置用户缓存
        result = await cache_manager.set_user_cache('test_user', {'user_id': 'test_user'})
        assert result is True
        
        # 测试获取用户缓存
        data = await cache_manager.get_user_cache('test_user')
        assert data == {'user_id': 'test_user'}
        
        # 测试删除用户缓存
        result = await cache_manager.delete_user_cache('test_user')
        assert result is True
    
    def test_build_key(self, cache_manager):
        """测试构建缓存键"""
        key = cache_manager._build_key(CacheNamespace.USER, 'test_key')
        assert key == 'test:user:test_key'
    
    def test_get_expire_time(self, cache_manager):
        """测试获取过期时间"""
        # 使用默认过期时间
        expire_time = cache_manager._get_expire_time(CacheNamespace.USER)
        assert expire_time == cache_manager.namespace_expires[CacheNamespace.USER]
        
        # 使用自定义过期时间
        expire_time = cache_manager._get_expire_time(CacheNamespace.USER, 1800)
        assert expire_time == 1800


class TestSessionManager:
    """会话管理器测试"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        cache_manager = Mock()
        cache_manager.set = AsyncMock(return_value=True)
        cache_manager.get = AsyncMock()
        cache_manager.delete = AsyncMock(return_value=True)
        return cache_manager
    
    @pytest.fixture
    def session_config(self):
        return {
            'session_expire': 3600,
            'max_sessions_per_user': 5,
            'session_cleanup_interval': 300
        }
    
    @pytest.fixture
    def session_manager(self, mock_cache_manager, session_config):
        return SessionManager(mock_cache_manager, session_config)
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager, mock_cache_manager):
        """测试创建会话"""
        # 模拟缓存操作
        mock_cache_manager.get.return_value = []  # 用户会话列表为空
        
        session_data = await session_manager.create_session(
            user_id='test_user',
            username='testuser',
            roles=['user'],
            department='IT',
            ip_address='127.0.0.1',
            user_agent='test-agent'
        )
        
        assert isinstance(session_data, SessionData)
        assert session_data.user_id == 'test_user'
        assert session_data.username == 'testuser'
        assert session_data.is_active is True
    
    @pytest.mark.asyncio
    async def test_get_session(self, session_manager, mock_cache_manager):
        """测试获取会话"""
        # 模拟会话数据
        session_dict = {
            'session_id': 'test_session',
            'user_id': 'test_user',
            'username': 'testuser',
            'roles': ['user'],
            'department': 'IT',
            'created_at': datetime.utcnow().isoformat(),
            'last_accessed': datetime.utcnow().isoformat(),
            'ip_address': '127.0.0.1',
            'user_agent': 'test-agent',
            'is_active': True,
            'metadata': {}
        }
        mock_cache_manager.get.return_value = session_dict
        
        session_data = await session_manager.get_session('test_session')
        
        assert isinstance(session_data, SessionData)
        assert session_data.session_id == 'test_session'
        assert session_data.user_id == 'test_user'
    
    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager, mock_cache_manager):
        """测试删除会话"""
        # 模拟获取会话
        session_dict = {
            'session_id': 'test_session',
            'user_id': 'test_user',
            'username': 'testuser',
            'roles': ['user'],
            'department': 'IT',
            'created_at': datetime.utcnow().isoformat(),
            'last_accessed': datetime.utcnow().isoformat(),
            'ip_address': '127.0.0.1',
            'user_agent': 'test-agent',
            'is_active': True,
            'metadata': {}
        }
        mock_cache_manager.get.return_value = session_dict
        
        result = await session_manager.delete_session('test_session')
        assert result is True
    
    def test_generate_session_id(self, session_manager):
        """测试生成会话ID"""
        session_id = session_manager.generate_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) > 0


class TestTaskManager:
    """任务管理器测试"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        cache_manager = Mock()
        cache_manager.set = AsyncMock(return_value=True)
        cache_manager.get = AsyncMock()
        cache_manager.delete = AsyncMock(return_value=True)
        cache_manager.keys = AsyncMock(return_value=[])
        return cache_manager
    
    @pytest.fixture
    def task_config(self):
        return {
            'default_queue': 'default',
            'task_timeout': 3600,
            'max_retries': 3,
            'retry_delay': 60
        }
    
    @pytest.fixture
    def task_manager(self, mock_cache_manager, task_config):
        with patch('storage.queue.task_manager.get_celery_app') as mock_celery:
            mock_app = Mock()
            mock_app.send_task.return_value = Mock(id='test_task_id')
            mock_celery.return_value = mock_app
            
            manager = TaskManager(mock_cache_manager, task_config)
            return manager
    
    @pytest.mark.asyncio
    async def test_submit_task(self, task_manager, mock_cache_manager):
        """测试提交任务"""
        task_id = await task_manager.submit_task(
            task_name='test.task',
            args=['arg1', 'arg2'],
            kwargs={'key': 'value'},
            priority=TaskPriority.HIGH
        )
        
        assert task_id == 'test_task_id'
        mock_cache_manager.set.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_task_info(self, task_manager, mock_cache_manager):
        """测试获取任务信息"""
        # 模拟任务信息
        task_dict = {
            'task_id': 'test_task_id',
            'task_name': 'test.task',
            'queue': 'default',
            'priority': TaskPriority.NORMAL.value,
            'args': [],
            'kwargs': {},
            'status': TaskStatus.PENDING.value,
            'created_at': datetime.utcnow().isoformat(),
            'retry_count': 0,
            'max_retries': 3
        }
        mock_cache_manager.get.return_value = task_dict
        
        with patch.object(task_manager, 'get_task_status', return_value=TaskStatus.PENDING):
            task_info = await task_manager.get_task_info('test_task_id')
            
            assert isinstance(task_info, TaskInfo)
            assert task_info.task_id == 'test_task_id'
            assert task_info.task_name == 'test.task'
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, task_manager, mock_cache_manager):
        """测试取消任务"""
        # 模拟任务信息
        task_dict = {
            'task_id': 'test_task_id',
            'task_name': 'test.task',
            'queue': 'default',
            'priority': TaskPriority.NORMAL.value,
            'args': [],
            'kwargs': {},
            'status': TaskStatus.PENDING.value,
            'created_at': datetime.utcnow().isoformat(),
            'retry_count': 0,
            'max_retries': 3
        }
        mock_cache_manager.get.return_value = task_dict
        
        with patch.object(task_manager, 'get_task_info') as mock_get_info:
            mock_get_info.return_value = TaskInfo.from_dict(task_dict)
            
            result = await task_manager.cancel_task('test_task_id')
            assert result is True


class TestMessageBroker:
    """消息代理测试"""
    
    @pytest.fixture
    def broker_config(self):
        return {
            'host': 'localhost',
            'port': 6379,
            'db': 3
        }
    
    @pytest.fixture
    def message_broker(self, broker_config):
        return MessageBroker(broker_config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, message_broker):
        """测试消息代理初始化"""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = Mock()
            mock_redis.return_value = mock_client
            
            await message_broker.initialize()
            assert message_broker._client is not None
    
    @pytest.mark.asyncio
    async def test_publish(self, message_broker):
        """测试发布消息"""
        mock_client = Mock()
        mock_client.publish = AsyncMock(return_value=1)
        message_broker._client = mock_client
        
        result = await message_broker.publish('test_topic', {'data': 'test'})
        assert result is True
        mock_client.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_subscribe(self, message_broker):
        """测试订阅主题"""
        mock_pubsub = Mock()
        mock_pubsub.subscribe = AsyncMock()
        message_broker._pubsub = mock_pubsub
        
        def test_callback(message):
            pass
        
        with patch.object(message_broker, '_start_listening', new_callable=AsyncMock):
            await message_broker.subscribe('test_topic', test_callback)
            
            assert 'test_topic' in message_broker._subscribers
            assert test_callback in message_broker._subscribers['test_topic']


if __name__ == "__main__":
    pytest.main([__file__])