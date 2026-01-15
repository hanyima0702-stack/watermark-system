"""
基础数据访问对象
提供通用的CRUD操作和数据库连接管理
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Type, TypeVar, Generic
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.exc import SQLAlchemyError
from contextlib import asynccontextmanager
import logging

from ..models.base import Base

T = TypeVar('T', bound=Base)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库连接管理器"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_async_engine(
            database_url,
            echo=False,  # 生产环境设为False
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    @asynccontextmanager
    async def get_session(self):
        """获取数据库会话"""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_tables(self):
        """创建数据表"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """删除数据表"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def close(self):
        """关闭数据库连接"""
        await self.engine.dispose()


class BaseDAO(Generic[T], ABC):
    """基础数据访问对象"""
    
    def __init__(self, db_manager: DatabaseManager, model_class: Type[T]):
        self.db_manager = db_manager
        self.model_class = model_class
    
    @asynccontextmanager
    async def get_session(self):
        """获取数据库会话"""
        async with self.db_manager.get_session() as session:
            yield session
    
    async def create(self, obj: T) -> T:
        """创建对象"""
        try:
            async with self.get_session() as session:
                session.add(obj)
                await session.flush()
                await session.refresh(obj)
                return obj
        except SQLAlchemyError as e:
            logger.error(f"创建{self.model_class.__name__}失败: {e}")
            raise
    
    async def get_by_id(self, obj_id: str, load_relationships: bool = False) -> Optional[T]:
        """根据ID获取对象"""
        try:
            async with self.get_session() as session:
                query = select(self.model_class).where(
                    getattr(self.model_class, self._get_primary_key()) == obj_id
                )
                
                if load_relationships:
                    query = self._add_relationship_loading(query)
                
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"获取{self.model_class.__name__}失败: {e}")
            raise
    
    async def get_all(self, limit: int = 100, offset: int = 0, 
                     load_relationships: bool = False) -> List[T]:
        """获取所有对象"""
        try:
            async with self.get_session() as session:
                query = select(self.model_class).limit(limit).offset(offset)
                
                if load_relationships:
                    query = self._add_relationship_loading(query)
                
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"获取{self.model_class.__name__}列表失败: {e}")
            raise
    
    async def update(self, obj_id: str, update_data: Dict[str, Any]) -> Optional[T]:
        """更新对象"""
        try:
            async with self.get_session() as session:
                query = update(self.model_class).where(
                    getattr(self.model_class, self._get_primary_key()) == obj_id
                ).values(**update_data).returning(self.model_class)
                
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"更新{self.model_class.__name__}失败: {e}")
            raise
    
    async def delete(self, obj_id: str) -> bool:
        """删除对象"""
        try:
            async with self.get_session() as session:
                query = delete(self.model_class).where(
                    getattr(self.model_class, self._get_primary_key()) == obj_id
                )
                result = await session.execute(query)
                return result.rowcount > 0
        except SQLAlchemyError as e:
            logger.error(f"删除{self.model_class.__name__}失败: {e}")
            raise
    
    async def exists(self, obj_id: str) -> bool:
        """检查对象是否存在"""
        try:
            async with self.get_session() as session:
                query = select(func.count()).select_from(
                    select(self.model_class).where(
                        getattr(self.model_class, self._get_primary_key()) == obj_id
                    ).subquery()
                )
                result = await session.execute(query)
                return result.scalar() > 0
        except SQLAlchemyError as e:
            logger.error(f"检查{self.model_class.__name__}存在性失败: {e}")
            raise
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """统计对象数量"""
        try:
            async with self.get_session() as session:
                query = select(func.count()).select_from(self.model_class)
                
                if filters:
                    query = self._apply_filters(query, filters)
                
                result = await session.execute(query)
                return result.scalar()
        except SQLAlchemyError as e:
            logger.error(f"统计{self.model_class.__name__}数量失败: {e}")
            raise
    
    async def find_by_filters(self, filters: Dict[str, Any], 
                             limit: int = 100, offset: int = 0,
                             order_by: Optional[str] = None,
                             load_relationships: bool = False) -> List[T]:
        """根据条件查询对象"""
        try:
            async with self.get_session() as session:
                query = select(self.model_class)
                
                # 应用过滤条件
                query = self._apply_filters(query, filters)
                
                # 应用排序
                if order_by:
                    query = self._apply_ordering(query, order_by)
                
                # 应用分页
                query = query.limit(limit).offset(offset)
                
                # 加载关联对象
                if load_relationships:
                    query = self._add_relationship_loading(query)
                
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"查询{self.model_class.__name__}失败: {e}")
            raise
    
    async def bulk_create(self, objects: List[T]) -> List[T]:
        """批量创建对象"""
        try:
            async with self.get_session() as session:
                session.add_all(objects)
                await session.flush()
                for obj in objects:
                    await session.refresh(obj)
                return objects
        except SQLAlchemyError as e:
            logger.error(f"批量创建{self.model_class.__name__}失败: {e}")
            raise
    
    async def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """批量更新对象"""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    update(self.model_class), updates
                )
                return result.rowcount
        except SQLAlchemyError as e:
            logger.error(f"批量更新{self.model_class.__name__}失败: {e}")
            raise
    
    def _get_primary_key(self) -> str:
        """获取主键字段名"""
        return self.model_class.__table__.primary_key.columns.keys()[0]
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """应用过滤条件"""
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                column = getattr(self.model_class, key)
                if isinstance(value, list):
                    query = query.where(column.in_(value))
                elif isinstance(value, dict):
                    # 支持范围查询等复杂条件
                    if 'gte' in value:
                        query = query.where(column >= value['gte'])
                    if 'lte' in value:
                        query = query.where(column <= value['lte'])
                    if 'gt' in value:
                        query = query.where(column > value['gt'])
                    if 'lt' in value:
                        query = query.where(column < value['lt'])
                    if 'like' in value:
                        query = query.where(column.like(f"%{value['like']}%"))
                else:
                    query = query.where(column == value)
        return query
    
    def _apply_ordering(self, query, order_by: str):
        """应用排序"""
        if order_by.startswith('-'):
            # 降序
            field = order_by[1:]
            if hasattr(self.model_class, field):
                query = query.order_by(getattr(self.model_class, field).desc())
        else:
            # 升序
            if hasattr(self.model_class, order_by):
                query = query.order_by(getattr(self.model_class, order_by))
        return query
    
    def _add_relationship_loading(self, query):
        """添加关联对象加载（子类可重写）"""
        return query
    
    @abstractmethod
    def get_model_class(self) -> Type[T]:
        """获取模型类"""
        return self.model_class